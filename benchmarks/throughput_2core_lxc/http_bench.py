"""
HTTP throughput bench for yantrikdb-server on 2-core LXC.

Hits /v1/remember (single) and /v1/remember/batch (batched) concurrently
and reports throughput, p50/p99 latency, and 503 rate.

Usage:
    python3 http_bench.py \
        --url http://127.0.0.1:7438 \
        --token "$TOKEN" \
        --mode single --concurrency 32 --duration 30
    python3 http_bench.py \
        --url http://127.0.0.1:7438 \
        --token "$TOKEN" \
        --mode batch --batch-size 50 --concurrency 8 --duration 30
"""

import argparse
import asyncio
import json
import math
import time
from statistics import median

import aiohttp


def precomputed_embedding(dim, seed):
    raw = [(seed + i) * 0.1 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1e-9
    return [x / norm for x in raw]


def pcts(durs_ms):
    if not durs_ms:
        return 0.0, 0.0, 0.0
    s = sorted(durs_ms)
    n = len(s)

    def p(q):
        return s[min(int(n * q), n - 1)]

    return p(0.50), p(0.95), p(0.99)


def gen_text(client_id, seq):
    return f"bench client {client_id} seq {seq} payload " + ("x" * 60)


async def worker_single(session, url, token, client_id, stop_evt, samples, counters, embed_dim):
    headers = {"Authorization": f"Bearer {token}"}
    seq = 0
    while not stop_evt.is_set():
        body = {
            "text": gen_text(client_id, seq),
            "importance": 0.5,
            "domain": "bench",
            "memory_type": "episodic",
        }
        if embed_dim:
            body["embedding"] = precomputed_embedding(embed_dim, client_id * 1_000_000 + seq)
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/remember",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                _ = await r.read()
                dur_ms = (time.perf_counter() - t0) * 1000
                samples.append(dur_ms)
                if r.status == 200:
                    counters["ok"] += 1
                elif r.status == 503:
                    counters["503"] += 1
                else:
                    counters[f"http_{r.status}"] = counters.get(f"http_{r.status}", 0) + 1
        except Exception as e:
            counters["err"] = counters.get("err", 0) + 1
            counters["last_err"] = str(e)[:100]
        seq += 1


async def worker_batch(session, url, token, client_id, batch_size, stop_evt, samples, counters, embed_dim):
    headers = {"Authorization": f"Bearer {token}"}
    seq = 0
    while not stop_evt.is_set():
        memories = []
        for i in range(batch_size):
            m = {
                "text": gen_text(client_id, seq + i),
                "importance": 0.5,
                "domain": "bench",
                "memory_type": "episodic",
            }
            if embed_dim:
                m["embedding"] = precomputed_embedding(embed_dim, client_id * 1_000_000 + seq + i)
            memories.append(m)
        body = {"memories": memories}
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/remember/batch",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                _ = await r.read()
                dur_ms = (time.perf_counter() - t0) * 1000
                samples.append((dur_ms, batch_size))
                if r.status == 200:
                    counters["ok_batches"] += 1
                    counters["ok_writes"] += batch_size
                elif r.status == 503:
                    counters["503"] += 1
                else:
                    counters[f"http_{r.status}"] = counters.get(f"http_{r.status}", 0) + 1
        except Exception as e:
            counters["err"] = counters.get("err", 0) + 1
            counters["last_err"] = str(e)[:100]
        seq += batch_size


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--mode", choices=["single", "batch"], required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--connector-limit", type=int, default=200)
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="If >0, pass a pre-computed embedding of this dim (skips server-side embedder).",
    )
    args = ap.parse_args()

    print(f"=== http_bench mode={args.mode} url={args.url} ===")
    print(
        f"config: concurrency={args.concurrency} duration={args.duration}s "
        f"batch_size={args.batch_size if args.mode == 'batch' else 'N/A'}"
    )

    connector = aiohttp.TCPConnector(limit=args.connector_limit, ttl_dns_cache=300)
    stop_evt = asyncio.Event()
    samples = []
    counters = {"ok": 0, "ok_batches": 0, "ok_writes": 0, "503": 0}

    async with aiohttp.ClientSession(connector=connector) as session:
        if args.mode == "single":
            workers = [
                asyncio.create_task(
                    worker_single(
                        session,
                        args.url,
                        args.token,
                        i,
                        stop_evt,
                        samples,
                        counters,
                        args.embed_dim,
                    )
                )
                for i in range(args.concurrency)
            ]
        else:
            workers = [
                asyncio.create_task(
                    worker_batch(
                        session,
                        args.url,
                        args.token,
                        i,
                        args.batch_size,
                        stop_evt,
                        samples,
                        counters,
                        args.embed_dim,
                    )
                )
                for i in range(args.concurrency)
            ]

        t_start = time.perf_counter()
        print(f"\n{'sec':>4} {'ok':>10} {'503':>6} {'err':>6} {'p50_ms':>8} {'p99_ms':>8}")
        last_ok = 0
        last_503 = 0
        last_err = 0
        for sec in range(args.duration):
            await asyncio.sleep(1)
            ok_now = counters.get("ok", 0) + counters.get("ok_writes", 0)
            er_now = counters.get("err", 0)
            f5_now = counters.get("503", 0)
            recent = (
                [d for d in samples[-2000:]]
                if args.mode == "single"
                else [d for d, _ in samples[-2000:]]
            )
            p50, _, p99 = pcts(recent)
            print(
                f"{sec + 1:>4} {ok_now - last_ok:>10} {f5_now - last_503:>6} "
                f"{er_now - last_err:>6} {p50:>8.1f} {p99:>8.1f}"
            )
            last_ok, last_503, last_err = ok_now, f5_now, er_now

        stop_evt.set()
        await asyncio.gather(*workers, return_exceptions=True)
        wall = time.perf_counter() - t_start

    print("\n=== summary ===")
    if args.mode == "single":
        durs = samples
        ok = counters.get("ok", 0)
        tput = ok / wall
        p50, p95, p99 = pcts(durs)
        print(f"wall: {wall:.1f}s")
        print(f"ok: {ok}  tput: {tput:.0f}/s  p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    else:
        durs = [d for d, _ in samples]
        ok_batches = counters.get("ok_batches", 0)
        ok_writes = counters.get("ok_writes", 0)
        batch_tput = ok_batches / wall
        write_tput = ok_writes / wall
        p50, p95, p99 = pcts(durs)
        print(f"wall: {wall:.1f}s")
        print(
            f"ok_batches: {ok_batches}  ok_writes: {ok_writes}  "
            f"batch_tput: {batch_tput:.1f}/s  write_tput: {write_tput:.0f}/s"
        )
        print(f"batch latency: p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    print(f"503 (admission): {counters.get('503', 0)}")
    print(f"errors: {counters.get('err', 0)}")
    extra = {
        k: v for k, v in counters.items() if k not in {"ok", "ok_batches", "ok_writes", "503", "err"}
    }
    if extra:
        print(f"other: {json.dumps(extra)}")


if __name__ == "__main__":
    asyncio.run(main())
