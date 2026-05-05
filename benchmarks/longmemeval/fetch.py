#!/usr/bin/env python3
"""Download a LongMemEval variant from HuggingFace.

The HF dataset viewer is broken (validation error on the `answer`
column's mixed types), so we bypass `datasets.load_dataset` and pull
the raw JSON from the resolve URL directly. Files are stored alongside
this script with a `.json` suffix added (the upstream repo dropped the
extension at some point).
"""
import argparse
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

# Direct HF resolve URLs. The files are LFS-backed but `resolve/main`
# returns the actual content (HF redirects through a CDN).
URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_m",
}


def download(variant: str, force: bool = False) -> Path:
    if variant not in URLS:
        raise SystemExit(f"unknown variant {variant!r}; pick from {list(URLS)}")
    url = URLS[variant]
    out = HERE / f"longmemeval_{variant}.json"
    if out.exists() and not force:
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"already downloaded: {out.name} ({size_mb:.1f} MB)")
        return out

    print(f"downloading {url} -> {out.name}")
    # Stream to file with progress every 5 MB.
    req = urllib.request.Request(url, headers={"User-Agent": "yantrikdb-bench/1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        chunk_size = 1024 * 1024  # 1 MB
        next_log = 5 * 1024 * 1024
        downloaded = 0
        with open(out, "wb") as f:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if downloaded >= next_log:
                    pct = (downloaded / total * 100) if total else 0
                    print(
                        f"  {downloaded / (1024 * 1024):.1f} MB"
                        + (f" / {total / (1024 * 1024):.1f} MB ({pct:.1f}%)" if total else "")
                    )
                    next_log += 5 * 1024 * 1024
    final_mb = out.stat().st_size / (1024 * 1024)
    print(f"done: {out.name} ({final_mb:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Download LongMemEval data")
    ap.add_argument(
        "--variant",
        default="oracle",
        choices=list(URLS),
        help="oracle (15MB, smallest) | s (278MB, headline) | m (2.75GB, scale)",
    )
    ap.add_argument("--force", action="store_true", help="redownload even if file exists")
    args = ap.parse_args()
    download(args.variant, force=args.force)


if __name__ == "__main__":
    main()
