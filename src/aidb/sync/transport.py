"""WebSocket sync transport for AIDB P2P replication.

JSON-RPC protocol over WebSocket:
  - HELLO: exchange actor_ids
  - PULL_OPS: request ops since watermark
  - PUSH_OPS: send ops batch
  - ACK: acknowledge + update watermark
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SyncProtocol:
    """Shared protocol logic for sync client and server."""

    def __init__(self, db: Any):
        self.db = db
        self.actor_id: str = db.actor_id

    def handle_message(self, raw: str) -> str | None:
        """Process an incoming JSON-RPC message, return response or None."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({"error": "invalid_json"})

        method = msg.get("method")
        params = msg.get("params", {})
        msg_id = msg.get("id")

        if method == "HELLO":
            return json.dumps({
                "id": msg_id,
                "result": {"actor_id": self.actor_id},
            })

        elif method == "PULL_OPS":
            since_hlc = params.get("since_hlc")
            since_op_id = params.get("since_op_id")
            exclude_actor = params.get("exclude_actor")
            limit = params.get("limit", 500)

            # Convert hex-encoded HLC back to bytes
            hlc_bytes = bytes.fromhex(since_hlc) if since_hlc else None

            ops = self.db.extract_ops_since(
                since_hlc=hlc_bytes,
                since_op_id=since_op_id,
                exclude_actor=exclude_actor,
                limit=limit,
            )

            # Serialize ops for transport (bytes -> hex)
            serialized = []
            for op in ops:
                sop = dict(op)
                if sop.get("hlc"):
                    sop["hlc"] = bytes(sop["hlc"]).hex()
                if sop.get("embedding_hash"):
                    sop["embedding_hash"] = bytes(sop["embedding_hash"]).hex()
                serialized.append(sop)

            return json.dumps({"id": msg_id, "result": {"ops": serialized}})

        elif method == "PUSH_OPS":
            ops = params.get("ops", [])

            # Deserialize ops from transport (hex -> bytes)
            deserialized = []
            for sop in ops:
                op = dict(sop)
                if op.get("hlc"):
                    op["hlc"] = bytes.fromhex(op["hlc"])
                if op.get("embedding_hash"):
                    op["embedding_hash"] = bytes.fromhex(op["embedding_hash"])
                deserialized.append(op)

            result = self.db.apply_ops(deserialized)
            return json.dumps({"id": msg_id, "result": result})

        elif method == "ACK":
            peer_actor = params.get("peer_actor")
            hlc_hex = params.get("hlc")
            op_id = params.get("op_id")
            if peer_actor and hlc_hex and op_id:
                self.db.set_peer_watermark(
                    peer_actor, bytes.fromhex(hlc_hex), op_id
                )
            return json.dumps({"id": msg_id, "result": {"ok": True}})

        return json.dumps({"id": msg_id, "error": f"unknown_method: {method}"})


class SyncServer:
    """WebSocket server that accepts sync connections from peers.

    Usage:
        server = SyncServer(db, host="0.0.0.0", port=8765)
        await server.start()
        # ... later ...
        await server.stop()
    """

    def __init__(self, db: Any, host: str = "0.0.0.0", port: int = 8765):
        self.db = db
        self.host = host
        self.port = port
        self.protocol = SyncProtocol(db)
        self._server = None

    async def _handler(self, websocket):
        """Handle a single WebSocket connection."""
        async for message in websocket:
            response = self.protocol.handle_message(message)
            if response:
                await websocket.send(response)

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for sync. "
                "Install with: pip install aidb[sync]"
            )

        self._server = await websockets.serve(
            self._handler, self.host, self.port
        )
        logger.info(f"Sync server listening on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


class SyncClient:
    """WebSocket client that connects to a peer and syncs bidirectionally.

    Usage:
        client = SyncClient(db, "ws://peer:8765")
        await client.sync_once()
    """

    def __init__(self, db: Any, peer_url: str):
        self.db = db
        self.peer_url = peer_url
        self.protocol = SyncProtocol(db)
        self._peer_actor: str | None = None
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send_rpc(self, ws, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC message and return the response."""
        msg = {"id": self._next_id(), "method": method}
        if params:
            msg["params"] = params
        await ws.send(json.dumps(msg))
        raw = await ws.recv()
        return json.loads(raw)

    async def sync_once(self) -> dict:
        """Perform one full bidirectional sync cycle.

        Returns: {"pushed": int, "pulled": int}
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for sync. "
                "Install with: pip install aidb[sync]"
            )

        async with websockets.connect(self.peer_url) as ws:
            # Step 1: Hello — exchange actor IDs
            resp = await self._send_rpc(ws, "HELLO")
            self._peer_actor = resp["result"]["actor_id"]

            # Step 2: Pull — get remote ops we haven't seen
            watermark = self.db.get_peer_watermark(self._peer_actor)
            pull_params: dict[str, Any] = {
                "exclude_actor": self.db.actor_id,
                "limit": 1000,
            }
            if watermark:
                pull_params["since_hlc"] = bytes(watermark["hlc"]).hex()
                pull_params["since_op_id"] = watermark["op_id"]

            resp = await self._send_rpc(ws, "PULL_OPS", pull_params)
            remote_ops = resp["result"]["ops"]

            # Apply pulled ops locally
            pulled = 0
            if remote_ops:
                deserialized = []
                for sop in remote_ops:
                    op = dict(sop)
                    if op.get("hlc"):
                        op["hlc"] = bytes.fromhex(op["hlc"])
                    if op.get("embedding_hash"):
                        op["embedding_hash"] = bytes.fromhex(op["embedding_hash"])
                    deserialized.append(op)

                result = self.db.apply_ops(deserialized)
                pulled = result["ops_applied"]

                # Update watermark for this peer
                if remote_ops:
                    last = remote_ops[-1]
                    hlc_hex = last["hlc"] if isinstance(last["hlc"], str) else bytes(last["hlc"]).hex()
                    self.db.set_peer_watermark(
                        self._peer_actor, bytes.fromhex(hlc_hex), last["op_id"]
                    )

            # Step 3: Push — send our ops to the peer
            peer_watermark_on_us = self.db.get_peer_watermark(self.db.actor_id)
            push_params: dict[str, Any] = {"exclude_actor": self._peer_actor}
            # We don't have a direct way to know the peer's watermark for US,
            # so we push all ops since the beginning (the peer will deduplicate)
            local_ops = self.db.extract_ops_since(
                exclude_actor=self._peer_actor,
                limit=1000,
            )

            pushed = 0
            if local_ops:
                serialized = []
                for op in local_ops:
                    sop = dict(op)
                    if sop.get("hlc"):
                        sop["hlc"] = bytes(sop["hlc"]).hex()
                    if sop.get("embedding_hash"):
                        sop["embedding_hash"] = bytes(sop["embedding_hash"]).hex()
                    serialized.append(sop)

                resp = await self._send_rpc(ws, "PUSH_OPS", {"ops": serialized})
                pushed = resp["result"].get("ops_applied", 0)

                # Acknowledge to peer
                if local_ops:
                    last = local_ops[-1]
                    hlc_hex = bytes(last["hlc"]).hex() if isinstance(last["hlc"], (bytes, bytearray)) else last["hlc"]
                    await self._send_rpc(ws, "ACK", {
                        "peer_actor": self.db.actor_id,
                        "hlc": hlc_hex,
                        "op_id": last["op_id"],
                    })

            return {"pushed": pushed, "pulled": pulled}


class SyncDaemon:
    """Background daemon that periodically syncs with known peers.

    Usage:
        daemon = SyncDaemon(db, peers=["ws://peer1:8765", "ws://peer2:8765"])
        await daemon.start()
        # ... later ...
        daemon.stop()
    """

    def __init__(
        self,
        db: Any,
        peers: list[str],
        interval: float = 30.0,
    ):
        self.db = db
        self.peers = peers
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._running = False

    async def _loop(self):
        """Main sync loop."""
        while self._running:
            for peer_url in self.peers:
                try:
                    client = SyncClient(self.db, peer_url)
                    result = await client.sync_once()
                    logger.info(
                        f"Synced with {peer_url}: "
                        f"pushed={result['pushed']}, pulled={result['pulled']}"
                    )
                except Exception as e:
                    logger.warning(f"Sync with {peer_url} failed: {e}")

            await asyncio.sleep(self.interval)

    async def start(self):
        """Start the daemon."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Sync daemon started (interval={self.interval}s, "
            f"peers={len(self.peers)})"
        )

    def stop(self):
        """Stop the daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None


def main():
    """CLI entry point for aidb-sync."""
    import click

    @click.command()
    @click.option("--db", required=True, help="Path to AIDB database")
    @click.option("--port", default=8765, help="WebSocket server port")
    @click.option("--peer", multiple=True, help="Peer URLs to sync with")
    @click.option("--interval", default=30.0, help="Sync interval in seconds")
    def run(db: str, port: int, peer: tuple, interval: float):
        """Run AIDB sync server + daemon."""
        from _aidb_rust import AIDB

        engine = AIDB(db)

        async def _run():
            server = SyncServer(engine, port=port)
            await server.start()

            if peer:
                daemon = SyncDaemon(engine, list(peer), interval=interval)
                await daemon.start()

            # Run forever
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

    run()
