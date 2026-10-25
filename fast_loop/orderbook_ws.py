"""
QuantNexus-L2 :: Resilient L2 Order-Book WebSocket Stream
=========================================================

Streams Top-20 BTC/USDT L2 snapshots with multi-endpoint failover:

    1. Binance Vision  (wss://data-stream.binance.vision)   – primary, no key, no geo-block
    2. Binance Global  (wss://stream.binance.com:9443)      – fallback
    3. Bybit V5 Spot   (wss://stream.bybit.com/v5/public/spot, orderbook.50) – fallback

Every message is normalised to::

    {"symbol": "BTC/USDT", "timestamp_ms": int, "bids": [[p, q], ...], "asks": [[p, q], ...]}

then pushed through ``MicrostructureSignalEngine`` and published to the
``RedisStateBus``. End-to-end tick latency is measured from the moment the
WebSocket frame is handed to us until Redis acknowledges the snapshot write.

Reconnects use exponential backoff with jitter, rotating through endpoints.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import time
from dataclasses import dataclass, field
from heapq import nlargest, nsmallest
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Sequence, Union

import websockets

from fast_loop.alpha_signals import MicrostructureSignalEngine, SignalMetrics

log = logging.getLogger("quantnexus.fast_loop.ws")

DEPTH = 20


# --------------------------------------------------------------------------- #
# Endpoint config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WsEndpoint:
    name: str
    url: str
    kind: str                                 # "binance" | "bybit"
    subscribe: Optional[dict] = None          # sent once after connect
    app_ping: Optional[dict] = None           # application-level heartbeat
    app_ping_interval_s: float = 20.0


DEFAULT_ENDPOINTS: tuple[WsEndpoint, ...] = (
    WsEndpoint(
        name="binance_vision",
        url="wss://data-stream.binance.vision/ws/btcusdt@depth20@100ms",
        kind="binance",
    ),
    WsEndpoint(
        name="binance_global",
        url="wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms",
        kind="binance",
    ),
    WsEndpoint(
        name="bybit_v5_spot",
        url="wss://stream.bybit.com/v5/public/spot",
        kind="bybit",
        subscribe={"op": "subscribe", "args": ["orderbook.50.BTCUSDT"]},
        app_ping={"op": "ping"},
        app_ping_interval_s=20.0,
    ),
)


# --------------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------------- #
NormalizedBook = dict[str, Any]


def _levels(raw: Sequence[Sequence[Any]], depth: int) -> list[list[float]]:
    return [[float(p), float(q)] for p, q, *_ in raw[:depth]]


class BinanceDepthParser:
    """Binance partial-depth stream (``@depth20``): every message is a full snapshot."""

    def __init__(self, symbol: str, depth: int = DEPTH) -> None:
        self.symbol, self.depth = symbol, depth

    def parse(self, msg: dict) -> Optional[NormalizedBook]:
        data = msg.get("data", msg)  # combined-stream wrapper support
        if "bids" not in data or "asks" not in data:
            return None
        ts = data.get("E") or data.get("T") or int(time.time() * 1000)
        return {
            "symbol": self.symbol,
            "timestamp_ms": int(ts),
            "bids": _levels(data["bids"], self.depth),
            "asks": _levels(data["asks"], self.depth),
        }


class BybitBookParser:
    """Bybit V5 orderbook: snapshot + incremental deltas → maintained local book."""

    def __init__(self, symbol: str, depth: int = DEPTH) -> None:
        self.symbol, self.depth = symbol, depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.has_snapshot = False

    @staticmethod
    def _apply(side: dict[float, float], updates: Sequence[Sequence[str]]) -> None:
        for p, q in updates:
            px, qty = float(p), float(q)
            if qty == 0.0:
                side.pop(px, None)
            else:
                side[px] = qty

    def parse(self, msg: dict) -> Optional[NormalizedBook]:
        topic = msg.get("topic", "")
        if not topic.startswith("orderbook"):
            return None  # subscribe ack / pong / other control frames
        data = msg.get("data") or {}
        mtype = msg.get("type")
        # Per Bybit docs: u == 1 means service restart → treat as snapshot.
        if mtype == "snapshot" or data.get("u") == 1:
            self.bids.clear()
            self.asks.clear()
            self.has_snapshot = True
        elif mtype == "delta":
            if not self.has_snapshot:
                return None
        else:
            return None
        self._apply(self.bids, data.get("b", ()))
        self._apply(self.asks, data.get("a", ()))
        if not self.bids or not self.asks:
            return None
        return {
            "symbol": self.symbol,
            "timestamp_ms": int(msg.get("ts") or time.time() * 1000),
            "bids": [[p, self.bids[p]] for p in nlargest(self.depth, self.bids)],
            "asks": [[p, self.asks[p]] for p in nsmallest(self.depth, self.asks)],
        }


def make_parser(ep: WsEndpoint, symbol: str, depth: int = DEPTH):
    if ep.kind == "binance":
        return BinanceDepthParser(symbol, depth)
    if ep.kind == "bybit":
        return BybitBookParser(symbol, depth)
    raise ValueError(f"unknown endpoint kind {ep.kind!r}")


# --------------------------------------------------------------------------- #
# Tick result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TickResult:
    seq: int
    endpoint: str
    book: NormalizedBook
    metrics: SignalMetrics
    signal_latency_ms: float      # signal-engine compute only
    redis_latency_ms: float       # Redis HSET round-trip (0.0 if no bus)
    tick_latency_ms: float        # frame received → parsed → signal → Redis ack


class StreamUnavailableError(ConnectionError):
    """Raised when all endpoints keep failing beyond ``max_reconnect_attempts``."""


TickCallback = Callable[[TickResult], Union[None, Awaitable[None]]]


# --------------------------------------------------------------------------- #
# Stream
# --------------------------------------------------------------------------- #
class L2OrderBookStream:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        engine: Optional[MicrostructureSignalEngine] = None,
        bus: Any = None,  # storage.redis_bus.RedisStateBus (duck-typed to avoid hard dep)
        endpoints: Sequence[WsEndpoint] = DEFAULT_ENDPOINTS,
        depth: int = DEPTH,
        open_timeout_s: float = 5.0,
        recv_timeout_s: float = 10.0,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        max_reconnect_attempts: Optional[int] = None,
        on_tick: Optional[TickCallback] = None,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint required")
        self.symbol = symbol
        self.engine = engine or MicrostructureSignalEngine(depth=depth)
        self.bus = bus
        self.endpoints = tuple(endpoints)
        self.depth = depth
        self.open_timeout_s = open_timeout_s
        self.recv_timeout_s = recv_timeout_s
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.max_reconnect_attempts = max_reconnect_attempts
        self.on_tick = on_tick

        self._stop = False
        self.active_endpoint: Optional[str] = None
        self.stats: dict[str, Any] = {
            "connects": 0,
            "failures": 0,
            "ticks": 0,
            "parse_errors": 0,
            "failures_by_endpoint": {ep.name: 0 for ep in self.endpoints},
        }

    # ---------------------------------------------------------------- control
    def stop(self) -> None:
        self._stop = True

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max_s, self.backoff_base_s * (2 ** max(0, attempt - 1)))
        return base * (0.8 + 0.4 * random.random())  # ±20% jitter

    @staticmethod
    async def _heartbeat(ws, payload: dict, interval: float) -> None:
        msg = json.dumps(payload)
        while True:
            await asyncio.sleep(interval)
            await ws.send(msg)

    # ------------------------------------------------------------- processing
    async def _process(self, raw: Union[str, bytes], parser, ep: WsEndpoint, t0: int) -> Optional[TickResult]:
        try:
            book = parser.parse(json.loads(raw))
        except (ValueError, KeyError, TypeError) as exc:
            self.stats["parse_errors"] += 1
            log.warning("parse error on %s: %s", ep.name, exc)
            return None
        if book is None:
            return None

        t_sig = time.perf_counter_ns()
        try:
            m = self.engine.update(book["bids"], book["asks"], book["timestamp_ms"])
        except ValueError as exc:  # crossed / empty book – skip tick, keep stream alive
            self.stats["parse_errors"] += 1
            log.warning("invalid book on %s: %s", ep.name, exc)
            return None
        sig_ms = (time.perf_counter_ns() - t_sig) / 1e6

        redis_ms = 0.0
        if self.bus is not None:
            res = await self.bus.update_orderbook_snapshot(
                self.symbol, m.best_bid, m.best_ask, m.weighted_obi, m.obi_zscore
            )
            redis_ms = res.latency_ms

        self.stats["ticks"] += 1
        return TickResult(
            seq=self.stats["ticks"],
            endpoint=ep.name,
            book=book,
            metrics=m,
            signal_latency_ms=sig_ms,
            redis_latency_ms=redis_ms,
            tick_latency_ms=(time.perf_counter_ns() - t0) / 1e6,
        )

    # ----------------------------------------------------------------- stream
    async def stream_ticks(self, max_ticks: Optional[int] = None) -> AsyncIterator[TickResult]:
        """Yield processed ticks forever (or until ``max_ticks`` / ``stop()``).

        Failover: on any connection error / timeout the next endpoint is tried
        after an exponential-backoff delay. The attempt counter resets on every
        successfully processed tick.
        """
        produced = 0
        attempt = 0
        idx = 0
        self._stop = False

        while not self._stop and (max_ticks is None or produced < max_ticks):
            ep = self.endpoints[idx % len(self.endpoints)]
            parser = make_parser(ep, self.symbol, self.depth)  # fresh state per connection
            hb_task: Optional[asyncio.Task] = None
            try:
                log.info("connecting to %s (%s)", ep.name, ep.url)
                async with websockets.connect(
                    ep.url,
                    open_timeout=self.open_timeout_s,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                    max_size=2**22,
                ) as ws:
                    if ep.subscribe:
                        await ws.send(json.dumps(ep.subscribe))
                    if ep.app_ping:
                        hb_task = asyncio.create_task(
                            self._heartbeat(ws, ep.app_ping, ep.app_ping_interval_s)
                        )
                    # Warm the Redis path: the WS/TLS handshake leaves the Redis
                    # socket idle, and the first write after idle pays a cold-path
                    # penalty (observed ~9 ms vs ~1 ms steady state).
                    if self.bus is not None:
                        await self.bus.ping()
                    self.stats["connects"] += 1
                    self.active_endpoint = ep.name
                    log.info("connected to %s", ep.name)

                    try:
                        while not self._stop:
                            raw = await asyncio.wait_for(ws.recv(), timeout=self.recv_timeout_s)
                            t0 = time.perf_counter_ns()
                            result = await self._process(raw, parser, ep, t0)
                            if result is None:
                                continue
                            attempt = 0
                            produced += 1
                            if self.on_tick is not None:
                                ret = self.on_tick(result)
                                if inspect.isawaitable(ret):
                                    await ret
                            yield result
                            if max_ticks is not None and produced >= max_ticks:
                                return
                    finally:
                        if hb_task is not None:
                            hb_task.cancel()
                if self._stop:
                    return
                raise ConnectionError(f"{ep.name}: connection closed by server")

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 – any transport failure → failover
                self.active_endpoint = None
                self.stats["failures"] += 1
                self.stats["failures_by_endpoint"][ep.name] += 1
                attempt += 1
                idx += 1
                if self.max_reconnect_attempts is not None and attempt > self.max_reconnect_attempts:
                    raise StreamUnavailableError(
                        f"all endpoints failed {attempt - 1} consecutive times; last: {exc!r}"
                    ) from exc
                delay = self._backoff(attempt)
                nxt = self.endpoints[idx % len(self.endpoints)].name
                log.warning("%s failed (%r); retry #%d via %s in %.2fs", ep.name, exc, attempt, nxt, delay)
                await asyncio.sleep(delay)

    async def run(self) -> None:
        """Run forever, driving ``on_tick`` callbacks (for the live trading loop)."""
        async for _ in self.stream_ticks():
            pass
