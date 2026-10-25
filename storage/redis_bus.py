"""
QuantNexus-L2 :: Redis Shared-State Bus
=======================================

Sub-millisecond in-memory state bus shared between:
  * the Fast Loop (order-book / execution engine) -> reads risk state, writes L2 snapshots
  * the Slow Loop (LangGraph agentic risk gate)   -> writes risk state

Every public read/write returns the measured round-trip latency in milliseconds
(``time.perf_counter`` wall-clock around the Redis call) so latency SLAs can be
asserted directly in tests and monitored in production.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_RISK_MULTIPLIER: float = 1.0
DEFAULT_CIRCUIT_BREAKER: bool = False
DEFAULT_REASON: str = "NORMAL"


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskState:
    risk_multiplier: float = DEFAULT_RISK_MULTIPLIER
    circuit_breaker: bool = DEFAULT_CIRCUIT_BREAKER
    reason: str = DEFAULT_REASON
    updated_at: Optional[float] = None  # unix epoch seconds; None => default (never set)
    is_default: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    best_bid: float
    best_ask: float
    obi: float
    obi_zscore: float
    mid_price: float = field(default=0.0)
    spread_bps: float = field(default=0.0)
    updated_at: float = field(default=0.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BusResult:
    """Generic wrapper: payload + measured latency."""

    data: object
    latency_ms: float


# --------------------------------------------------------------------------- #
# Bus
# --------------------------------------------------------------------------- #
class RedisStateBus:
    """Async Redis wrapper for global risk state and L2 order-book snapshots."""

    RISK_KEY = "risk:state"
    OB_KEY_FMT = "orderbook:{symbol}"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> None:
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", 6379))
        self.db = int(db if db is not None else os.getenv("REDIS_DB", 0))
        self.password = password or os.getenv("REDIS_PASSWORD") or None
        self.key_prefix = key_prefix or os.getenv("REDIS_KEY_PREFIX", "qn")
        self._client: Optional[aioredis.Redis] = None

    # ------------------------------------------------------------------ infra
    def _key(self, suffix: str) -> str:
        return f"{self.key_prefix}:{suffix}"

    @property
    def client(self) -> aioredis.Redis:
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if (
            self._client is None
            or (current_loop is not None and getattr(self, "_loop", None) is not current_loop)
            or (getattr(self, "_loop", None) is not None and self._loop.is_closed())
        ):
            self._client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_keepalive=True,
                health_check_interval=30,
            )
            self._loop = current_loop
        return self._client

    @property
    def sync_client(self):
        if not hasattr(self, "_sync_client") or self._sync_client is None:
            import redis
            self._sync_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
        return self._sync_client

    async def connect(self) -> float:
        """Open the connection pool and warm it up. Returns PING latency (ms)."""
        _ = self.client
        return await self.ping()

    async def ping(self) -> float:
        t0 = time.perf_counter()
        await self.client.ping()
        return (time.perf_counter() - t0) * 1000.0

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None
        if hasattr(self, "_sync_client") and self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def __aenter__(self) -> "RedisStateBus":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------ risk state
    async def get_risk_state(self) -> BusResult:
        """Read global risk state. Falls back to defaults if never set."""
        t0 = time.perf_counter()
        raw = await self.client.hgetall(self._key(self.RISK_KEY))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not raw:
            return BusResult(RiskState(), latency_ms)

        state = RiskState(
            risk_multiplier=float(raw.get("risk_multiplier", DEFAULT_RISK_MULTIPLIER)),
            circuit_breaker=raw.get("circuit_breaker", "0") == "1",
            reason=raw.get("reason", DEFAULT_REASON),
            updated_at=float(raw["updated_at"]) if "updated_at" in raw else None,
            is_default=False,
        )
        return BusResult(state, latency_ms)

    async def set_risk_state(
        self, risk_multiplier: float, circuit_breaker: bool, reason: str
    ) -> BusResult:
        """Atomically overwrite global risk state. Returns written state + latency."""
        if not 0.0 <= risk_multiplier <= 10.0:
            raise ValueError(f"risk_multiplier out of bounds [0, 10]: {risk_multiplier}")

        now = time.time()
        mapping = {
            "risk_multiplier": repr(float(risk_multiplier)),
            "circuit_breaker": "1" if circuit_breaker else "0",
            "reason": reason,
            "updated_at": repr(now),
        }
        t0 = time.perf_counter()
        await self.client.hset(self._key(self.RISK_KEY), mapping=mapping)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state = RiskState(
            risk_multiplier=float(risk_multiplier),
            circuit_breaker=bool(circuit_breaker),
            reason=reason,
            updated_at=now,
            is_default=False,
        )
        return BusResult(state, latency_ms)

    async def clear_risk_state(self) -> None:
        """Delete stored risk state so subsequent reads return defaults."""
        await self.client.delete(self._key(self.RISK_KEY))

    # ------------------------------------------------------- order-book state
    async def update_orderbook_snapshot(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        obi: float,
        obi_zscore: float,
    ) -> BusResult:
        """Write the latest top-of-book + order-book-imbalance metrics for a symbol."""
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("best_bid and best_ask must be positive")
        if best_ask < best_bid:
            raise ValueError(f"Crossed book: bid={best_bid} > ask={best_ask}")

        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 1e4
        now = time.time()

        snap = OrderBookSnapshot(
            symbol=symbol,
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            obi=float(obi),
            obi_zscore=float(obi_zscore),
            mid_price=mid,
            spread_bps=spread_bps,
            updated_at=now,
        )
        mapping = {k: (repr(v) if isinstance(v, float) else str(v)) for k, v in snap.to_dict().items()}

        t0 = time.perf_counter()
        await self.client.hset(self._key(self.OB_KEY_FMT.format(symbol=symbol)), mapping=mapping)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BusResult(snap, latency_ms)

    async def get_orderbook_snapshot(self, symbol: str) -> BusResult:
        """Read latest snapshot for a symbol. ``data`` is None if never written."""
        t0 = time.perf_counter()
        raw = await self.client.hgetall(self._key(self.OB_KEY_FMT.format(symbol=symbol)))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not raw:
            return BusResult(None, latency_ms)

        snap = OrderBookSnapshot(
            symbol=raw["symbol"],
            best_bid=float(raw["best_bid"]),
            best_ask=float(raw["best_ask"]),
            obi=float(raw["obi"]),
            obi_zscore=float(raw["obi_zscore"]),
            mid_price=float(raw.get("mid_price", 0.0)),
            spread_bps=float(raw.get("spread_bps", 0.0)),
            updated_at=float(raw.get("updated_at", 0.0)),
        )
        return BusResult(snap, latency_ms)

    async def delete_orderbook_snapshot(self, symbol: str) -> None:
        await self.client.delete(self._key(self.OB_KEY_FMT.format(symbol=symbol)))

    # ------------------------------------------------------------ sync methods (for UI/Streamlit)
    def get_risk_state_sync(self) -> BusResult:
        """Synchronous version of get_risk_state for non-async environments (Streamlit)."""
        t0 = time.perf_counter()
        raw = self.sync_client.hgetall(self._key(self.RISK_KEY))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not raw:
            return BusResult(RiskState(), latency_ms)

        state = RiskState(
            risk_multiplier=float(raw.get("risk_multiplier", DEFAULT_RISK_MULTIPLIER)),
            circuit_breaker=raw.get("circuit_breaker", "0") == "1",
            reason=raw.get("reason", DEFAULT_REASON),
            updated_at=float(raw["updated_at"]) if "updated_at" in raw else None,
            is_default=False,
        )
        return BusResult(state, latency_ms)

    def get_orderbook_snapshot_sync(self, symbol: str) -> BusResult:
        """Synchronous version of get_orderbook_snapshot for non-async environments (Streamlit)."""
        t0 = time.perf_counter()
        raw = self.sync_client.hgetall(self._key(self.OB_KEY_FMT.format(symbol=symbol)))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not raw:
            return BusResult(None, latency_ms)

        snap = OrderBookSnapshot(
            symbol=raw["symbol"],
            best_bid=float(raw["best_bid"]),
            best_ask=float(raw["best_ask"]),
            obi=float(raw["obi"]),
            obi_zscore=float(raw["obi_zscore"]),
            mid_price=float(raw.get("mid_price", 0.0)),
            spread_bps=float(raw.get("spread_bps", 0.0)),
            updated_at=float(raw.get("updated_at", 0.0)),
        )
        return BusResult(snap, latency_ms)
