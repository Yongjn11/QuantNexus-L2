"""
QuantNexus-L2 :: Risk-Gated TWAP Execution Engine
=================================================

Sub-millisecond risk-gated TWAP execution engine. Converts entry and exit
intent into child order slices while reading global risk state from the
``RedisStateBus`` before every single slice.

Key Responsibilities:
---------------------
1. Pre-Trade Risk Gate:
   - Evaluated at Tick 0 before launching any child slice.
   - If ``circuit_breaker == True`` or ``risk_multiplier <= 0.0``, the parent
     order is rejected immediately (status = 'REJECTED_BY_AI').
   - For new entries, position size is scaled: ``effective_qty = base_qty * risk_multiplier``.
   - For exit/flatten orders (``is_exit=True``), liquidation is permitted even during
     circuit breakers to allow emergency protective exits.

2. Mid-Flight Circuit Breaker Preemption:
   - Before executing each slice k in {1..N}, the engine queries ``bus.get_risk_state()``.
   - If a sudden Black-Swan event triggers (circuit_breaker == True or risk_multiplier == 0.0),
     all remaining unfilled slices are immediately aborted.
   - Execution status is set to 'ABORTED_BY_AI', preserving filled quantity and recording
     the protected/cancelled notional USD.

3. Slippage Alpha Benchmarking:
   - Compares achieved ``twap_exec_price`` against ``naive_market_price``
     (what an aggressive retail market order sweeping the entire quantity at Tick 0 would pay).
   - Computes ``slippage_saved_bps`` and ``slippage_saved_usd``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Sequence, Union

from fast_loop.alpha_signals import MicrostructureSignalEngine, SweepEstimate
from storage.redis_bus import RedisStateBus, RiskState

BookType = Any  # tuple (bids, asks) or dict {"bids": ..., "asks": ...} or TickResult


class ExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABORTED_BY_AI = "ABORTED_BY_AI"
    REJECTED_BY_AI = "REJECTED_BY_AI"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SliceExecution:
    slice_index: int
    target_qty: float
    filled_qty: float
    vwap: float
    slippage_vs_touch_bps: float
    slippage_vs_mid_bps: float
    risk_check_latency_ms: float
    timestamp_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TWAPResult:
    symbol: str
    side: str
    status: ExecutionStatus
    base_qty: float
    effective_qty: float
    filled_qty: float
    unfilled_qty: float
    twap_exec_price: float
    naive_market_price: float
    slippage_saved_bps: float
    slippage_saved_usd: float
    risk_multiplier_used: float
    circuit_breaker_triggered: bool
    abort_reason: Optional[str] = None
    slices_filled: int = 0
    slices_total: int = 0
    protected_notional_usd: float = 0.0
    slices: list[SliceExecution] = field(default_factory=list)
    risk_latencies_ms: list[float] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class TWAPExecutionEngine:
    """Asynchronous TWAP execution engine with per-slice sub-ms Redis risk gating."""

    def __init__(
        self,
        bus: RedisStateBus,
        signal_engine: Optional[MicrostructureSignalEngine] = None,
        default_slices: int = 5,
        default_interval_ms: float = 0.0,
    ) -> None:
        self.bus = bus
        self.signal_engine = signal_engine or MicrostructureSignalEngine()
        self.default_slices = default_slices
        self.default_interval_ms = default_interval_ms

    @staticmethod
    def _extract_book(raw: BookType) -> tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]:
        """Normalize various book inputs to (bids, asks)."""
        if hasattr(raw, "book") and isinstance(raw.book, dict):
            # Duck-typed TickResult
            return raw.book["bids"], raw.book["asks"]
        if isinstance(raw, dict) and "bids" in raw and "asks" in raw:
            return raw["bids"], raw["asks"]
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return raw[0], raw[1]
        raise ValueError(f"Unsupported order book format: {type(raw)}")

    async def _next_book(
        self,
        source: Any,
        index: int,
    ) -> tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]:
        """Fetch the next order book from an async generator, iterator, list, or callable."""
        if inspect.isasyncgen(source) or hasattr(source, "__anext__"):
            item = await source.__anext__()
            return self._extract_book(item)
        if isinstance(source, (list, tuple)):
            if not source:
                raise ValueError("Empty book source list")
            item = source[index] if index < len(source) else source[-1]
            return self._extract_book(item)
        if callable(source):
            res = source(index)
            if inspect.isawaitable(res):
                res = await res
            return self._extract_book(res)
        if hasattr(source, "__next__"):
            return self._extract_book(next(source))
        raise ValueError(f"Unsupported book source type: {type(source)}")

    async def execute_order(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        book_source: Any,
        num_slices: Optional[int] = None,
        interval_ms: Optional[float] = None,
        is_exit: bool = False,
        before_slice_hook: Optional[Callable[[int], Any]] = None,
    ) -> TWAPResult:
        """Execute a parent order across child TWAP slices with per-slice risk gating."""
        t_start = time.perf_counter()
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side!r}. Must be 'BUY' or 'SELL'.")
        if base_qty <= 0:
            raise ValueError("base_qty must be strictly positive")

        n_slices = num_slices or self.default_slices
        if n_slices < 1:
            raise ValueError("num_slices must be >= 1")
        sleep_interval = (
            interval_ms if interval_ms is not None else self.default_interval_ms
        ) / 1000.0

        risk_latencies: list[float] = []

        # ------------------------------------------------------------------- #
        # 1. Pre-Trade Risk Gate (Tick 0)
        # ------------------------------------------------------------------- #
        risk_res = await self.bus.get_risk_state()
        risk_latencies.append(risk_res.latency_ms)
        risk_state: RiskState = risk_res.data
        risk_mult = risk_state.risk_multiplier
        circuit_breaker = risk_state.circuit_breaker

        # Exits are permitted to flatten even during circuit breaker / risk=0
        if not is_exit and (circuit_breaker or risk_mult <= 0.0):
            reason = (
                f"PRE_TRADE_GATED: circuit_breaker={circuit_breaker}, "
                f"risk_multiplier={risk_mult} ({risk_state.reason})"
            )
            return TWAPResult(
                symbol=symbol,
                side=side_norm,
                status=ExecutionStatus.REJECTED_BY_AI,
                base_qty=base_qty,
                effective_qty=0.0,
                filled_qty=0.0,
                unfilled_qty=base_qty,
                twap_exec_price=0.0,
                naive_market_price=0.0,
                slippage_saved_bps=0.0,
                slippage_saved_usd=0.0,
                risk_multiplier_used=risk_mult,
                circuit_breaker_triggered=circuit_breaker,
                abort_reason=reason,
                slices_filled=0,
                slices_total=n_slices,
                protected_notional_usd=0.0,
                slices=[],
                risk_latencies_ms=risk_latencies,
                duration_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # For entries, scale target quantity by risk multiplier; for exits, close target qty
        effective_qty = base_qty if is_exit else base_qty * risk_mult
        if effective_qty <= 0.0:
            return TWAPResult(
                symbol=symbol,
                side=side_norm,
                status=ExecutionStatus.REJECTED_BY_AI,
                base_qty=base_qty,
                effective_qty=0.0,
                filled_qty=0.0,
                unfilled_qty=base_qty,
                twap_exec_price=0.0,
                naive_market_price=0.0,
                slippage_saved_bps=0.0,
                slippage_saved_usd=0.0,
                risk_multiplier_used=risk_mult,
                circuit_breaker_triggered=circuit_breaker,
                abort_reason="Effective quantity after risk multiplier scaling is zero",
                slices_filled=0,
                slices_total=n_slices,
                protected_notional_usd=0.0,
                slices=[],
                risk_latencies_ms=risk_latencies,
                duration_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # ------------------------------------------------------------------- #
        # 2. Benchmark Naive Market Sweep Price at Tick 0
        # ------------------------------------------------------------------- #
        initial_bids, initial_asks = await self._next_book(book_source, 0)
        # Naive market order sweeps the effective size (or base size) in one aggressive sweep
        naive_sweep: SweepEstimate = self.signal_engine.estimate_naive_sweep_price(
            initial_bids, initial_asks, side_norm, effective_qty
        )
        naive_market_price = naive_sweep.vwap

        # ------------------------------------------------------------------- #
        # 3. Child Slice Execution Loop
        # ------------------------------------------------------------------- #
        slice_qty = effective_qty / n_slices
        filled_qty = 0.0
        total_notional_usd = 0.0
        slice_records: list[SliceExecution] = []
        status = ExecutionStatus.COMPLETED
        abort_reason: Optional[str] = None
        cb_triggered = circuit_breaker

        for k in range(n_slices):
            # Optional hook (e.g. for deterministic simulation / injecting slow-loop events)
            if before_slice_hook is not None:
                hook_res = before_slice_hook(k)
                if inspect.isawaitable(hook_res):
                    await hook_res

            # Sub-millisecond Risk Gating before EVERY slice
            risk_check = await self.bus.get_risk_state()
            risk_latencies.append(risk_check.latency_ms)
            current_risk: RiskState = risk_check.data

            # If NOT an exit, check for Mid-Flight Circuit Breaker Preemption
            if not is_exit and (current_risk.circuit_breaker or current_risk.risk_multiplier <= 0.0):
                status = ExecutionStatus.ABORTED_BY_AI
                cb_triggered = True
                abort_reason = (
                    f"MID_FLIGHT_PREEMPTION at slice {k + 1}/{n_slices}: "
                    f"circuit_breaker={current_risk.circuit_breaker}, "
                    f"risk_multiplier={current_risk.risk_multiplier} "
                    f"({current_risk.reason})"
                )
                break

            # Fetch the order book for this slice
            bids, asks = (
                (initial_bids, initial_asks) if k == 0 else await self._next_book(book_source, k)
            )

            # Execute child slice (small size walks top of book with minimal impact)
            slice_sweep = self.signal_engine.estimate_naive_sweep_price(
                bids, asks, side_norm, slice_qty
            )
            slice_vwap = slice_sweep.vwap

            filled_qty += slice_qty
            total_notional_usd += slice_qty * slice_vwap

            slice_records.append(
                SliceExecution(
                    slice_index=k + 1,
                    target_qty=slice_qty,
                    filled_qty=slice_qty,
                    vwap=slice_vwap,
                    slippage_vs_touch_bps=slice_sweep.slippage_vs_touch_bps,
                    slippage_vs_mid_bps=slice_sweep.slippage_vs_mid_bps,
                    risk_check_latency_ms=risk_check.latency_ms,
                    timestamp_ms=int(time.time() * 1000),
                )
            )

            if sleep_interval > 0 and k < n_slices - 1:
                await asyncio.sleep(sleep_interval)

        # ------------------------------------------------------------------- #
        # 4. Aggregate Performance & Slippage Alpha
        # ------------------------------------------------------------------- #
        twap_exec_price = (
            (total_notional_usd / filled_qty) if filled_qty > 0.0 else naive_market_price
        )
        unfilled_qty = effective_qty - filled_qty
        protected_notional_usd = unfilled_qty * naive_market_price

        # Slippage saved vs. naive sweep
        if side_norm == "BUY":
            slippage_saved_bps = (
                (naive_market_price - twap_exec_price) / naive_market_price * 10000.0
            )
            slippage_saved_usd = (naive_market_price - twap_exec_price) * filled_qty
        else:  # SELL
            slippage_saved_bps = (
                (twap_exec_price - naive_market_price) / naive_market_price * 10000.0
            )
            slippage_saved_usd = (twap_exec_price - naive_market_price) * filled_qty

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        return TWAPResult(
            symbol=symbol,
            side=side_norm,
            status=status,
            base_qty=base_qty,
            effective_qty=effective_qty,
            filled_qty=filled_qty,
            unfilled_qty=unfilled_qty,
            twap_exec_price=twap_exec_price,
            naive_market_price=naive_market_price,
            slippage_saved_bps=slippage_saved_bps,
            slippage_saved_usd=slippage_saved_usd,
            risk_multiplier_used=risk_mult,
            circuit_breaker_triggered=cb_triggered,
            abort_reason=abort_reason,
            slices_filled=len(slice_records),
            slices_total=n_slices,
            protected_notional_usd=protected_notional_usd,
            slices=slice_records,
            risk_latencies_ms=risk_latencies,
            duration_ms=duration_ms,
        )
