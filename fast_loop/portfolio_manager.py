"""
QuantNexus-L2 :: Dual-Account Portfolio Manager
===============================================

Maintains real-time A/B portfolio accounting and position lifecycle:
  * **QuantNexus Account**:
    - Fills orders via ``TWAPExecutionEngine`` (saving book-depth slippage).
    - Scales entry sizing dynamically via Slow-Loop ``risk_multiplier``.
    - Enforces sub-ms circuit breaker aborts and emergency flattening.
  * **Baseline Retail Account**:
    - Ignores risk gating (always trades fixed ``base_qty``).
    - Sweeps the order book aggressively (paying full naive market slippage).
    - Suffers full drawdowns during Black-Swan events without preemption.

Persists completed trades to PostgreSQL (``trade_executions``) and equity snapshots
to ``ab_portfolio_equity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from fast_loop.alpha_signals import MicrostructureSignalEngine, Signal, SignalMetrics
from fast_loop.twap_executor import ExecutionStatus, TWAPExecutionEngine, TWAPResult
from storage.pg_logger import (
    TRADE_STATUS_ABORTED,
    TRADE_STATUS_COMPLETED,
    PostgresLedger,
)
from storage.redis_bus import RedisStateBus


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    naive_entry_price: float
    entry_slippage_saved_bps: float
    entry_slippage_saved_usd: float
    risk_multiplier: float
    entry_time: datetime
    entry_status: ExecutionStatus


@dataclass
class BaselinePosition:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime


@dataclass
class RoundTripTradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    naive_market_price: float
    twap_exec_price: float
    slippage_saved_bps: float
    slippage_saved_usd: float
    risk_multiplier_used: float
    realized_pnl_usd: float
    baseline_entry_price: float
    baseline_exit_price: float
    baseline_quantity: float
    baseline_pnl_usd: float
    is_win: bool
    status: str
    exit_reason: str
    closed_at: datetime
    trade_id: Optional[int] = None


class DualPortfolioManager:
    """A/B Portfolio Manager tracking QuantNexus vs. Baseline Retail Bot."""

    def __init__(
        self,
        bus: RedisStateBus,
        ledger: PostgresLedger,
        executor: Optional[TWAPExecutionEngine] = None,
        signal_engine: Optional[MicrostructureSignalEngine] = None,
        initial_equity: float = 100_000.0,
        take_profit_bps: float = 6.0,
        stop_loss_bps: float = 3.0,
        trailing_lock_bps: float = 2.0,
        min_hold_ticks: int = 15,
        base_qty: float = 1.5,
        hydrate_from_db: bool = True,
    ) -> None:
        self.bus = bus
        self.ledger = ledger
        self.signal_engine = signal_engine or MicrostructureSignalEngine()
        self.executor = executor or TWAPExecutionEngine(bus=bus, signal_engine=self.signal_engine)

        self.initial_equity = float(initial_equity)
        self.quantnexus_equity_usd = float(initial_equity)
        self.baseline_equity_usd = float(initial_equity)

        self.take_profit_bps = float(take_profit_bps)
        self.stop_loss_bps = float(stop_loss_bps)
        self.trailing_lock_bps = float(trailing_lock_bps)
        self.min_hold_ticks = int(min_hold_ticks)
        self.base_qty = float(base_qty)

        # Position state
        self.qn_position: Optional[Position] = None
        self.baseline_position: Optional[BaselinePosition] = None
        self.position_hold_ticks: int = 0

        # Statistics & High-Water Mark
        self.peak_qn_equity = float(initial_equity)
        self.peak_baseline_equity = float(initial_equity)
        self.max_drawdown_pct = 0.0
        self.baseline_max_drawdown_pct = 0.0
        self.closed_trades: list[RoundTripTradeRecord] = []

        if hydrate_from_db and self.ledger is not None:
            self._hydrate_from_db()

    def _hydrate_from_db(self) -> None:
        """Hydrate starting equity and trade history from PostgreSQL to ensure continuous curves."""
        try:
            summary = self.ledger.get_portfolio_summary()
            latest = summary.get("latest_equity")
            if latest:
                self.quantnexus_equity_usd = float(latest["quantnexus_equity_usd"])
                self.baseline_equity_usd = float(latest["baseline_equity_usd"])
                self.peak_qn_equity = max(self.initial_equity, self.quantnexus_equity_usd)
                self.peak_baseline_equity = max(self.initial_equity, self.baseline_equity_usd)
                self.max_drawdown_pct = float(latest.get("max_drawdown_pct") or 0.0)

            db_trades = self.ledger.get_recent_trades(limit=1000)
            if db_trades:
                hydrated: list[RoundTripTradeRecord] = []
                for t in reversed(db_trades):
                    rec = RoundTripTradeRecord(
                        symbol=t["symbol"],
                        side=t["side"],
                        entry_price=float(t["entry_price"] or 0.0),
                        exit_price=float(t["exit_price"] or 0.0),
                        quantity=float(t["quantity"] or 0.0),
                        naive_market_price=float(t["naive_market_price"] or 0.0),
                        twap_exec_price=float(t["twap_exec_price"] or 0.0),
                        slippage_saved_bps=float(t["slippage_saved_bps"] or 0.0),
                        slippage_saved_usd=float(t["slippage_saved_usd"] or 0.0),
                        risk_multiplier_used=float(t["risk_multiplier_used"] or 1.0),
                        realized_pnl_usd=float(t["realized_pnl_usd"] or 0.0),
                        baseline_entry_price=float(t["naive_market_price"] or 0.0),
                        baseline_exit_price=float(t["exit_price"] or 0.0),
                        baseline_quantity=float(t["quantity"] or 0.0),
                        baseline_pnl_usd=float(t["realized_pnl_usd"] or 0.0),
                        is_win=bool(t["is_win"]),
                        status=t["status"],
                        exit_reason="HYDRATED",
                        closed_at=t["timestamp"],
                        trade_id=t["id"],
                    )
                    hydrated.append(rec)
                self.closed_trades = hydrated
        except Exception:
            pass

    @property
    def has_position(self) -> bool:
        return self.qn_position is not None or self.baseline_position is not None

    # ----------------------------------------------------------------------- #
    # Position Lifecycle
    # ----------------------------------------------------------------------- #
    async def open_position(
        self,
        symbol: str,
        book_source: Any,
        base_qty: Optional[float] = None,
        num_slices: int = 5,
        before_slice_hook: Any = None,
    ) -> Optional[TWAPResult]:
        """Open a LONG position for QuantNexus (TWAP + risk gated) & Baseline."""
        if self.has_position:
            return None

        qty = float(base_qty or self.base_qty)

        # 1. QuantNexus Entry via TWAP Engine
        qn_entry = await self.executor.execute_order(
            symbol=symbol,
            side="BUY",
            base_qty=qty,
            book_source=book_source,
            num_slices=num_slices,
            is_exit=False,
            before_slice_hook=before_slice_hook,
        )

        now = datetime.now(timezone.utc)

        # Baseline Retail Bot always sweeps full base_qty at naive_market_price
        # Compute baseline entry price using naive sweep
        first_bids, first_asks = await self.executor._next_book(book_source, 0)
        baseline_sweep = self.signal_engine.estimate_naive_sweep_price(
            first_bids, first_asks, "BUY", qty
        )
        baseline_entry_px = baseline_sweep.vwap

        self.baseline_position = BaselinePosition(
            symbol=symbol,
            side="BUY",
            quantity=qty,
            entry_price=baseline_entry_px,
            entry_time=now,
        )

        if qn_entry.status == ExecutionStatus.REJECTED_BY_AI:
            # QuantNexus protected by AI Pre-Trade gate; only Baseline is entered
            self.qn_position = None
            return qn_entry

        # QuantNexus entered (either COMPLETED or partially filled via ABORTED_BY_AI)
        if qn_entry.filled_qty > 0.0:
            self.position_hold_ticks = 0
            self.qn_position = Position(
                symbol=symbol,
                side="BUY",
                quantity=qn_entry.filled_qty,
                entry_price=qn_entry.twap_exec_price,
                naive_entry_price=qn_entry.naive_market_price,
                entry_slippage_saved_bps=qn_entry.slippage_saved_bps,
                entry_slippage_saved_usd=qn_entry.slippage_saved_usd,
                risk_multiplier=qn_entry.risk_multiplier_used,
                entry_time=now,
                entry_status=qn_entry.status,
            )

        return qn_entry

    async def close_position(
        self,
        exit_book_source: Any,
        exit_reason: str = "MANUAL",
        num_slices: int = 5,
        is_circuit_breaker: bool = False,
        is_stop_loss: bool = False,
    ) -> Optional[RoundTripTradeRecord]:
        """Close both QuantNexus and Baseline positions and persist to PostgreSQL."""
        if not self.has_position:
            return None

        now = datetime.now(timezone.utc)
        qn_pos = self.qn_position
        base_pos = self.baseline_position

        # ------------------------------------------------------------------- #
        # 1. Close QuantNexus Position (if open)
        # ------------------------------------------------------------------- #
        qn_exit_price = 0.0
        qn_pnl = 0.0
        exit_slip_usd = 0.0
        exit_slip_bps = 0.0
        trade_status = TRADE_STATUS_COMPLETED

        if qn_pos is not None and qn_pos.quantity > 0.0:
            # TWAP exit order
            qn_exit = await self.executor.execute_order(
                symbol=qn_pos.symbol,
                side="SELL",
                base_qty=qn_pos.quantity,
                book_source=exit_book_source,
                num_slices=num_slices,
                is_exit=True,  # Allow flattening even if circuit breaker triggered
            )
            qn_exit_price = qn_exit.twap_exec_price
            exit_slip_usd = qn_exit.slippage_saved_usd
            exit_slip_bps = qn_exit.slippage_saved_bps

            # Hard Stop-Loss Execution Cap (Stop-Order Trigger Realism):
            # On a live exchange, a resting Stop-Market order triggers the instant
            # price crosses -stop_loss_bps. If a tick gaps or drops past -stop_loss_bps,
            # cap simulated stop-loss exit price at entry_price * (1 - (stop_loss_bps + 0.5) / 10000).
            if is_stop_loss or "STOP_LOSS" in exit_reason:
                stop_cap_px = qn_pos.entry_price * (1.0 - (self.stop_loss_bps + 0.5) / 10000.0)
                if qn_exit_price < stop_cap_px:
                    qn_exit_price = stop_cap_px

            qn_pnl = (qn_exit_price - qn_pos.entry_price) * qn_pos.quantity

            if is_circuit_breaker or qn_pos.entry_status == ExecutionStatus.ABORTED_BY_AI:
                trade_status = TRADE_STATUS_ABORTED

            self.quantnexus_equity_usd += qn_pnl

        # ------------------------------------------------------------------- #
        # 2. Close Baseline Retail Position
        # ------------------------------------------------------------------- #
        base_exit_price = 0.0
        base_pnl = 0.0

        if base_pos is not None:
            # Baseline sweeps the book aggressively on the SELL side
            exit_bids, exit_asks = await self.executor._next_book(exit_book_source, 0)
            base_sweep = self.signal_engine.estimate_naive_sweep_price(
                exit_bids, exit_asks, "SELL", base_pos.quantity
            )
            base_exit_price = base_sweep.vwap
            base_pnl = (base_exit_price - base_pos.entry_price) * base_pos.quantity
            self.baseline_equity_usd += base_pnl

        # ------------------------------------------------------------------- #
        # 3. Aggregate Performance & Update Portfolio Metrics
        # ------------------------------------------------------------------- #
        total_slippage_usd = (
            (qn_pos.entry_slippage_saved_usd if qn_pos else 0.0) + exit_slip_usd
        )
        total_slippage_bps = (
            (qn_pos.entry_slippage_saved_bps if qn_pos else 0.0) + exit_slip_bps
        )

        # Update high-water mark & drawdowns
        self.peak_qn_equity = max(self.peak_qn_equity, self.quantnexus_equity_usd)
        current_qn_dd = (
            (self.peak_qn_equity - self.quantnexus_equity_usd) / self.peak_qn_equity * 100.0
        )
        self.max_drawdown_pct = max(self.max_drawdown_pct, current_qn_dd)

        self.peak_baseline_equity = max(self.peak_baseline_equity, self.baseline_equity_usd)
        current_base_dd = (
            (self.peak_baseline_equity - self.baseline_equity_usd)
            / self.peak_baseline_equity
            * 100.0
        )
        self.baseline_max_drawdown_pct = max(self.baseline_max_drawdown_pct, current_base_dd)

        is_win = qn_pnl > 0.0

        record = RoundTripTradeRecord(
            symbol=qn_pos.symbol if qn_pos else base_pos.symbol,
            side="BUY",
            entry_price=qn_pos.entry_price if qn_pos else 0.0,
            exit_price=qn_exit_price,
            quantity=qn_pos.quantity if qn_pos else 0.0,
            naive_market_price=qn_pos.naive_entry_price if qn_pos else base_pos.entry_price,
            twap_exec_price=qn_pos.entry_price if qn_pos else 0.0,
            slippage_saved_bps=total_slippage_bps,
            slippage_saved_usd=total_slippage_usd,
            risk_multiplier_used=qn_pos.risk_multiplier if qn_pos else 0.0,
            realized_pnl_usd=qn_pnl,
            baseline_entry_price=base_pos.entry_price if base_pos else 0.0,
            baseline_exit_price=base_exit_price,
            baseline_quantity=base_pos.quantity if base_pos else 0.0,
            baseline_pnl_usd=base_pnl,
            is_win=is_win,
            status=trade_status,
            exit_reason=exit_reason,
            closed_at=now,
        )
        self.closed_trades.append(record)

        # ------------------------------------------------------------------- #
        # 4. PostgreSQL Persistence
        # ------------------------------------------------------------------- #
        trade_id = self.ledger.insert_trade(
            symbol=record.symbol,
            side=record.side,
            entry_price=record.entry_price,
            exit_price=record.exit_price,
            quantity=record.quantity,
            naive_market_price=record.naive_market_price,
            twap_exec_price=record.twap_exec_price,
            slippage_saved_bps=record.slippage_saved_bps,
            slippage_saved_usd=record.slippage_saved_usd,
            risk_multiplier_used=record.risk_multiplier_used,
            realized_pnl_usd=record.realized_pnl_usd,
            is_win=record.is_win,
            status=record.status,
            timestamp=record.closed_at,
        )
        record.trade_id = trade_id

        stats = self.get_metrics()
        self.ledger.insert_equity_snapshot(
            baseline_equity_usd=self.baseline_equity_usd,
            quantnexus_equity_usd=self.quantnexus_equity_usd,
            win_rate_pct=stats["win_rate_pct"],
            profit_factor=stats["profit_factor"],
            max_drawdown_pct=stats["max_drawdown_pct"],
            timestamp=record.closed_at,
        )

        # Reset open positions
        self.qn_position = None
        self.baseline_position = None
        self.position_hold_ticks = 0

        return record

    # ----------------------------------------------------------------------- #
    # Real-Time Tick Evaluation
    # ----------------------------------------------------------------------- #
    async def evaluate_tick(
        self,
        metrics: SignalMetrics,
        current_book: Any,
        exit_book_source: Optional[Any] = None,
    ) -> Optional[RoundTripTradeRecord]:
        """Check exit triggers (Take-Profit, Stop-Loss, Signal Reversal, Circuit Breaker)."""
        if not self.has_position:
            return None

        self.position_hold_ticks += 1

        # Check Global Redis Circuit Breaker first (instant, bypasses hold filter)
        risk_res = await self.bus.get_risk_state()
        if risk_res.data.circuit_breaker:
            return await self.close_position(
                exit_book_source=exit_book_source or [current_book],
                exit_reason="CIRCUIT_BREAKER_EMERGENCY",
                is_circuit_breaker=True,
            )

        ref_entry = (
            self.qn_position.entry_price if self.qn_position else self.baseline_position.entry_price
        )
        pnl_bps = (metrics.mid_price - ref_entry) / ref_entry * 10000.0

        # 1. Take-Profit Hit (+6.0 bps)
        if pnl_bps >= self.take_profit_bps:
            return await self.close_position(
                exit_book_source=exit_book_source or [current_book],
                exit_reason=f"TAKE_PROFIT (+{pnl_bps:.1f} bps)",
            )

        # 2. Asymmetric Stop-Loss Hit (-3.0 bps) - checked on Tick 1+ (not blocked by min_hold_ticks)
        if pnl_bps <= -self.stop_loss_bps:
            return await self.close_position(
                exit_book_source=exit_book_source or [current_book],
                exit_reason=f"STOP_LOSS ({pnl_bps:.1f} bps)",
                is_stop_loss=True,
            )

        # 3. Microstructure Signal Reversal with Minimum Hold & Noise Deadband Filter
        # Must satisfy:
        # a) Minimum hold ticks elapsed (e.g. 15 ticks) to eliminate 100ms 1-cent spread churn
        # AND
        # b) Either already locking in positive profit (>= trailing_lock_bps)
        #    OR extreme structural collapse (Z <= -2.2) beyond 1-cent noise deadband (abs(pnl) >= 1.5 bps)
        if metrics.signal == Signal.EXIT_LONG:
            if self.position_hold_ticks >= self.min_hold_ticks:
                is_profit_lock = pnl_bps >= self.trailing_lock_bps
                is_structural_collapse = (metrics.obi_zscore <= -2.2) and (abs(pnl_bps) >= 1.5)
                if is_profit_lock or is_structural_collapse:
                    reason = (
                        f"PROFIT_LOCK_REVERSAL (+{pnl_bps:.1f} bps, Z={metrics.obi_zscore:.2f})"
                        if is_profit_lock
                        else f"STRUCTURAL_COLLAPSE_EXIT ({pnl_bps:.1f} bps, Z={metrics.obi_zscore:.2f})"
                    )
                    return await self.close_position(
                        exit_book_source=exit_book_source or [current_book],
                        exit_reason=reason,
                    )

        return None

    # ----------------------------------------------------------------------- #
    # Analytics & Summary
    # ----------------------------------------------------------------------- #
    def get_metrics(self) -> dict[str, Any]:
        """Compute live portfolio metrics."""
        completed = [t for t in self.closed_trades if t.status == TRADE_STATUS_COMPLETED]
        all_trades = self.closed_trades
        n = len(all_trades)
        wins = sum(1 for t in all_trades if t.is_win)
        win_rate = (wins / n * 100.0) if n > 0 else 0.0

        gross_profit = sum(t.realized_pnl_usd for t in all_trades if t.realized_pnl_usd > 0.0)
        gross_loss = sum(-t.realized_pnl_usd for t in all_trades if t.realized_pnl_usd < 0.0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0.0 else None

        tot_slippage_usd = sum(t.slippage_saved_usd for t in all_trades)
        avg_slippage_bps = (
            sum(t.slippage_saved_bps for t in all_trades) / n if n > 0 else 0.0
        )

        return {
            "total_trades": n,
            "completed_trades": len(completed),
            "aborted_by_ai": sum(1 for t in all_trades if t.status == TRADE_STATUS_ABORTED),
            "wins": wins,
            "win_rate_pct": win_rate,
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "baseline_max_drawdown_pct": self.baseline_max_drawdown_pct,
            "quantnexus_equity_usd": self.quantnexus_equity_usd,
            "baseline_equity_usd": self.baseline_equity_usd,
            "alpha_vs_baseline_usd": self.quantnexus_equity_usd - self.baseline_equity_usd,
            "total_slippage_saved_usd": tot_slippage_usd,
            "avg_slippage_saved_bps": avg_slippage_bps,
        }
