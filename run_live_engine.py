"""
QuantNexus-L2 :: Continuous Live Paper-Trading Orchestrator
===========================================================

Orchestrates the entire live trading stack in real time:
  1. Streams live Top-20 L2 order book updates via ``L2OrderBookStream``.
  2. Computes microstructural alpha signals (Weighted OBI, Rolling Z-score) on every tick.
  3. Updates ``RedisStateBus`` sub-millisecond snapshot.
  4. Manages the dual-account portfolio lifecycle (QuantNexus vs. Baseline Retail Bot)
     via ``DualPortfolioManager`` and ``TWAPExecutionEngine``.
  5. Enforces real-time Take-Profit, Stop-Loss, Signal Reversal, and AI Circuit Breaker exits.
  6. Persists every trade execution and A/B portfolio equity update to PostgreSQL.

Provides both a standalone CLI entry point (``python run_live_engine.py``) and a
thread-safe background singleton daemon (``start_background_live_engine()``) for the
Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from typing import Any, Optional

from dotenv import load_dotenv

from fast_loop.alpha_signals import MicrostructureSignalEngine, Signal, SignalMetrics
from fast_loop.orderbook_ws import L2OrderBookStream, TickResult
from fast_loop.portfolio_manager import DualPortfolioManager, RoundTripTradeRecord
from fast_loop.twap_executor import TWAPExecutionEngine
from storage.pg_logger import PostgresLedger
from storage.redis_bus import RedisStateBus

load_dotenv()
log = logging.getLogger("quantnexus.live_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class LiveQuantNexusOrchestrator:
    """Continuous real-time paper trading engine."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        entry_z: float = 1.5,  # Adaptive sensitivity for regular live paper-trades
        exit_z: float = -1.0,
        take_profit_bps: float = 6.0,
        stop_loss_bps: float = 3.0,
        trailing_lock_bps: float = 2.0,
        min_hold_ticks: int = 15,
        base_qty: float = 1.5,
        bus: Optional[RedisStateBus] = None,
        ledger: Optional[PostgresLedger] = None,
    ) -> None:
        self.symbol = symbol
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps
        self.trailing_lock_bps = trailing_lock_bps
        self.min_hold_ticks = min_hold_ticks
        self.base_qty = base_qty

        self.bus = bus or RedisStateBus()
        self.ledger = ledger or PostgresLedger()
        self.ledger.init_db()

        self.signal_engine = MicrostructureSignalEngine(
            depth=20,
            decay_lambda=0.15,
            window=100,
            warmup_ticks=15,
            entry_z=entry_z,
            exit_z=exit_z,
            min_sigma=0.05,
        )
        self.executor = TWAPExecutionEngine(bus=self.bus, signal_engine=self.signal_engine)
        self.portfolio_manager = DualPortfolioManager(
            bus=self.bus,
            ledger=self.ledger,
            executor=self.executor,
            signal_engine=self.signal_engine,
            take_profit_bps=take_profit_bps,
            stop_loss_bps=stop_loss_bps,
            trailing_lock_bps=trailing_lock_bps,
            min_hold_ticks=min_hold_ticks,
            base_qty=base_qty,
            hydrate_from_db=True,
        )

        self.stream: Optional[L2OrderBookStream] = None
        self._stop_requested = False
        self.tick_count = 0
        self.last_tick: Optional[TickResult] = None
        self.last_trade: Optional[RoundTripTradeRecord] = None
        self.start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self.stream is not None and not self._stop_requested

    def stop(self) -> None:
        self._stop_requested = True
        if self.stream is not None:
            self.stream.stop()

    async def on_tick(self, tick: TickResult) -> None:
        """Executed on every live WebSocket tick frame."""
        self.tick_count += 1
        self.last_tick = tick
        m = tick.metrics
        bids = tick.book["bids"]
        asks = tick.book["asks"]
        current_book = (bids, asks)

        # 1. If in position, monitor real-time exit conditions (TP, SL, Reversal, Circuit Breaker)
        if self.portfolio_manager.has_position:
            exit_books = [current_book] * 5
            closed_trade = await self.portfolio_manager.evaluate_tick(
                metrics=m,
                current_book=current_book,
                exit_book_source=exit_books,
            )
            if closed_trade is not None:
                self.last_trade = closed_trade
                log.info(
                    "CLOSED TRADE [%s]: QN PnL = $%.2f | Baseline PnL = $%.2f | Saved = $%.2f (%s)",
                    closed_trade.symbol,
                    closed_trade.realized_pnl_usd,
                    closed_trade.baseline_pnl_usd,
                    closed_trade.slippage_saved_usd,
                    closed_trade.exit_reason,
                )

        # 2. If flat, check for LONG_ENTRY signal
        elif m.signal == Signal.LONG_ENTRY:
            log.info(
                "SIGNAL LONG_ENTRY (Z=%.2f, wOBI=%.4f, Spread=%.2fbps) -> Initiating TWAP...",
                m.obi_zscore,
                m.weighted_obi,
                m.spread_bps,
            )
            entry_books = [current_book] * 5
            entry_res = await self.portfolio_manager.open_position(
                symbol=self.symbol,
                book_source=entry_books,
                base_qty=self.base_qty,
                num_slices=5,
            )
            if entry_res is not None:
                log.info(
                    "POSITION OPENED: %s %s @ $%.2f (status: %s, risk mult: %.2f)",
                    entry_res.side,
                    entry_res.filled_qty,
                    entry_res.twap_exec_price,
                    entry_res.status.value,
                    entry_res.risk_multiplier_used,
                )

    async def run(self, max_ticks: Optional[int] = None, duration_sec: Optional[float] = None) -> None:
        """Run continuous live market streaming and paper-trading loop."""
        self._stop_requested = False
        self.start_time = time.time()
        await self.bus.connect()

        self.stream = L2OrderBookStream(
            symbol=self.symbol,
            engine=self.signal_engine,
            bus=self.bus,
            max_reconnect_attempts=10,
        )

        log.info("Starting QuantNexus-L2 Live Engine on %s (entry_z=%.2f)...", self.symbol, self.entry_z)

        try:
            async for tick in self.stream.stream_ticks(max_ticks=max_ticks):
                await self.on_tick(tick)

                if duration_sec is not None and (time.time() - self.start_time) >= duration_sec:
                    log.info("Duration limit of %.1f seconds reached.", duration_sec)
                    break

                if self._stop_requested:
                    break
        except asyncio.CancelledError:
            log.info("Live engine task cancelled.")
        finally:
            self.stop()
            log.info("QuantNexus-L2 Live Engine stopped. Total ticks: %d", self.tick_count)

    async def execute_demo_trade(self) -> Optional[RoundTripTradeRecord]:
        """Instant demo trigger: forces an immediate synthetic round-trip trade cycle."""
        await self.bus.connect()
        # Fetch current real top of book if available, else standard price
        snap_res = await self.bus.get_orderbook_snapshot(self.symbol)
        mid = snap_res.data.mid_price if snap_res.data else 70_000.0

        # Construct realistic books
        best_bid = mid - 0.5
        best_ask = mid + 0.5
        entry_bids = [[best_bid - i * 5.0, 1.0] for i in range(20)]
        entry_asks = [[best_ask + i * 5.0, 1.0] for i in range(20)]
        entry_books = [(entry_bids, entry_asks)] * 5

        # Open Long
        await self.portfolio_manager.open_position(
            symbol=self.symbol,
            book_source=entry_books,
            base_qty=self.base_qty,
            num_slices=5,
        )

        # Simulate price rising +15 bps for Take-Profit
        exit_mid = mid * (1.0 + self.take_profit_bps / 10000.0 + 0.0002)
        exit_bids = [[exit_mid - 0.5 - i * 5.0, 1.0] for i in range(20)]
        exit_asks = [[exit_mid + 0.5 + i * 5.0, 1.0] for i in range(20)]
        exit_books = [(exit_bids, exit_asks)] * 5

        record = await self.portfolio_manager.close_position(
            exit_book_source=exit_books,
            exit_reason=f"DEMO_TRIGGER: TAKE_PROFIT (+{self.take_profit_bps:.1f} bps)",
        )
        self.last_trade = record
        return record

    def get_summary(self) -> dict[str, Any]:
        """Live status summary dictionary for the dashboard."""
        last_m = self.last_tick.metrics if self.last_tick else None
        p_metrics = self.portfolio_manager.get_metrics()
        return {
            "symbol": self.symbol,
            "is_running": self.is_running,
            "tick_count": self.tick_count,
            "uptime_seconds": (time.time() - self.start_time) if self.start_time else 0.0,
            "mid_price": last_m.mid_price if last_m else 0.0,
            "best_bid": last_m.best_bid if last_m else 0.0,
            "best_ask": last_m.best_ask if last_m else 0.0,
            "spread_bps": last_m.spread_bps if last_m else 0.0,
            "weighted_obi": last_m.weighted_obi if last_m else 0.0,
            "obi_zscore": last_m.obi_zscore if last_m else 0.0,
            "signal": last_m.signal.value if last_m else "NEUTRAL",
            "tick_latency_ms": self.last_tick.tick_latency_ms if self.last_tick else 0.0,
            "has_position": self.portfolio_manager.has_position,
            "position_qty": self.portfolio_manager.qn_position.quantity if self.portfolio_manager.qn_position else 0.0,
            "position_entry": self.portfolio_manager.qn_position.entry_price if self.portfolio_manager.qn_position else 0.0,
            **p_metrics,
        }


# --------------------------------------------------------------------------- #
# Thread-Safe Background Daemon Manager (Singleton for Streamlit UI)
# --------------------------------------------------------------------------- #
_daemon_lock = threading.RLock()
_daemon_thread: Optional[threading.Thread] = None
_daemon_loop: Optional[asyncio.AbstractEventLoop] = None
_orchestrator_instance: Optional[LiveQuantNexusOrchestrator] = None


def get_orchestrator(symbol: str = "BTC/USDT", entry_z: float = 1.5) -> LiveQuantNexusOrchestrator:
    """Retrieve or initialize the global orchestrator instance."""
    global _orchestrator_instance
    with _daemon_lock:
        if _orchestrator_instance is None:
            _orchestrator_instance = LiveQuantNexusOrchestrator(symbol=symbol, entry_z=entry_z)
        return _orchestrator_instance


def _run_daemon_loop(orch: LiveQuantNexusOrchestrator) -> None:
    """Worker target for background daemon thread."""
    global _daemon_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _daemon_loop = loop
    try:
        loop.run_until_complete(orch.run())
    except asyncio.CancelledError:
        log.info("Daemon loop cancelled.")
    except Exception as exc:
        log.warning("Daemon loop exited with exception: %r", exc)
    finally:
        loop.close()
        _daemon_loop = None


def start_background_live_engine(
    symbol: str = "BTC/USDT", entry_z: float = 1.5
) -> LiveQuantNexusOrchestrator:
    """Start the background live streaming daemon thread if not already running."""
    global _daemon_thread, _orchestrator_instance
    with _daemon_lock:
        orch = get_orchestrator(symbol=symbol, entry_z=entry_z)
        if _daemon_thread is None or not _daemon_thread.is_alive():
            orch._stop_requested = False
            t = threading.Thread(
                target=_run_daemon_loop,
                args=(orch,),
                name="QuantNexus-LiveEngine-Daemon",
                daemon=True,
            )
            t.start()
            _daemon_thread = t
            log.info("Background Live Engine daemon thread started.")
        return orch


def stop_background_live_engine() -> None:
    """Gracefully signal the background live engine to stop."""
    global _daemon_thread, _orchestrator_instance, _daemon_loop
    with _daemon_lock:
        if _orchestrator_instance is not None:
            _orchestrator_instance.stop()
        if _daemon_loop is not None and _daemon_loop.is_running():
            for task in asyncio.all_tasks(_daemon_loop):
                _daemon_loop.call_soon_threadsafe(task.cancel)
        if _daemon_thread is not None and _daemon_thread.is_alive():
            _daemon_thread.join(timeout=3.0)
            _daemon_thread = None
            log.info("Background Live Engine daemon thread stopped.")


def is_live_engine_running() -> bool:
    """Check if the background live engine is actively running."""
    with _daemon_lock:
        return _daemon_thread is not None and _daemon_thread.is_alive()


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantNexus-L2 Continuous Paper Trading Engine")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol (default: BTC/USDT)")
    parser.add_argument("--entry-z", type=float, default=1.5, help="Entry Z-score threshold (default: 1.5)")
    parser.add_argument("--duration-sec", type=float, default=None, help="Max run duration in seconds")
    parser.add_argument("--max-ticks", type=int, default=None, help="Max ticks to stream")
    args = parser.parse_args()

    orchestrator = LiveQuantNexusOrchestrator(
        symbol=args.symbol,
        entry_z=args.entry_z,
    )
    try:
        asyncio.run(orchestrator.run(max_ticks=args.max_ticks, duration_sec=args.duration_sec))
    except KeyboardInterrupt:
        orchestrator.stop()
        print("\nShutdown by user.")
