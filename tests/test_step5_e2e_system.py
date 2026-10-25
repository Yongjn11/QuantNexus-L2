"""
QuantNexus-L2 :: STEP 5 End-to-End System & Dashboard Integration Verification
==============================================================================

Run:
    python -m pytest tests/test_step5_e2e_system.py -v -s

Verifies:
  * Test 1: Live continuous streaming paper-trading engine (20 live ticks from Binance Vision,
    Redis order book snapshot updates, signal calculations, clean shutdown).
  * Test 2: Background daemon manager lifecycle (start, verify running, stop) and
    dashboard data provider validation.
"""

from __future__ import annotations

import asyncio
import time
import pytest

from run_live_engine import (
    LiveQuantNexusOrchestrator,
    is_live_engine_running,
    start_background_live_engine,
    stop_background_live_engine,
)
from storage.pg_logger import PostgresLedger
from storage.redis_bus import RedisStateBus


def _banner(title: str) -> None:
    print("\n" + "=" * 96 + f"\n  {title}\n" + "=" * 96)


@pytest.fixture
async def bus():
    b = RedisStateBus(key_prefix="test_live")
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
def ledger():
    lg = PostgresLedger()
    lg.init_db()
    yield lg
    lg.close()


# --------------------------------------------------------------------------- #
# Test 1: Live Continuous Streaming Paper-Trading Orchestrator
# --------------------------------------------------------------------------- #
@pytest.mark.live
async def test_1_live_orchestrator_continuous_stream(bus: RedisStateBus, ledger: PostgresLedger):
    _banner("TEST 1: Live Continuous Streaming Paper-Trading Orchestrator (20 Live Ticks)")

    orchestrator = LiveQuantNexusOrchestrator(
        symbol="BTC/USDT",
        entry_z=1.5,
        bus=bus,
        ledger=ledger,
    )

    t0 = time.time()
    # Stream exactly 20 live ticks
    await orchestrator.run(max_ticks=20)
    duration = time.time() - t0

    print(f"Streaming Complete:")
    print(f"  Ticks Ingested    : {orchestrator.tick_count}")
    print(f"  Duration          : {duration:.2f} s")
    print(f"  Last Tick Latency : {orchestrator.last_tick.tick_latency_ms:.2f} ms")
    print(f"  Last Mid Price    : ${orchestrator.last_tick.metrics.mid_price:,.2f}")
    print(f"  Last Weighted OBI : {orchestrator.last_tick.metrics.weighted_obi:+.4f}")
    print(f"  Last Z-Score      : {orchestrator.last_tick.metrics.obi_zscore:+.2f}σ")
    print(f"  Active Stream     : {orchestrator.is_running}")

    # Assertions
    assert orchestrator.tick_count == 20
    assert orchestrator.last_tick is not None
    assert orchestrator.last_tick.book["symbol"] == "BTC/USDT"
    assert not orchestrator.is_running  # Clean shutdown

    # Verify Redis snapshot was updated to the latest live tick
    snap_res = await bus.get_orderbook_snapshot("BTC/USDT")
    snap = snap_res.data
    assert snap is not None
    assert snap.mid_price == orchestrator.last_tick.metrics.mid_price
    assert snap.obi == pytest.approx(orchestrator.last_tick.metrics.weighted_obi)


# --------------------------------------------------------------------------- #
# Test 2: Background Daemon Manager & Dashboard Data Provider
# --------------------------------------------------------------------------- #
def test_2_background_daemon_and_dashboard_data(ledger: PostgresLedger):
    _banner("TEST 2: Background Daemon Lifecycle & Dashboard Data Integrity")

    # 1. Start daemon
    orch = start_background_live_engine(symbol="BTC/USDT", entry_z=1.5)
    time.sleep(1.0)  # Allow thread to initialize and connect
    assert is_live_engine_running() is True
    print("[DAEMON STARTED]: Live engine thread active in background.")

    # Let daemon stream a few ticks
    time.sleep(2.0)
    summary = orch.get_summary()
    print(f"  Daemon Uptime      : {summary['uptime_seconds']:.1f}s")
    print(f"  Ticks Processed    : {summary['tick_count']}")
    print(f"  Current Mid Price  : ${summary['mid_price']:,.2f}")
    print(f"  Running Status     : {summary['is_running']}")

    # 2. Stop daemon
    stop_background_live_engine()
    assert is_live_engine_running() is False
    print("[DAEMON STOPPED]: Live engine gracefully joined.")

    # 3. Verify Dashboard Data-Fetching Methods with Zero Errors
    print("\nVerifying Dashboard Data Providers:")
    ps = ledger.get_portfolio_summary()
    assert isinstance(ps, dict)
    print(f"  Portfolio Summary Keys : {list(ps.keys())}")

    trades = ledger.get_recent_trades(limit=10)
    assert isinstance(trades, list)
    print(f"  Recent Trades Count    : {len(trades)}")

    audits = ledger.get_recent_risk_audits(limit=10)
    assert isinstance(audits, list)
    print(f"  Risk Audits Count      : {len(audits)}")

    curve = ledger.get_equity_curve(limit=50)
    assert isinstance(curve, list)
    print(f"  Equity Curve Points    : {len(curve)}")
