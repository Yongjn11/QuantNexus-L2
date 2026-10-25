"""
QuantNexus-L2 :: STEP 1 Infrastructure Verification
====================================================

Run:
    python -m pytest tests/test_step1_infra.py -v -s

Requires the Docker stack to be up:
    docker compose up -d
"""

from __future__ import annotations

import statistics
from datetime import datetime

import pytest
from sqlalchemy import delete, inspect

from storage.pg_logger import (
    TRADE_STATUS_COMPLETED,
    PostgresLedger,
    ab_portfolio_equity,
    risk_audit_logs,
    trade_executions,
)
from storage.redis_bus import RedisStateBus

READ_LATENCY_SLA_MS = 5.0
BENCH_ITERATIONS = 1000


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
async def bus():
    b = RedisStateBus()
    try:
        await b.connect()
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Cannot connect to Redis at {b.host}:{b.port} -> {exc}")
    yield b
    await b.close()


@pytest.fixture
def ledger():
    lg = PostgresLedger()
    created: dict = {"trade": [], "equity": [], "audit": []}
    lg._test_created = created  # track rows for teardown
    yield lg
    # Teardown: remove only rows inserted by this test so the ledger stays clean.
    with lg.engine.begin() as conn:
        if created["trade"]:
            conn.execute(delete(trade_executions).where(trade_executions.c.id.in_(created["trade"])))
        if created["equity"]:
            conn.execute(
                delete(ab_portfolio_equity).where(ab_portfolio_equity.c.id.in_(created["equity"]))
            )
        if created["audit"]:
            conn.execute(delete(risk_audit_logs).where(risk_audit_logs.c.id.in_(created["audit"])))
    lg.close()


def _banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
# Test 1 — Redis sub-ms latency & risk-state gating
# --------------------------------------------------------------------------- #
async def test_1_redis_latency_and_risk_gating(bus: RedisStateBus):
    _banner("TEST 1: Redis Shared-State Bus — Sub-ms Latency & Risk Gating")

    try:
        # 1. Default state (key absent -> defaults)
        await bus.clear_risk_state()
        res = await bus.get_risk_state()
        default = res.data
        print(f"[default ] {default.to_dict()}  (read {res.latency_ms:.4f} ms)")
        assert default.is_default is True
        assert default.risk_multiplier == 1.0
        assert default.circuit_breaker is False
        assert default.reason == "NORMAL"

        # 2. Black-Swan event from Slow Loop
        w = await bus.set_risk_state(
            risk_multiplier=0.0, circuit_breaker=True, reason="BLACK_SWAN: exchange insolvency rumor"
        )
        print(f"[write   ] Black-Swan state written in {w.latency_ms:.4f} ms")

        # 3. Fast Loop reads it back
        r = await bus.get_risk_state()
        state = r.data
        print(f"[readback] {state.to_dict()}")
        print(f">>> REDIS RISK-STATE READ LATENCY: {r.latency_ms:.4f} ms "
              f"(SLA < {READ_LATENCY_SLA_MS} ms)")
        assert state.is_default is False
        assert state.risk_multiplier == 0.0
        assert state.circuit_breaker is True
        assert state.reason.startswith("BLACK_SWAN")
        assert r.latency_ms < READ_LATENCY_SLA_MS

        # 4. Statistical latency profile (single reads can be noisy)
        lat = [(await bus.get_risk_state()).latency_ms for _ in range(BENCH_ITERATIONS)]
        lat.sort()
        p50 = statistics.median(lat)
        p99 = lat[int(0.99 * len(lat)) - 1]
        print(f"[bench   ] {BENCH_ITERATIONS} reads -> min {lat[0]:.4f} | p50 {p50:.4f} | "
              f"mean {statistics.fmean(lat):.4f} | p99 {p99:.4f} | max {lat[-1]:.4f} ms")
        assert p50 < 1.0, f"p50 read latency not sub-millisecond: {p50:.4f} ms"
        assert p99 < READ_LATENCY_SLA_MS

    finally:
        # 5. Reset to NORMAL regardless of outcome
        await bus.set_risk_state(risk_multiplier=1.0, circuit_breaker=False, reason="NORMAL")

    final = (await bus.get_risk_state()).data
    print(f"[reset   ] {final.to_dict()}")
    assert final.risk_multiplier == 1.0 and final.circuit_breaker is False


async def test_1b_redis_orderbook_snapshot_roundtrip(bus: RedisStateBus):
    _banner("TEST 1b: Redis Order-Book Snapshot Round-Trip")
    symbol = "TEST:BTC/USDT"
    try:
        w = await bus.update_orderbook_snapshot(
            symbol, best_bid=67450.10, best_ask=67450.60, obi=0.3125, obi_zscore=2.41
        )
        r = await bus.get_orderbook_snapshot(symbol)
        snap = r.data
        print(f"[write] {w.latency_ms:.4f} ms | [read] {r.latency_ms:.4f} ms")
        print(f"[snap ] {snap.to_dict()}")
        assert snap.symbol == symbol
        assert snap.best_bid == 67450.10 and snap.best_ask == 67450.60
        assert snap.obi == 0.3125 and snap.obi_zscore == 2.41
        assert snap.mid_price == pytest.approx(67450.35)
        assert snap.spread_bps == pytest.approx(0.5 / 67450.35 * 1e4)
        assert r.latency_ms < READ_LATENCY_SLA_MS
        assert (await bus.get_orderbook_snapshot("TEST:NOPE")).data is None
        with pytest.raises(ValueError):
            await bus.update_orderbook_snapshot(symbol, 100.0, 99.0, 0.0, 0.0)  # crossed
    finally:
        await bus.delete_orderbook_snapshot(symbol)


# --------------------------------------------------------------------------- #
# Test 2 — PostgreSQL schema & A/B ledger insertion
# --------------------------------------------------------------------------- #
def test_2_postgres_schema_and_ab_ledger(ledger: PostgresLedger):
    _banner("TEST 2: PostgreSQL Ledger — Schema & A/B Insertion")
    created = ledger._test_created

    # 1. Schema
    tables = ledger.init_db()
    insp = inspect(ledger.engine)
    live = set(insp.get_table_names())
    print(f"[schema] managed={tables} | present={sorted(live & set(tables))}")
    for t in ("trade_executions", "ab_portfolio_equity", "risk_audit_logs"):
        assert t in live, f"missing table {t}"

    expected_trade_cols = {
        "id", "timestamp", "symbol", "side", "entry_price", "exit_price", "quantity",
        "naive_market_price", "twap_exec_price", "slippage_saved_bps", "slippage_saved_usd",
        "risk_multiplier_used", "realized_pnl_usd", "is_win", "status",
    }
    trade_cols = {c["name"] for c in insp.get_columns("trade_executions")}
    assert expected_trade_cols == trade_cols
    ts_type = next(c for c in insp.get_columns("trade_executions") if c["name"] == "timestamp")["type"]
    assert getattr(ts_type, "timezone", False) is True, "timestamp must be TIMESTAMPTZ"

    # 2. Trade: BUY 1.5 BTC, TWAP saved 12.5 bps vs naive market sweep, +$145.50 PnL
    qty = 1.5
    naive_px = 67500.00
    twap_px = naive_px * (1 - 12.5 / 1e4)          # 67415.625
    saved_usd = (naive_px - twap_px) * qty          # 126.5625
    entry_px = twap_px
    exit_px = entry_px + 145.50 / qty               # +$145.50 realized
    trade_id = ledger.insert_trade(
        symbol="BTC/USDT", side="buy",
        entry_price=entry_px, exit_price=exit_px, quantity=qty,
        naive_market_price=naive_px, twap_exec_price=twap_px,
        slippage_saved_bps=12.5, slippage_saved_usd=saved_usd,
        risk_multiplier_used=1.0, realized_pnl_usd=145.50,
        status=TRADE_STATUS_COMPLETED,
    )
    created["trade"].append(trade_id)

    # 3. A/B equity snapshot
    eq_id = ledger.insert_equity_snapshot(
        baseline_equity_usd=99_980.00, quantnexus_equity_usd=100_145.50,
        win_rate_pct=100.0, profit_factor=None, max_drawdown_pct=0.02,
    )
    created["equity"].append(eq_id)

    # 4. Risk audit
    audit_id = ledger.insert_risk_audit(
        headline="SEC announces emergency review of major stablecoin issuer reserves",
        risk_score=0.87, new_risk_multiplier=0.25, circuit_breaker_triggered=False,
        reasoning="Elevated contagion risk; de-lever to 25% notional, no full halt.",
    )
    created["audit"].append(audit_id)

    # 5. Read back & verify integrity
    t = ledger.get_trade(trade_id)
    e = ledger.get_equity_snapshot(eq_id)
    a = ledger.get_risk_audit(audit_id)
    print(f"[trade ] {t}")
    print(f"[equity] {e}")
    print(f"[audit ] {a}")

    assert t["symbol"] == "BTC/USDT" and t["side"] == "BUY"
    assert t["status"] == "COMPLETED"
    assert t["slippage_saved_bps"] == pytest.approx(12.5)
    assert t["slippage_saved_usd"] == pytest.approx(126.5625)
    assert t["twap_exec_price"] == pytest.approx(67415.625)
    assert t["realized_pnl_usd"] == pytest.approx(145.50)
    assert (t["exit_price"] - t["entry_price"]) * t["quantity"] == pytest.approx(145.50)
    assert t["is_win"] is True  # derived from PnL
    assert isinstance(t["timestamp"], datetime) and t["timestamp"].tzinfo is not None

    assert e["baseline_equity_usd"] == pytest.approx(99_980.00)
    assert e["quantnexus_equity_usd"] == pytest.approx(100_145.50)
    assert e["quantnexus_equity_usd"] - e["baseline_equity_usd"] == pytest.approx(165.50)

    assert a["risk_score"] == pytest.approx(0.87)
    assert a["new_risk_multiplier"] == pytest.approx(0.25)
    assert a["circuit_breaker_triggered"] is False
    assert "stablecoin" in a["headline"]

    # 6. Constraint enforcement
    with pytest.raises(ValueError):
        ledger.insert_trade("BTC/USDT", "BUY", 1.0, 1.0, 1.0, status="BOGUS")

    # 7. Portfolio summary
    summary = ledger.get_portfolio_summary()
    print(f"[summary] {summary}")
    assert summary["completed_trades"] >= 1
    assert summary["latest_equity"] is not None
    assert summary["total_slippage_saved_usd"] >= 126.5625 - 1e-9
