"""
QuantNexus-L2 :: STEP 4 Asynchronous Agentic Risk Gating & LangGraph Verification
==================================================================================

Run:
    python -m pytest tests/test_step4_agentic_risk_loop.py -v -s

Verifies:
  * Test 1: Live Gemini 3.8 Flash LangGraph evaluation across 3 distinct market regimes:
      - Benign (ETF inflow) -> multiplier = 1.0, circuit_breaker = False
      - Macro Caution (CPI shock) -> 0.20 <= multiplier <= 0.60, circuit_breaker = False
      - Black-Swan (Exchange halt) -> multiplier = 0.0, circuit_breaker = True
  * Test 2: End-to-end integration: LangGraph agent updates Redis state and mid-flight
    preempts Fast-Loop TWAP execution engine at Slice #3 (0 tokens via cache).
  * Test 3: PostgreSQL risk_audit_logs verification, leaving records intact for DBeaver.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import delete, select

from fast_loop.alpha_signals import MicrostructureSignalEngine
from fast_loop.twap_executor import ExecutionStatus, TWAPExecutionEngine
from slow_loop.news_ingestor import NewsAndChaosIngestor
from slow_loop.risk_agent import AgenticRiskGatekeeper, RiskAssessment
from storage.pg_logger import (
    PostgresLedger,
    risk_audit_logs,
)
from storage.redis_bus import RedisStateBus


def _banner(title: str) -> None:
    print("\n" + "=" * 96 + f"\n  {title}\n" + "=" * 96)


def make_20_level_book(mid: float = 70_000.0, spread: float = 1.0):
    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    bids = [[best_bid - i * 5.0, 1.0] for i in range(20)]
    asks = [[best_ask + i * 5.0, 1.0] for i in range(20)]
    return bids, asks


@pytest.fixture
async def bus():
    b = RedisStateBus()
    await b.connect()
    # Reset to normal state
    await b.set_risk_state(risk_multiplier=1.0, circuit_breaker=False, reason="NORMAL")
    yield b
    await b.set_risk_state(risk_multiplier=1.0, circuit_breaker=False, reason="NORMAL")
    await b.close()


@pytest.fixture
def ledger():
    lg = PostgresLedger()
    lg.init_db()
    yield lg
    lg.close()


@pytest.fixture
def gatekeeper(bus: RedisStateBus, ledger: PostgresLedger):
    return AgenticRiskGatekeeper(bus=bus, ledger=ledger)


# --------------------------------------------------------------------------- #
# Test 1: Live Gemini LangGraph Three-Regime Transitions
# --------------------------------------------------------------------------- #
async def test_1_live_gemini_langgraph_three_regime_transitions(
    gatekeeper: AgenticRiskGatekeeper, bus: RedisStateBus
):
    _banner("TEST 1: Live Gemini 3.8 Flash LangGraph — Three Regime Transitions")

    # Clear previous audit logs so this test run leaves clean records
    with gatekeeper.ledger.engine.begin() as conn:
        conn.execute(delete(risk_audit_logs))

    # Pre-populate order book snapshot in Redis for Node 1 context enrichment
    await bus.update_orderbook_snapshot(
        symbol="BTC/USDT",
        best_bid=70_450.0,
        best_ask=70_450.5,
        obi=0.15,
        obi_zscore=0.45,
    )

    test_cases = [
        {
            "name": "Regime 1: Benign Structural Inflow",
            "headline": "Institutional spot Bitcoin ETF records $450M net daily inflow amid steady market liquidity",
            "expected_cb": False,
            "min_mult": 1.0,
            "max_mult": 1.0,
        },
        {
            "name": "Regime 2: Macro Volatility Caution",
            "headline": "US CPI inflation prints hotter than expected; Federal Reserve signals delay in rate cuts",
            "expected_cb": False,
            "min_mult": 0.20,
            "max_mult": 0.60,
        },
        {
            "name": "Regime 3: Black-Swan Severe Tail Risk",
            "headline": "BREAKING: Major crypto exchange halts all client withdrawals amid liquidity freeze and regulatory probe",
            "expected_cb": True,
            "min_mult": 0.0,
            "max_mult": 0.0,
        },
    ]

    for tc in test_cases:
        print(f"\n--- Evaluating [{tc['name']}] ---")
        print(f"Headline: \"{tc['headline']}\"")

        # Run through LangGraph
        result = await gatekeeper.evaluate_headline(
            headline=tc["headline"], source="LIVE_TEST", use_live_llm=True
        )

        print(f"  Engine Used          : {result['engine_used']}")
        print(f"  Tokens Used          : {result['tokens_used']}")
        print(f"  Evaluation Latency   : {result['latency_ms']:.2f} ms")
        print(f"  Assessed Risk Score  : {result['final_risk_score']:.2f}")
        print(f"  Category             : {result['final_category']}")
        print(f"  Assigned Multiplier  : {result['final_risk_multiplier']:.2f}")
        print(f"  Circuit Breaker      : {result['final_circuit_breaker']}")
        print(f"  Institutional Reason : \"{result['final_reasoning']}\"")
        print(f"  PostgreSQL Audit ID  : {result['audit_id']}")

        # Verify Redis bus state was updated to match
        redis_state = (await bus.get_risk_state()).data
        assert redis_state.circuit_breaker == tc["expected_cb"]
        assert tc["min_mult"] <= redis_state.risk_multiplier <= tc["max_mult"]

        # Assertions on LangGraph output
        assert result["final_circuit_breaker"] == tc["expected_cb"]
        assert tc["min_mult"] <= result["final_risk_multiplier"] <= tc["max_mult"]
        assert len(result["final_reasoning"]) > 0
        assert result["audit_id"] is not None


# --------------------------------------------------------------------------- #
# Test 2: End-to-End Slow Loop Preempts Fast-Loop TWAP Execution
# --------------------------------------------------------------------------- #
async def test_2_end_to_end_slow_loop_preempts_fast_loop_twap(
    gatekeeper: AgenticRiskGatekeeper, bus: RedisStateBus
):
    _banner("TEST 2: End-to-End Integration — Slow Loop Preempts Fast Loop TWAP")

    # 1. Reset Redis to NORMAL
    await bus.set_risk_state(risk_multiplier=1.0, circuit_breaker=False, reason="NORMAL")
    engine = MicrostructureSignalEngine()
    executor = TWAPExecutionEngine(bus=bus, signal_engine=engine, default_slices=5)

    base_qty = 2.5
    books = [make_20_level_book(mid=70_000.0) for _ in range(5)]
    black_swan_headline = (
        "BREAKING: Major crypto exchange halts all client withdrawals amid liquidity freeze and regulatory probe"
    )

    # 2. Asynchronous Slow-Loop hook: evaluates headline between slice 2 and slice 3
    async def slow_loop_alert_hook(slice_idx: int):
        if slice_idx == 2:  # slice 0 and 1 have completed; slice 2 is about to run
            print(f"\n[SLOW-LOOP AGENT ALERT] Injecting Black-Swan Event before Slice #{slice_idx + 1}...")
            # Evaluates headline (hits in-memory cache: 0 tokens, instant)
            res = await gatekeeper.evaluate_headline(
                headline=black_swan_headline,
                source="INTERCEPTION_HOOK",
                use_live_llm=True,
            )
            print(
                f"  -> Agent updated Redis: multiplier={res['final_risk_multiplier']}, "
                f"circuit_breaker={res['final_circuit_breaker']} ({res['engine_used']})"
            )

    # 3. Launch Fast-Loop TWAP Order
    result = await executor.execute_order(
        symbol="BTC/USDT",
        side="BUY",
        base_qty=base_qty,
        book_source=books,
        num_slices=5,
        before_slice_hook=slow_loop_alert_hook,
    )

    print(f"\n[FAST-LOOP TWAP INTERCEPTED]:")
    print(f"  Status                 : {result.status.value}")
    print(f"  Abort Reason           : {result.abort_reason}")
    print(f"  Filled Quantity        : {result.filled_qty:.2f} / {result.effective_qty:.2f} BTC")
    print(f"  Unfilled Quantity      : {result.unfilled_qty:.2f} BTC (PROTECTED)")
    print(f"  Protected Notional USD : ${result.protected_notional_usd:,.2f}")
    print(f"  Slices Executed        : {result.slices_filled} / {result.slices_total}")

    # Assertions
    assert result.status == ExecutionStatus.ABORTED_BY_AI
    assert result.circuit_breaker_triggered is True
    assert result.slices_filled == 2
    assert result.filled_qty == pytest.approx(1.0)  # Slices 1 & 2 filled
    assert result.unfilled_qty == pytest.approx(1.5)  # Slices 3, 4, 5 aborted
    assert result.protected_notional_usd > 0.0


# --------------------------------------------------------------------------- #
# Test 3: PostgreSQL Risk Audit Logs Verification for DBeaver
# --------------------------------------------------------------------------- #
async def test_3_verify_postgres_risk_audit_logs_for_dbeaver(ledger: PostgresLedger):
    _banner("TEST 3: Verify PostgreSQL risk_audit_logs (DBeaver Ready)")

    with ledger.engine.connect() as conn:
        records = list(
            conn.execute(
                select(risk_audit_logs).order_by(risk_audit_logs.c.id.asc())
            ).mappings()
        )

    print(f"\nFound {len(records)} risk audit log records in PostgreSQL:")
    print("=" * 110)
    print(f"{'ID':>4} | {'Timestamp (UTC)':>23} | {'Risk':>5} | {'Mult':>5} | {'CB':>5} | Headline")
    print("-" * 110)
    for r in records:
        ts_str = str(r["timestamp"])[:23]
        cb_str = "YES" if r["circuit_breaker_triggered"] else "NO"
        print(
            f"{r['id']:>4} | {ts_str:>23} | {r['risk_score']:>5.2f} | {r['new_risk_multiplier']:>5.2f} | {cb_str:>5} | {r['headline'][:50]}..."
        )
        print(f"       Reasoning: {r['reasoning']}\n")
    print("=" * 110)
    print(">>> All risk_audit_logs records left intact in PostgreSQL for DBeaver inspection!")

    # Assertions
    assert len(records) >= 3
    for r in records:
        assert r["headline"] is not None and len(r["headline"]) > 0
        assert r["reasoning"] is not None and len(r["reasoning"]) > 0
        assert 0.0 <= r["risk_score"] <= 1.0
        assert 0.0 <= r["new_risk_multiplier"] <= 1.0
