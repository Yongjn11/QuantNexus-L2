"""
QuantNexus-L2 :: STEP 2 Order-Book Engine & Microstructure Signal Verification
==============================================================================

Run:
    python -m pytest tests/test_step2_orderbook_signals.py -v -s

Test 3 requires outbound internet (live BTC/USDT WebSocket) and the Redis
container (``docker compose up -d``). Deselect it offline with ``-m "not live"``.
"""

from __future__ import annotations

import asyncio
import math
import statistics

import pytest

from fast_loop.alpha_signals import MicrostructureSignalEngine, Signal
from fast_loop.orderbook_ws import (
    BinanceDepthParser,
    BybitBookParser,
    L2OrderBookStream,
)
from storage.redis_bus import RedisStateBus

LIVE_TICKS = 20
TICK_LATENCY_SLA_MS = 5.0
MID = 70_000.0
TICK = 0.5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _banner(title: str) -> None:
    print("\n" + "=" * 96 + f"\n  {title}\n" + "=" * 96)


def synthetic_book(k: int, bid_wall: float = 1.0, ask_wall: float = 1.0, wall_levels: int = 5):
    """Deterministic 20-level book centred on $70,000.

    Quantities carry small deterministic ±5% noise (so the rolling σ is non-zero);
    levels 1..``wall_levels`` are multiplied by the wall factor.
    """
    bids, asks = [], []
    for i in range(20):
        qb = 1.0 * (1.0 + 0.05 * math.sin(1.3 * k + 0.7 * i))
        qa = 1.0 * (1.0 + 0.05 * math.cos(1.1 * k + 0.9 * i))
        if i < wall_levels:
            qb *= bid_wall
            qa *= ask_wall
        bids.append([MID - TICK * (i + 1), qb])
        asks.append([MID + TICK * (i + 1), qa])
    return bids, asks


@pytest.fixture
async def bus():
    b = RedisStateBus()
    try:
        await b.connect()
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Cannot connect to Redis at {b.host}:{b.port} -> {exc}")
    yield b
    await b.close()


# --------------------------------------------------------------------------- #
# Test 1 — deterministic OBI / Z-score / signal generation
# --------------------------------------------------------------------------- #
def test_1_deterministic_obi_and_zscore_signals():
    _banner("TEST 1: Deterministic Weighted-OBI, Rolling Z-Score & Signal Generation")
    eng = MicrostructureSignalEngine(decay_lambda=0.15, warmup_ticks=15, entry_z=1.8, exit_z=-1.0)

    # Weights sanity: exponential decay, top level dominates.
    assert eng.weights[0] == 1.0
    assert all(a > b for a, b in zip(eng.weights, eng.weights[1:]))
    assert eng.weights[19] == pytest.approx(math.exp(-0.15 * 19))

    history = []
    # Ticks 1–25: balanced book.
    for k in range(25):
        history.append(eng.update(*synthetic_book(k), timestamp_ms=k))
    # Ticks 26–30: heavy bid walls on levels 1–5.
    for k in range(25, 30):
        history.append(eng.update(*synthetic_book(k, bid_wall=10.0), timestamp_ms=k))

    print(f"{'tick':>4} {'mid':>10} {'spread':>7} {'rawOBI':>8} {'wOBI':>8} {'Z':>9}  signal")
    for m in history:
        print(f"{m.tick_index:>4} {m.mid_price:>10.2f} {m.spread_bps:>7.4f} {m.raw_obi:>8.4f} "
              f"{m.weighted_obi:>8.4f} {m.obi_zscore:>9.3f}  {m.signal.value}")

    balanced = history[:25]
    # Warm-up gating: no signal and Z=0 before 15 prior observations.
    assert all(m.signal == Signal.NEUTRAL and m.obi_zscore == 0.0 for m in balanced[:15])
    assert all(not m.warmed_up for m in balanced[:15]) and balanced[15].warmed_up
    # Balanced regime: OBI ≈ 0, no false entries.
    assert all(abs(m.weighted_obi) < 0.1 for m in balanced)
    assert all(m.signal != Signal.LONG_ENTRY for m in balanced)
    # Top-of-book math.
    m0 = balanced[0]
    assert m0.best_bid == MID - TICK and m0.best_ask == MID + TICK
    assert m0.mid_price == MID
    assert m0.spread_bps == pytest.approx(1.0 / MID * 1e4)

    wall = history[25]
    assert wall.weighted_obi > 0.5
    assert wall.weighted_obi > wall.raw_obi  # depth weighting amplifies top-of-book walls
    assert wall.obi_zscore >= 1.8
    assert wall.signal == Signal.LONG_ENTRY == "LONG_ENTRY"
    assert max(m.obi_zscore for m in history[25:]) >= 1.8

    # Heavy ask wall → exit.
    ask_wall = eng.update(*synthetic_book(30, ask_wall=10.0), timestamp_ms=30)
    print(f"{ask_wall.tick_index:>4} {ask_wall.mid_price:>10.2f} {ask_wall.spread_bps:>7.4f} "
          f"{ask_wall.raw_obi:>8.4f} {ask_wall.weighted_obi:>8.4f} {ask_wall.obi_zscore:>9.3f}  "
          f"{ask_wall.signal.value}   <-- ask wall")
    assert ask_wall.weighted_obi < -0.5
    assert ask_wall.obi_zscore <= -1.0
    assert ask_wall.signal == Signal.EXIT_LONG

    # σ-floor regression: with the textbook (σ+ε) denominator, the same ±5% noise
    # on a balanced book produces false signals; the floor suppresses them.
    raw_eng = MicrostructureSignalEngine(min_sigma=0.0)
    raw_sigs = [raw_eng.update(*synthetic_book(k)).signal for k in range(25)]
    n_false = sum(s != Signal.NEUTRAL for s in raw_sigs)
    print(f"[sigma-floor] min_sigma=0 -> {n_false} false signals on balanced noise; "
          f"min_sigma={eng.min_sigma} -> 0")
    assert n_false > 0

    # Spread filter blocks entries but never exits.
    assert eng.evaluate_signal(5.0, spread_bps=eng.max_spread_bps + 1) == Signal.NEUTRAL
    assert eng.evaluate_signal(-5.0, spread_bps=1e3) == Signal.EXIT_LONG

    # Input validation.
    with pytest.raises(ValueError):
        eng.update([[70001.0, 1.0]], [[70000.0, 1.0]])  # crossed
    with pytest.raises(ValueError):
        eng.update([], [[70000.0, 1.0]])


def test_1b_exchange_message_normalizers():
    _banner("TEST 1b: Binance / Bybit Message Normalisation (fallback-path correctness)")
    bn = BinanceDepthParser("BTC/USDT").parse(
        {"lastUpdateId": 1, "bids": [["70000.1", "0.5"], ["70000.0", "1.2"]],
         "asks": [["70000.2", "0.3"], ["70000.5", "2.0"]]}
    )
    assert bn["symbol"] == "BTC/USDT" and isinstance(bn["timestamp_ms"], int)
    assert bn["bids"][0] == [70000.1, 0.5] and bn["asks"][1] == [70000.5, 2.0]

    p = BybitBookParser("BTC/USDT")
    assert p.parse({"success": True, "op": "subscribe"}) is None
    assert p.parse({"topic": "orderbook.50.BTCUSDT", "type": "delta", "ts": 1,
                    "data": {"b": [["1", "1"]], "a": [], "u": 5}}) is None  # delta before snapshot
    snap = p.parse({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "ts": 1000,
                    "data": {"s": "BTCUSDT", "u": 10,
                             "b": [["69999", "1"], ["70000", "2"], ["69998", "3"]],
                             "a": [["70002", "1"], ["70001", "4"]]}})
    assert snap["bids"][0] == [70000.0, 2.0] and snap["asks"][0] == [70001.0, 4.0]
    delta = p.parse({"topic": "orderbook.50.BTCUSDT", "type": "delta", "ts": 1100,
                     "data": {"u": 11, "b": [["70000", "0"], ["70000.5", "0.7"]],
                              "a": [["70001", "0"]]}})
    assert delta["timestamp_ms"] == 1100
    assert delta["bids"][:2] == [[70000.5, 0.7], [69999.0, 1.0]]  # 70000 removed, new best
    assert delta["asks"] == [[70002.0, 1.0]]
    print(f"[binance] {bn}\n[bybit  ] {delta}")


# --------------------------------------------------------------------------- #
# Test 2 — naive market sweep / slippage
# --------------------------------------------------------------------------- #
def test_2_naive_market_sweep_slippage_math():
    _banner("TEST 2: Naive Market-Order Sweep — VWAP & Slippage")
    asks = [[70_000.0, 0.5], [70_010.0, 1.0], [70_050.0, 2.0]]
    asks += [[70_050.0 + 10.0 * i, 1.0] for i in range(1, 18)]  # levels 4–20
    bids = [[69_990.0 - 10.0 * i, 1.0] for i in range(20)]
    assert len(asks) == len(bids) == 20

    est = MicrostructureSignalEngine.estimate_naive_sweep_price(bids, asks, "BUY", 2.5)
    print(f"[BUY 2.5] {est.to_dict()}")
    expected_vwap = (0.5 * 70_000 + 1.0 * 70_010 + 1.0 * 70_050) / 2.5
    assert expected_vwap == 70_024.0
    assert est.vwap == pytest.approx(70_024.0, abs=1e-9)
    assert est.levels_consumed == 3
    assert est.extrapolated_qty == 0.0 and est.visible_qty_filled == pytest.approx(2.5)
    assert est.worst_price == 70_050.0
    assert est.notional_usd == pytest.approx(175_060.0)
    mid = (69_990.0 + 70_000.0) / 2
    assert est.mid_price == mid
    assert est.slippage_vs_touch_bps == pytest.approx(24.0 / 70_000.0 * 1e4)   # ≈ 3.4286 bps
    assert est.slippage_vs_mid_bps == pytest.approx((70_024.0 - mid) / mid * 1e4)  # ≈ 4.1437 bps
    assert est.slippage_vs_touch_bps > 0 and est.slippage_vs_mid_bps > est.slippage_vs_touch_bps

    # Exactly-at-touch fill → zero touch slippage.
    small = MicrostructureSignalEngine.estimate_naive_sweep_price(bids, asks, "BUY", 0.25)
    assert small.vwap == 70_000.0 and small.slippage_vs_touch_bps == 0.0

    # SELL side walks bids; slippage still positive (= cost).
    sell = MicrostructureSignalEngine.estimate_naive_sweep_price(bids, asks, "sell", 2.5)
    assert sell.vwap == pytest.approx((69_990 + 69_980 + 0.5 * 69_970) / 2.5)
    assert sell.slippage_vs_touch_bps > 0

    # Beyond visible depth → residual extrapolated at L20 price + penalty.
    visible = sum(q for _, q in asks)  # 20.5 BTC
    big = MicrostructureSignalEngine.estimate_naive_sweep_price(
        bids, asks, "BUY", visible + 4.5, beyond_depth_penalty_bps=25.0
    )
    l20 = asks[-1][0]
    exp_notional = sum(p * q for p, q in asks) + 4.5 * l20 * (1 + 25.0 / 1e4)
    print(f"[BUY {visible + 4.5}] vwap={big.vwap:.4f} worst={big.worst_price:.4f} "
          f"extrap={big.extrapolated_qty} slip_mid={big.slippage_vs_mid_bps:.3f}bps")
    assert big.extrapolated_qty == pytest.approx(4.5)
    assert big.levels_consumed == 20
    assert big.worst_price == pytest.approx(l20 * 1.0025)
    assert big.vwap == pytest.approx(exp_notional / (visible + 4.5))

    with pytest.raises(ValueError):
        MicrostructureSignalEngine.estimate_naive_sweep_price(bids, asks, "BUY", 0)
    with pytest.raises(ValueError):
        MicrostructureSignalEngine.estimate_naive_sweep_price(bids, asks, "HOLD", 1)


# --------------------------------------------------------------------------- #
# Test 3 — LIVE BTC/USDT WebSocket → signal engine → Redis
# --------------------------------------------------------------------------- #
@pytest.mark.live
async def test_3_live_websocket_stream_to_redis(bus: RedisStateBus):
    _banner(f"TEST 3: LIVE BTC/USDT L2 Stream → Signal Engine → Redis ({LIVE_TICKS} ticks)")
    engine = MicrostructureSignalEngine()
    stream = L2OrderBookStream(symbol="BTC/USDT", engine=engine, bus=bus, max_reconnect_attempts=6)

    ticks = []
    async with asyncio.timeout(90):
        async for t in stream.stream_ticks(max_ticks=LIVE_TICKS):
            ticks.append(t)

    print(f"endpoint: {ticks[-1].endpoint} | connects={stream.stats['connects']} "
          f"failures={stream.stats['failures_by_endpoint']}")
    print(f"{'#':>3} {'exch_ts_ms':>14} {'mid':>12} {'bid':>11} {'ask':>11} {'sprd_bps':>9} "
          f"{'rawOBI':>8} {'wOBI':>8} {'Z':>8} {'vol_bps':>8} {'signal':>11} "
          f"{'sig_ms':>7} {'redis_ms':>8} {'tick_ms':>8}")
    for t in ticks:
        m = t.metrics
        z = f"{m.obi_zscore:>8.3f}" if m.warmed_up else f"{'warmup':>8}"
        print(f"{t.seq:>3} {t.book['timestamp_ms']:>14} {m.mid_price:>12.2f} {m.best_bid:>11.2f} "
              f"{m.best_ask:>11.2f} {m.spread_bps:>9.4f} {m.raw_obi:>8.4f} {m.weighted_obi:>8.4f} "
              f"{z} {m.realized_vol_bps:>8.4f} {m.signal.value:>11} "
              f"{t.signal_latency_ms:>7.4f} {t.redis_latency_ms:>8.4f} {t.tick_latency_ms:>8.4f}")

    lat = sorted(t.tick_latency_ms for t in ticks)
    print(f"\n>>> TICK LATENCY (frame→signal→Redis ack): min {lat[0]:.4f} | "
          f"p50 {statistics.median(lat):.4f} | mean {statistics.fmean(lat):.4f} | "
          f"max {lat[-1]:.4f} ms  (SLA < {TICK_LATENCY_SLA_MS} ms)")

    assert len(ticks) == LIVE_TICKS
    for t in ticks:
        m = t.metrics
        assert t.book["symbol"] == "BTC/USDT"
        assert 1 <= len(t.book["bids"]) <= 20 and 1 <= len(t.book["asks"]) <= 20
        assert all(a[0] > b[0] for a, b in zip(t.book["bids"], t.book["bids"][1:]))  # sorted desc
        assert all(a[0] < b[0] for a, b in zip(t.book["asks"], t.book["asks"][1:]))  # sorted asc
        assert 0 < m.best_bid < m.best_ask, f"invalid touch at tick {t.seq}"
        assert -1.0 <= m.weighted_obi <= 1.0 and -1.0 <= m.raw_obi <= 1.0
        assert math.isfinite(m.obi_zscore) and math.isfinite(m.realized_vol_bps)
        assert m.signal in (Signal.LONG_ENTRY, Signal.EXIT_LONG, Signal.NEUTRAL)
        assert t.redis_latency_ms > 0.0
        assert t.tick_latency_ms < TICK_LATENCY_SLA_MS, f"tick {t.seq}: {t.tick_latency_ms:.3f} ms"
    assert ticks[-1].metrics.warmed_up  # 20 ticks > 15 warm-up

    # Redis read-back must equal the last processed tick.
    rb = await bus.get_orderbook_snapshot("BTC/USDT")
    snap, last = rb.data, ticks[-1].metrics
    print(f"[redis readback {rb.latency_ms:.4f} ms] {snap.to_dict()}")
    assert snap is not None
    assert snap.best_bid == last.best_bid and snap.best_ask == last.best_ask
    assert snap.obi == pytest.approx(last.weighted_obi)
    assert snap.obi_zscore == pytest.approx(last.obi_zscore)
    assert snap.mid_price == pytest.approx(last.mid_price)

    # Sweep estimate on the live book (informational).
    bk = ticks[-1].book
    sw = MicrostructureSignalEngine.estimate_naive_sweep_price(bk["bids"], bk["asks"], "BUY", 5.0)
    print(f"[live sweep BUY 5 BTC] vwap={sw.vwap:.2f} levels={sw.levels_consumed} "
          f"extrap={sw.extrapolated_qty:.4f} slip_vs_mid={sw.slippage_vs_mid_bps:.3f} bps "
          f"slip_vs_touch={sw.slippage_vs_touch_bps:.3f} bps")
    assert sw.vwap >= bk["asks"][0][0]
