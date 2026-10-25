"""
QuantNexus-L2 :: Microstructure Alpha Signal Engine
===================================================

Deterministic, allocation-light signal math for Top-N L2 order-book snapshots.

Inputs (per tick)
-----------------
``bids`` / ``asks``: sequences of ``[price, qty]`` sorted best -> worst.

Outputs
-------
* Top-of-book: ``best_bid``, ``best_ask``, ``mid_price``, ``spread_bps``
* ``raw_obi``       – (ΣQ_bid − ΣQ_ask) / (ΣQ_bid + ΣQ_ask) over the top N levels
* ``weighted_obi``  – same, with exponential depth weights w_i = exp(−λ·i)
* ``obi_zscore``    – Z = (OBI_t − μ) / (σ + ε) over a rolling window of *prior* ticks
* ``realized_vol_bps`` – rolling stdev of per-tick mid log-returns, in bps
* ``signal``        – LONG_ENTRY / EXIT_LONG / NEUTRAL

Design notes
------------
* The Z-score is computed against the rolling window of *previous* ticks
  (the current OBI is appended afterwards). Including the current observation
  in its own baseline would dampen exactly the shocks we are trying to detect
  and introduce a subtle look-ahead bias.
* Dispersion floor: Z uses ``max(σ, min_sigma)``. In quiet, static books σ of
  the OBI can collapse to ~1e-4, at which point a single 0.0001 lot change
  produces |Z| > 1 – pure noise masquerading as signal. The floor requires an
  economically meaningful OBI move (≥ entry_z·min_sigma) before firing.
  Set ``min_sigma=0`` to recover the textbook (OBI−μ)/(σ+ε).
* Rolling mean / variance are maintained with O(1) running sums (with periodic
  exact re-computation to cancel floating-point drift), so per-tick cost is
  independent of the window length.
* Pure-Python math on ≤20 levels is faster than NumPy here (no array
  construction overhead); typical ``update()`` cost is a few microseconds.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Sequence

Level = Sequence[float]  # [price, qty]
Book = Sequence[Level]


class Signal(str, Enum):
    """String enum so ``Signal.LONG_ENTRY == 'LONG_ENTRY'`` holds."""

    LONG_ENTRY = "LONG_ENTRY"
    EXIT_LONG = "EXIT_LONG"
    NEUTRAL = "NEUTRAL"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    timestamp_ms: Optional[int]
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    raw_obi: float
    weighted_obi: float
    obi_zscore: float
    realized_vol_bps: float
    signal: Signal
    warmed_up: bool
    tick_index: int
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    trend_ok: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        return d


@dataclass(frozen=True, slots=True)
class SweepEstimate:
    """Result of walking the book with a single aggressive market order."""

    side: str                      # "BUY" (lifts asks) | "SELL" (hits bids)
    quantity: float
    vwap: float
    best_touch: float
    mid_price: float
    worst_price: float
    notional_usd: float
    slippage_vs_mid_bps: float     # positive = cost
    slippage_vs_touch_bps: float   # positive = cost
    levels_consumed: int
    visible_qty_filled: float
    extrapolated_qty: float        # qty beyond visible depth (penalised)

    def to_dict(self) -> dict:
        return asdict(self)


class _RollingStats:
    """O(1) rolling mean / variance over a fixed-length window."""

    __slots__ = ("buf", "_sum", "_sumsq", "_pushes", "_resync_every")

    def __init__(self, maxlen: int) -> None:
        self.buf: deque[float] = deque(maxlen=maxlen)
        self._sum = 0.0
        self._sumsq = 0.0
        self._pushes = 0
        self._resync_every = maxlen * 10

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, x: float) -> None:
        buf = self.buf
        if len(buf) == buf.maxlen:
            old = buf[0]
            self._sum -= old
            self._sumsq -= old * old
        buf.append(x)
        self._sum += x
        self._sumsq += x * x
        self._pushes += 1
        if self._pushes % self._resync_every == 0:  # cancel FP drift
            self._sum = math.fsum(buf)
            self._sumsq = math.fsum(v * v for v in buf)

    def mean(self) -> float:
        n = len(self.buf)
        return self._sum / n if n else 0.0

    def std(self, ddof: int = 0) -> float:
        n = len(self.buf)
        if n - ddof <= 0:
            return 0.0
        mu = self._sum / n
        var = (self._sumsq - n * mu * mu) / (n - ddof)
        return math.sqrt(var) if var > 0.0 else 0.0

    def clear(self) -> None:
        self.buf.clear()
        self._sum = self._sumsq = 0.0
        self._pushes = 0


class MicrostructureSignalEngine:
    """Stateful per-symbol microstructure signal engine."""

    def __init__(
        self,
        depth: int = 20,
        decay_lambda: float = 0.15,
        window: int = 100,
        warmup_ticks: int = 15,
        entry_z: float = 1.8,
        exit_z: float = -1.0,
        max_spread_bps: float = 5.0,
        epsilon: float = 1e-6,
        min_sigma: float = 0.05,
        ema_fast_ticks: int = 15,
        ema_slow_ticks: int = 50,
        trend_tolerance_bps: float = 0.5,
        trend_filter: bool = True,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not 2 <= warmup_ticks <= window:
            raise ValueError("require 2 <= warmup_ticks <= window")
        if exit_z >= entry_z:
            raise ValueError("exit_z must be < entry_z")
        if min_sigma < 0:
            raise ValueError("min_sigma must be >= 0")

        self.depth = depth
        self.decay_lambda = decay_lambda
        self.window = window
        self.warmup_ticks = warmup_ticks
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_spread_bps = max_spread_bps
        self.epsilon = epsilon
        self.min_sigma = min_sigma

        # Short-term trend confirmation (anti "falling knife" entry filter)
        if not 1 <= ema_fast_ticks < ema_slow_ticks:
            raise ValueError("require 1 <= ema_fast_ticks < ema_slow_ticks")
        self.trend_filter = trend_filter
        self.trend_tolerance_bps = trend_tolerance_bps
        self._alpha_fast = 2.0 / (ema_fast_ticks + 1)
        self._alpha_slow = 2.0 / (ema_slow_ticks + 1)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None

        self.weights: tuple[float, ...] = tuple(math.exp(-decay_lambda * i) for i in range(depth))
        self._obi = _RollingStats(window)
        self._ret = _RollingStats(window)
        self._last_mid: Optional[float] = None
        self.tick_count = 0
        self.last_metrics: Optional[SignalMetrics] = None

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._obi.clear()
        self._ret.clear()
        self._last_mid = None
        self._ema_fast = None
        self._ema_slow = None
        self.tick_count = 0
        self.last_metrics = None

    @property
    def warmed_up(self) -> bool:
        return len(self._obi) >= self.warmup_ticks

    # ------------------------------------------------------------------- math
    @staticmethod
    def _imbalance(b: float, a: float) -> float:
        tot = b + a
        return (b - a) / tot if tot > 0.0 else 0.0

    def compute_obi(self, bids: Book, asks: Book) -> tuple[float, float]:
        """Return ``(raw_obi, weighted_obi)`` over the top ``depth`` levels."""
        d = self.depth
        w = self.weights
        raw_b = raw_a = wb = wa = 0.0
        for i, lvl in enumerate(bids[:d]):
            q = float(lvl[1])
            raw_b += q
            wb += w[i] * q
        for i, lvl in enumerate(asks[:d]):
            q = float(lvl[1])
            raw_a += q
            wa += w[i] * q
        return self._imbalance(raw_b, raw_a), self._imbalance(wb, wa)

    def _update_trend(self, mid: float) -> tuple[float, float, bool]:
        """Update fast/slow mid EMAs; return (ema_fast, ema_slow, trend_ok).

        trend_ok == momentum is non-falling: both the fast EMA sits at or above
        the slow EMA, and the current mid price sits no more than
        ``trend_tolerance_bps`` (0.5 bps) below the slow EMA.
        """
        if self._ema_fast is None:
            self._ema_fast = self._ema_slow = mid
        else:
            self._ema_fast += self._alpha_fast * (mid - self._ema_fast)
            self._ema_slow += self._alpha_slow * (mid - self._ema_slow)
        floor = self._ema_slow * (1.0 - self.trend_tolerance_bps / 1e4)
        ok = (not self.trend_filter) or (self._ema_fast >= self._ema_slow and mid >= floor)
        return self._ema_fast, self._ema_slow, ok

    def evaluate_signal(
        self,
        obi_zscore: float,
        spread_bps: float,
        warmed_up: bool = True,
        trend_ok: bool = True,
    ) -> Signal:
        """Map Z-score + spread + trend to a discrete trading signal.

        The spread and trend filters gate *entries only*: exits must never be
        blocked by a wide book or a falling market (that is precisely when you
        want out).
        """
        if not warmed_up:
            return Signal.NEUTRAL
        if obi_zscore >= self.entry_z and spread_bps <= self.max_spread_bps and trend_ok:
            return Signal.LONG_ENTRY
        if obi_zscore <= self.exit_z:
            return Signal.EXIT_LONG
        return Signal.NEUTRAL

    def update(self, bids: Book, asks: Book, timestamp_ms: Optional[int] = None) -> SignalMetrics:
        """Ingest one L2 snapshot and return the full metric set + signal."""
        if not bids or not asks:
            raise ValueError("empty book side")
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0.0 or best_ask <= 0.0:
            raise ValueError("non-positive top-of-book price")
        if best_ask < best_bid:
            raise ValueError(f"crossed book: bid={best_bid} > ask={best_ask}")

        mid = 0.5 * (best_bid + best_ask)
        spread_bps = (best_ask - best_bid) / mid * 1e4
        raw_obi, w_obi = self.compute_obi(bids, asks)

        # Z-score vs. PRIOR window, then append current observation.
        warmed = len(self._obi) >= self.warmup_ticks
        if warmed:
            sigma = max(self._obi.std(), self.min_sigma)
            z = (w_obi - self._obi.mean()) / (sigma + self.epsilon)
        else:
            z = 0.0
        self._obi.push(w_obi)

        # Micro-volatility of mid log-returns (bps per tick).
        if self._last_mid is not None:
            self._ret.push(math.log(mid / self._last_mid) * 1e4)
        self._last_mid = mid
        vol = self._ret.std(ddof=1) if len(self._ret) >= 2 else 0.0
        ema_f, ema_s, trend_ok = self._update_trend(mid)

        self.tick_count += 1
        m = SignalMetrics(
            timestamp_ms=timestamp_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread_bps=spread_bps,
            raw_obi=raw_obi,
            weighted_obi=w_obi,
            obi_zscore=z,
            realized_vol_bps=vol,
            signal=self.evaluate_signal(z, spread_bps, warmed, trend_ok),
            warmed_up=warmed,
            tick_index=self.tick_count,
            ema_fast=ema_f,
            ema_slow=ema_s,
            trend_ok=trend_ok,
        )
        self.last_metrics = m
        return m

    # ------------------------------------------------------- execution costs
    @staticmethod
    def estimate_naive_sweep_price(
        bids: Book,
        asks: Book,
        side: str,
        quantity: float,
        beyond_depth_penalty_bps: float = 25.0,
    ) -> SweepEstimate:
        """Walk the book as a single aggressive market order of ``quantity``.

        BUY lifts asks, SELL hits bids. If ``quantity`` exceeds visible depth,
        the residual is assumed to fill at the last visible level price worsened
        by ``beyond_depth_penalty_bps`` (conservative hidden-liquidity proxy).
        Slippage figures are signed so that *positive = cost to the trader*.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not bids or not asks:
            raise ValueError("empty book side")
        s = side.upper()
        if s not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")

        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        mid = 0.5 * (best_bid + best_ask)
        book, touch, sign = (asks, best_ask, 1.0) if s == "BUY" else (bids, best_bid, -1.0)

        remaining = float(quantity)
        notional = 0.0
        levels = 0
        last_px = touch
        for lvl in book:
            if remaining <= 0.0:
                break
            px, q = float(lvl[0]), float(lvl[1])
            if q <= 0.0:
                continue
            take = q if q < remaining else remaining
            notional += take * px
            remaining -= take
            levels += 1
            last_px = px

        visible_filled = float(quantity) - remaining
        extrapolated = 0.0
        worst = last_px
        if remaining > 1e-12:
            worst = last_px * (1.0 + sign * beyond_depth_penalty_bps / 1e4)
            notional += remaining * worst
            extrapolated = remaining

        vwap = notional / float(quantity)
        return SweepEstimate(
            side=s,
            quantity=float(quantity),
            vwap=vwap,
            best_touch=touch,
            mid_price=mid,
            worst_price=worst,
            notional_usd=notional,
            slippage_vs_mid_bps=sign * (vwap - mid) / mid * 1e4,
            slippage_vs_touch_bps=sign * (vwap - touch) / touch * 1e4,
            levels_consumed=levels,
            visible_qty_filled=visible_filled,
            extrapolated_qty=extrapolated,
        )
