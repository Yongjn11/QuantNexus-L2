"""
QuantNexus-L2 :: Real-Time Institutional Trading & Risk Dashboard
=================================================================

Auto-refreshing, real-time institutional operations cockpit built with Streamlit
and Plotly.

Features:
  1. Auto-Streaming Daemon: Launches ``LiveQuantNexusOrchestrator`` in the background
     on startup, continuously streaming live BTC/USDT L2 ticks from Binance Vision.
  2. Sub-Second Auto-Refresh: Uses ``@st.fragment(run_every=1.0)`` to re-render
     live prices, Top-20 depth walls, OBI Z-scores, and equity hands-free.
  3. A/B Alpha Monitoring: Real-time equity curve benchmarking QuantNexus vs.
     Baseline Retail Bot.
  4. Real-Time Order Book Depth & Microstructure Gauge: Visualizes bid/ask depth walls.
  5. Interactive Chaos Controls: Live Slow-Loop LangGraph override allowing
     immediate injection of macro or Black-Swan shocks into the running engine.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path when launched via Streamlit
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

from run_live_engine import (
    LiveQuantNexusOrchestrator,
    get_orchestrator,
    is_live_engine_running,
    start_background_live_engine,
    stop_background_live_engine,
)
from slow_loop.risk_agent import AgenticRiskGatekeeper
from storage.pg_logger import PostgresLedger
from storage.redis_bus import RedisStateBus

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="QuantNexus-L2 | Real-Time Execution Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS
st.markdown(
    """
    <style>
    /* Dark Theme Container Accents */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #131924 0%, #171f2e 100%);
        border: 1px solid #232d3f;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.78rem;
        margin-top: 4px;
    }
    .badge-green {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.4);
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-yellow {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(255, 214, 0, 0.15);
        color: #ffd600;
        border: 1px solid rgba(255, 214, 0, 0.4);
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-red {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.4);
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pulsing-dot {
        height: 10px;
        width: 10px;
        background-color: #00e676;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #00e676;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Resource Caching
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_ledger() -> PostgresLedger:
    lg = PostgresLedger()
    lg.init_db()
    return lg


@st.cache_resource
def get_bus() -> RedisStateBus:
    return RedisStateBus()


@st.cache_resource
def get_gatekeeper(_bus: RedisStateBus, _ledger: PostgresLedger) -> AgenticRiskGatekeeper:
    return AgenticRiskGatekeeper(bus=_bus, ledger=_ledger)


# --------------------------------------------------------------------------- #
# Auto-Start Live Background Engine (Hands-Free Operation)
# --------------------------------------------------------------------------- #
# Automatically ensure the live Binance streaming daemon is active on launch
if "auto_started" not in st.session_state:
    start_background_live_engine(symbol="BTC/USDT", entry_z=1.5)
    st.session_state["auto_started"] = True


# --------------------------------------------------------------------------- #
# Sidebar Controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚡ Engine Control")
    live_running = is_live_engine_running()

    stream_toggle = st.toggle(
        "🟢 Live BTC Stream & Auto-Trade",
        value=live_running,
        help="Continuously stream live BTC/USDT L2 ticks from Binance Vision and paper trade automatically.",
    )

    if stream_toggle and not live_running:
        start_background_live_engine(symbol="BTC/USDT", entry_z=1.5)
        st.rerun()
    elif not stream_toggle and live_running:
        stop_background_live_engine()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧪 Slow-Loop Chaos Injector")
    st.caption("Inject macro shocks or Black-Swan events into the running system:")

    bus = get_bus()
    ledger = get_ledger()
    gatekeeper = get_gatekeeper(bus, ledger)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("🟢 Normal (1.0x)", help="Reset to benign structural inflows"):
            asyncio.run(
                gatekeeper.evaluate_headline(
                    "Institutional spot Bitcoin ETF records $450M net daily inflow amid steady market liquidity",
                    source="DASHBOARD_PRESET",
                )
            )
            st.toast("✅ Injected Benign Regime (1.0x)", icon="🟢")
    with col_b2:
        if st.button("🟡 Macro (0.6x)", help="Trigger macro rate cut delay caution"):
            asyncio.run(
                gatekeeper.evaluate_headline(
                    "US CPI inflation prints hotter than expected; Federal Reserve signals delay in rate cuts",
                    source="DASHBOARD_PRESET",
                )
            )
            st.toast("⚠️ Injected Macro Caution (0.6x)", icon="🟡")
    with col_b3:
        if st.button("🔴 Freeze (0.0x)", help="Emergency exchange withdrawal freeze"):
            asyncio.run(
                gatekeeper.evaluate_headline(
                    "BREAKING: Major crypto exchange halts all client withdrawals amid liquidity freeze and regulatory probe",
                    source="DASHBOARD_PRESET",
                )
            )
            st.toast("🚨 Injected Black-Swan Circuit Breaker!", icon="🔴")

    custom_headline = st.text_input(
        "Custom Financial Headline:",
        placeholder="e.g. SEC announces emergency stablecoin probe...",
    )
    if st.button("🤖 Evaluate with Gemini 3.8 Flash", use_container_width=True):
        if custom_headline.strip():
            with st.spinner("LangGraph Agent evaluating headline with Gemini..."):
                res = asyncio.run(
                    gatekeeper.evaluate_headline(
                        headline=custom_headline.strip(), source="DASHBOARD_USER"
                    )
                )
                st.success(
                    f"Assessed: {res['final_category']} (Risk: {res['final_risk_score']:.2f}, Mult: {res['final_risk_multiplier']:.2f}x)"
                )
                st.caption(f"Reasoning: {res['final_reasoning']}")

    st.markdown("---")
    st.markdown("### ⚡ Fast-Forward Demo")
    if st.button("🎯 Force Instant Demo Trade", use_container_width=True, help="Force an immediate round-trip paper trade cycle."):
        orch = get_orchestrator()
        with st.spinner("Executing TWAP entry & take-profit exit..."):
            trade_rec = asyncio.run(orch.execute_demo_trade())
            if trade_rec:
                st.toast(
                    f"🎉 Demo Trade Closed: +${trade_rec.realized_pnl_usd:,.2f} (Saved ${trade_rec.slippage_saved_usd:.2f})",
                    icon="💰",
                )


# --------------------------------------------------------------------------- #
# Auto-Refreshing Dashboard Fragment (Updates every 1.0s Hands-Free)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=1.0)
def render_live_dashboard():
    bus = get_bus()
    ledger = get_ledger()
    orch = get_orchestrator()

    # 1. Fetch live Redis State (using sync methods for sub-ms UI render)
    try:
        risk_res = bus.get_risk_state_sync()
        risk_state = risk_res.data
        ob_res = bus.get_orderbook_snapshot_sync("BTC/USDT")
        ob_snap = ob_res.data
    except Exception as exc:
        st.warning(f"Connecting to Redis State Bus... ({exc})")
        return

    # 2. Fetch Postgres Portfolio Summary & Curves
    try:
        summary = ledger.get_portfolio_summary()
        trades = ledger.get_recent_trades(limit=15)
        audits = ledger.get_recent_risk_audits(limit=15)
        equity_curve = ledger.get_equity_curve(limit=200)
    except Exception as exc:
        st.error(f"PostgreSQL connection error: {exc}")
        return

    orch_summary = orch.get_summary()

    # --- Live Header Banner ---
    running = is_live_engine_running()
    status_badge = (
        '<span class="badge-green"><span class="pulsing-dot"></span>LIVE WEBSOCKET STREAMING</span>'
        if running
        else '<span class="badge-red">STREAMING PAUSED</span>'
    )
    pos_badge = (
        f'<span class="badge-green">LONG IN-POSITION ({orch_summary["position_qty"]:.2f} BTC @ ${orch_summary["position_entry"]:,.1f})</span>'
        if orch_summary["has_position"]
        else '<span class="badge-yellow">POSITION: FLAT</span>'
    )
    cb_badge = (
        '<span class="badge-red">CIRCUIT BREAKER: TRIGGERED</span>'
        if risk_state.circuit_breaker
        else '<span class="badge-green">CIRCUIT BREAKER: ARMED</span>'
    )

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <h2 style="margin:0; font-size:1.8rem; font-weight:800; letter-spacing:-0.03em;">
                    ⚡ QuantNexus-L2 <span style="font-weight:400; color:#8b949e; font-size:1.1rem;">| Real-Time Order Book Execution Engine</span>
                </h2>
                <div style="margin-top:6px; display:flex; gap:10px; align-items:center;">
                    {status_badge}
                    {pos_badge}
                    {cb_badge}
                    <span style="font-size:0.8rem; color:#8b949e;">Ticks Processed: <b style="color:#fff;">{orch_summary['tick_count']:,}</b></span>
                    <span style="font-size:0.8rem; color:#8b949e;">Latency: <b style="color:#00e676;">{orch_summary['tick_latency_ms']:.2f} ms</b></span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; color:#8b949e;">BTC/USDT MID PRICE</div>
                <div style="font-size:2.0rem; font-weight:800; color:#ffffff; font-variant-numeric: tabular-nums;">
                    ${(ob_snap.mid_price if ob_snap else 0.0):,.2f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI Metric Cards (Row 1 & Row 2) ---
    k1, k2, k3, k4, k5 = st.columns(5)

    qn_eq = summary.get("latest_equity", {}).get("quantnexus_equity_usd", 100_000.0) if summary.get("latest_equity") else 100_000.0
    base_eq = summary.get("latest_equity", {}).get("baseline_equity_usd", 100_000.0) if summary.get("latest_equity") else 100_000.0
    alpha_usd = qn_eq - base_eq
    alpha_pct = (alpha_usd / 100_000.0) * 100.0
    net_pnl = qn_eq - 100_000.0
    net_pnl_pct = (net_pnl / 100_000.0) * 100.0
    pnl_sign = "+" if net_pnl >= 0 else ""
    pnl_color = "#00e676" if net_pnl >= 0 else "#ff1744"

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">QuantNexus Equity</div>
                <div class="metric-value">${qn_eq:,.2f}</div>
                <div class="metric-sub" style="color:{pnl_color};">Net PnL: {pnl_sign}${net_pnl:,.2f} ({pnl_sign}{net_pnl_pct:.2f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        alpha_color = "#00e676" if alpha_usd >= 0 else "#ff1744"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Net Alpha vs. Baseline</div>
                <div class="metric-value" style="color:{alpha_color};">+${alpha_usd:,.2f}</div>
                <div class="metric-sub" style="color:{alpha_color};">Outperformance: +{alpha_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        wr = summary.get("win_rate_pct", 0.0)
        n_trades = summary.get("total_trades", summary.get("completed_trades", 0) + summary.get("aborted_by_ai", 0))
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value">{wr:.1f}%</div>
                <div class="metric-sub" style="color:#8b949e;">{summary.get('wins', 0)} Wins / {n_trades} Closed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        pf = summary.get("profit_factor")
        pf_str = f"{pf:.2f}" if pf is not None else "N/A"
        gp = summary.get('gross_profit_usd', 0.0) or 0.0
        gl = summary.get('gross_loss_usd', 0.0) or 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Profit Factor</div>
                <div class="metric-value">{pf_str}</div>
                <div class="metric-sub" style="color:#8b949e;">Gross: +${gp:,.1f} vs -${gl:,.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        max_dd = summary.get("latest_equity", {}).get("max_drawdown_pct", 0.0) if summary.get("latest_equity") else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value">{max_dd:.2f}%</div>
                <div class="metric-sub" style="color:#00e676;">Asymmetric Capital Shield</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Row 2 (Engineering & State Bus)
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        tot_slip = summary.get("total_slippage_saved_usd", 0.0)
        avg_bps = summary.get("avg_slippage_saved_bps", 0.0) or 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Slippage Saved</div>
                <div class="metric-value" style="color:#00e676;">${tot_slip:,.2f}</div>
                <div class="metric-sub" style="color:#8b949e;">Avg Saved: <b>{avg_bps:.2f} bps</b> / fill</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        mult = risk_state.risk_multiplier
        mult_color = "#00e676" if mult >= 0.8 else ("#ffd600" if mult >= 0.2 else "#ff1744")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">AI Risk Multiplier</div>
                <div class="metric-value" style="color:{mult_color};">{mult:.2f}x</div>
                <div class="metric-sub" style="color:#8b949e;">Redis State Bus (Sub-ms)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r3:
        w_obi = ob_snap.obi if ob_snap else 0.0
        obi_color = "#00e676" if w_obi > 0 else "#ff1744"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Weighted OBI (Depth-Decay)</div>
                <div class="metric-value" style="color:{obi_color};">{w_obi:+.4f}</div>
                <div class="metric-sub" style="color:#8b949e;">Top-20 Liquidity Imbalance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r4:
        z = ob_snap.obi_zscore if ob_snap else 0.0
        z_color = "#00e676" if z >= 1.5 else ("#ff1744" if z <= -1.0 else "#8b949e")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Rolling Z-Score (100 ticks)</div>
                <div class="metric-value" style="color:{z_color};">{z:+.2f}σ</div>
                <div class="metric-sub" style="color:#8b949e;">Entry Threshold: &ge; +1.50σ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r5:
        spread = ob_snap.spread_bps if ob_snap else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">L2 Touch Spread</div>
                <div class="metric-value">{spread:.2f} bps</div>
                <div class="metric-sub" style="color:#8b949e;">Bid: ${(ob_snap.best_bid if ob_snap else 0.0):,.1f} | Ask: ${(ob_snap.best_ask if ob_snap else 0.0):,.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # --- Main Charts Area (2 Columns) ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📊 Live Top-20 L2 Order Book Depth Wall")
        # Build Depth Wall Chart from latest tick
        if orch.last_tick and "bids" in orch.last_tick.book and "asks" in orch.last_tick.book:
            bids = orch.last_tick.book["bids"][:10]
            asks = orch.last_tick.book["asks"][:10]

            fig_depth = make_subplots(
                rows=1, cols=2, shared_yaxes=False,
                subplot_titles=("Bid Depth Wall (Buyers)", "Ask Depth Wall (Sellers)"),
                horizontal_spacing=0.08,
            )

            # Bids: descending from top
            b_prices = [f"${p:,.1f}" for p, _ in reversed(bids)]
            b_qtys = [q for _, q in reversed(bids)]

            # Asks: ascending from top
            a_prices = [f"${p:,.1f}" for p, _ in asks]
            a_qtys = [q for _, q in asks]

            fig_depth.add_trace(
                go.Bar(
                    y=b_prices, x=b_qtys, orientation="h",
                    marker=dict(color="#00e676", line=dict(color="#00b0ff", width=1)),
                    name="Bids (BTC)",
                ),
                row=1, col=1,
            )
            fig_depth.add_trace(
                go.Bar(
                    y=a_prices, x=a_qtys, orientation="h",
                    marker=dict(color="#ff1744", line=dict(color="#ff5252", width=1)),
                    name="Asks (BTC)",
                ),
                row=1, col=2,
            )

            fig_depth.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0b0e14",
                plot_bgcolor="#111622",
                font=dict(color="#8b949e", size=10),
                showlegend=False,
            )
            fig_depth.update_xaxes(title_text="Quantity (BTC)", gridcolor="#1f293d")
            fig_depth.update_yaxes(gridcolor="#1f293d")
            st.plotly_chart(fig_depth, use_container_width=True)
        else:
            st.info("Waiting for first live L2 tick snapshot from Binance Vision...")

    with col_right:
        st.markdown("#### 📈 Dual-Account A/B Equity Curve (QuantNexus vs. Baseline)")
        if equity_curve:
            df_eq = pd.DataFrame(equity_curve)
            df_eq["trade_num"] = range(1, len(df_eq) + 1)

            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=df_eq["trade_num"],
                    y=df_eq["quantnexus_equity_usd"],
                    mode="lines+markers",
                    name="QuantNexus Engine (TWAP + AI Gated)",
                    line=dict(color="#00e676", width=2.5),
                    marker=dict(size=6, color="#00e676"),
                )
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=df_eq["trade_num"],
                    y=df_eq["baseline_equity_usd"],
                    mode="lines+markers",
                    name="Baseline Retail Bot (Naive Sweep)",
                    line=dict(color="#ff1744", width=2.0, dash="dash"),
                    marker=dict(size=5, color="#ff1744"),
                )
            )
            fig_eq.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0b0e14",
                plot_bgcolor="#111622",
                font=dict(color="#8b949e", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Closed Trade Sequence", gridcolor="#1f293d"),
                yaxis=dict(title="Portfolio Equity (USD)", gridcolor="#1f293d"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No closed trades recorded yet. Engine is running in paper mode...")

    # --- Lower Tabs: Ledger & AI Risk Audit Log ---
    st.markdown("---")
    tab_ledger, tab_audit = st.tabs([
        "📋 PostgreSQL Trade Execution Ledger (`trade_executions`)",
        "🛡️ LangGraph AI Risk Audit Trail (`risk_audit_logs`)",
    ])

    with tab_ledger:
        if trades:
            df_trades = pd.DataFrame(trades)
            # Format display
            cols = [
                "id", "timestamp", "symbol", "side", "quantity",
                "entry_price", "exit_price", "twap_exec_price",
                "slippage_saved_usd", "realized_pnl_usd", "status",
            ]
            display_cols = [c for c in cols if c in df_trades.columns]
            st.dataframe(
                df_trades[display_cols].style.format({
                    "entry_price": "${:,.2f}",
                    "exit_price": "${:,.2f}",
                    "twap_exec_price": "${:,.2f}",
                    "slippage_saved_usd": "${:,.2f}",
                    "realized_pnl_usd": "${:,.2f}",
                    "quantity": "{:.3f} BTC",
                }),
                use_container_width=True,
                height=220,
            )
        else:
            st.info("Trade ledger is clean. Trades appear automatically on signal execution.")

    with tab_audit:
        if audits:
            df_audits = pd.DataFrame(audits)
            cols = ["id", "timestamp", "risk_score", "new_risk_multiplier", "circuit_breaker_triggered", "headline", "reasoning"]
            display_cols = [c for c in cols if c in df_audits.columns]
            st.dataframe(
                df_audits[display_cols],
                use_container_width=True,
                height=220,
            )
        else:
            st.info("No risk audit records yet. Inject headlines in sidebar to observe LangGraph.")


# Run the live fragment
render_live_dashboard()
