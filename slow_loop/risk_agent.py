"""
QuantNexus-L2 :: Agentic Risk Gatekeeper (LangGraph + Gemini 3.8 Flash)
========================================================================

Stateful, asynchronous risk gating engine implemented with LangGraph (StateGraph)
and the official Google GenAI SDK (``gemini-3.8-flash``).

Pipeline Architecture (StateGraph):
-----------------------------------
  1. ``enrich_market_context``:
     Reads the latest real-time L2 order book snapshot from ``RedisStateBus``
     (e.g., best_bid, best_ask, spread_bps, OBI, Z-score) to provide current
     microstructure context to the LLM.
  2. ``evaluate_tail_risk``:
     Evaluates the headline. Checks the in-memory cache first (0 tokens). If uncached,
     queries Gemini 3.8 Flash with structured JSON output enforcement (<150 input tokens,
     thinking_budget=0 for instant execution).
     Zero-Downtime Fallback: If external API quota/network issues occur or live LLM
     is disabled, seamlessly falls back to a deterministic financial lexicon classifier.
  3. ``enforce_gating_policy_and_audit``:
     Applies strict institutional risk guardrails:
       - risk_score >= 0.80  -> risk_multiplier = 0.0, circuit_breaker = True (TAIL_RISK_BLACK_SWAN)
       - 0.40 <= risk_score < 0.80 -> 0.20 <= risk_multiplier <= 0.60, circuit_breaker = False (MACRO_CAUTION)
       - risk_score < 0.40   -> risk_multiplier = 1.0, circuit_breaker = False (NORMAL)
     Updates ``RedisStateBus`` sub-millisecond state and logs audit records to
     PostgreSQL ``risk_audit_logs``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from storage.pg_logger import PostgresLedger
from storage.redis_bus import RedisStateBus

load_dotenv()
log = logging.getLogger("quantnexus.slow_loop.risk_agent")

# Try importing google.genai
try:
    from google import genai
    from google.genai import types

    HAVE_GOOGLE_GENAI = True
except ImportError:
    HAVE_GOOGLE_GENAI = False


# --------------------------------------------------------------------------- #
# Structured Risk Assessment Schema
# --------------------------------------------------------------------------- #
class RiskAssessment(BaseModel):
    """Institutional Tail-Risk Assessment Schema."""

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk score in [0.0, 1.0]: 0.0-0.39 benign, 0.40-0.79 macro caution, 0.80-1.0 tail risk/black swan",
    )
    recommended_multiplier: float = Field(
        ..., ge=0.0, le=1.0, description="Recommended position multiplier in [0.0, 1.0]"
    )
    trigger_circuit_breaker: bool = Field(
        ..., description="True if emergency liquidation / halt is required"
    )
    category: str = Field(
        ..., description="'NORMAL', 'MACRO_CAUTION', or 'TAIL_RISK_BLACK_SWAN'"
    )
    reasoning: str = Field(
        ..., description="Concise institutional risk justification under 30 words"
    )


class RiskGateState(TypedDict):
    """LangGraph workflow state."""

    headline: str
    source: str
    use_live_llm: bool
    market_context: dict[str, Any]
    raw_assessment: Optional[RiskAssessment]
    final_risk_score: float
    final_risk_multiplier: float
    final_circuit_breaker: bool
    final_category: str
    final_reasoning: str
    engine_used: str  # "GEMINI_LIVE" | "IN_MEMORY_CACHE" | "DETERMINISTIC_FALLBACK"
    tokens_used: int
    latency_ms: float
    audit_id: Optional[int]


# --------------------------------------------------------------------------- #
# Deterministic Financial Lexicon Classifier (Zero-Downtime Fallback)
# --------------------------------------------------------------------------- #
def classify_deterministic(headline: str, market_context: dict[str, Any]) -> RiskAssessment:
    """Institutional deterministic classifier guaranteeing 100% uptime without API dependencies."""
    text = headline.lower()

    # Black-Swan / Severe Tail Risk Keywords
    tail_risk_keywords = (
        "halt withdrawal",
        "halts withdrawal",
        "halting withdrawal",
        "freeze withdrawal",
        "liquidity freeze",
        "insolvency",
        "insolvent",
        "hack",
        "hacked",
        "exploit",
        "sec freeze",
        "regulatory probe",
        "de-peg",
        "bank run",
        "emergency halt",
        "chapter 11",
        "bankruptcy",
        "fraud",
        "subpoena",
        "investigation",
    )
    if any(kw in text for kw in tail_risk_keywords):
        return RiskAssessment(
            risk_score=0.95,
            recommended_multiplier=0.0,
            trigger_circuit_breaker=True,
            category="TAIL_RISK_BLACK_SWAN",
            reasoning="Severe tail-risk event: exchange insolvency / regulatory freeze requires immediate capital preservation.",
        )

    # Macro Caution / Volatility Expansion Keywords
    macro_caution_keywords = (
        "cpi",
        "inflation",
        "hotter than expected",
        "hawkish",
        "rate hike",
        "delay in rate cut",
        "delay in cuts",
        "tariff",
        "fed",
        "fomc",
        "stagflation",
        "treasury yield",
        "geopolitical",
        "recession",
    )
    if any(kw in text for kw in macro_caution_keywords):
        return RiskAssessment(
            risk_score=0.55,
            recommended_multiplier=0.40,
            trigger_circuit_breaker=False,
            category="MACRO_CAUTION",
            reasoning="Elevated macroeconomic volatility: tighter monetary outlook warrants systematic exposure de-risking.",
        )

    # Benign / Positive Structural Liquidity Keywords
    benign_keywords = (
        "inflow",
        "etf",
        "steady liquidity",
        "steady market",
        "adoption",
        "partnership",
        "upgrade",
        "growth",
        "record inflow",
    )
    if any(kw in text for kw in benign_keywords):
        return RiskAssessment(
            risk_score=0.10,
            recommended_multiplier=1.0,
            trigger_circuit_breaker=False,
            category="NORMAL",
            reasoning="Benign market conditions with favorable structural inflows and sound liquidity metrics.",
        )

    # Default Unclassified
    return RiskAssessment(
        risk_score=0.15,
        recommended_multiplier=1.0,
        trigger_circuit_breaker=False,
        category="NORMAL",
        reasoning="Routine market headline with neutral liquidity and volatility implications.",
    )


# --------------------------------------------------------------------------- #
# Agentic Risk Gatekeeper
# --------------------------------------------------------------------------- #
class AgenticRiskGatekeeper:
    """Stateful LangGraph workflow managing Slow-Loop risk gating and audits."""

    # Cost-saving in-memory cache shared across evaluations
    _headline_cache: dict[str, RiskAssessment] = {}

    def __init__(
        self,
        bus: RedisStateBus,
        ledger: PostgresLedger,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.bus = bus
        self.ledger = ledger
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        # Initialize Gemini Client if available
        self._gemini_client = None
        if HAVE_GOOGLE_GENAI and self.api_key:
            try:
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                log.warning("Could not initialize Google GenAI Client: %r", exc)

        # Build and compile LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the 3-node LangGraph workflow."""
        builder = StateGraph(RiskGateState)
        builder.add_node("enrich_market_context", self._node_enrich_context)
        builder.add_node("evaluate_tail_risk", self._node_evaluate_tail_risk)
        builder.add_node("enforce_gating_policy_and_audit", self._node_enforce_policy)

        builder.set_entry_point("enrich_market_context")
        builder.add_edge("enrich_market_context", "evaluate_tail_risk")
        builder.add_edge("evaluate_tail_risk", "enforce_gating_policy_and_audit")
        builder.add_edge("enforce_gating_policy_and_audit", END)

        return builder.compile()

    # ----------------------------------------------------------------------- #
    # Node 1: Enrich Market Context from Redis
    # ----------------------------------------------------------------------- #
    async def _node_enrich_context(self, state: RiskGateState) -> dict[str, Any]:
        """Fetch real-time order book snapshot from Redis."""
        context: dict[str, Any] = {}
        try:
            res = await self.bus.get_orderbook_snapshot("BTC/USDT")
            snap = res.data
            if snap is not None:
                context = {
                    "best_bid": snap.best_bid,
                    "best_ask": snap.best_ask,
                    "mid_price": snap.mid_price,
                    "spread_bps": round(snap.spread_bps, 4),
                    "obi": round(snap.obi, 4),
                    "obi_zscore": round(snap.obi_zscore, 3),
                }
        except Exception as exc:
            log.warning("Could not fetch order book context: %r", exc)

        return {"market_context": context}

    # ----------------------------------------------------------------------- #
    # Node 2: Evaluate Tail Risk (Cache -> Gemini 3.8 Flash -> Deterministic)
    # ----------------------------------------------------------------------- #
    async def _node_evaluate_tail_risk(self, state: RiskGateState) -> dict[str, Any]:
        headline = state["headline"]
        cache_key = headline.strip().lower()

        # 1. Cost-Saving In-Memory Cache Check
        if cache_key in self._headline_cache:
            cached = self._headline_cache[cache_key]
            return {
                "raw_assessment": cached,
                "engine_used": "IN_MEMORY_CACHE",
                "tokens_used": 0,
            }

        assessment: Optional[RiskAssessment] = None
        engine_used = "DETERMINISTIC_FALLBACK"
        tokens_used = 0

        # 2. Primary Live LLM Mode (Gemini 3.8 Flash)
        if state["use_live_llm"] and self._gemini_client is not None:
            t0 = time.perf_counter()
            prompt = (
                f"Role: Chief Risk Officer for crypto institutional HFT.\n"
                f"Assess market risk from this headline and orderbook context:\n"
                f"Headline: \"{headline}\"\n"
                f"L2 Context: {state.get('market_context', {})}\n"
                f"Output JSON strictly conforming to schema."
            )

            models_to_try = [self.model_name, "gemini-2.5-flash"]
            for model in models_to_try:
                try:
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RiskAssessment,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                    resp = self._gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if resp and resp.text:
                        assessment = RiskAssessment.model_validate_json(resp.text)
                        engine_used = "GEMINI_LIVE"
                        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                            tokens_used = resp.usage_metadata.total_token_count or 60
                        else:
                            tokens_used = 60
                        log.info(
                            "Gemini evaluation succeeded on model %s in %.2f ms",
                            model,
                            (time.perf_counter() - t0) * 1000,
                        )
                        break
                except Exception as exc:
                    log.warning(
                        "Gemini API call on %s failed: %r. Trying fallback...", model, exc
                    )

        # 3. Zero-Downtime Deterministic Fallback Engine
        if assessment is None:
            assessment = classify_deterministic(headline, state.get("market_context", {}))
            engine_used = "DETERMINISTIC_FALLBACK"
            tokens_used = 0

        # Cache the result
        self._headline_cache[cache_key] = assessment

        return {
            "raw_assessment": assessment,
            "engine_used": engine_used,
            "tokens_used": tokens_used,
        }

    # ----------------------------------------------------------------------- #
    # Node 3: Enforce Institutional Gating Policy & Log Audit
    # ----------------------------------------------------------------------- #
    async def _node_enforce_policy(self, state: RiskGateState) -> dict[str, Any]:
        raw = state["raw_assessment"]
        if raw is None:
            raw = classify_deterministic(state["headline"], state.get("market_context", {}))

        # Institutional Policy Guardrails
        if raw.risk_score >= 0.80:
            final_mult = 0.0
            final_cb = True
            category = "TAIL_RISK_BLACK_SWAN"
        elif 0.40 <= raw.risk_score < 0.80:
            # Enforce [0.20, 0.60] bound
            final_mult = max(0.20, min(0.60, float(raw.recommended_multiplier)))
            final_cb = False
            category = "MACRO_CAUTION"
        else:
            final_mult = 1.0
            final_cb = False
            category = "NORMAL"

        final_score = raw.risk_score
        reasoning = raw.reasoning

        # Write to Redis State Bus
        await self.bus.set_risk_state(
            risk_multiplier=final_mult,
            circuit_breaker=final_cb,
            reason=f"{category}: {reasoning}",
        )

        # Write to PostgreSQL risk_audit_logs
        audit_id = self.ledger.insert_risk_audit(
            headline=state["headline"],
            risk_score=final_score,
            new_risk_multiplier=final_mult,
            circuit_breaker_triggered=final_cb,
            reasoning=f"[{state['engine_used']}] {reasoning}",
        )

        return {
            "final_risk_score": final_score,
            "final_risk_multiplier": final_mult,
            "final_circuit_breaker": final_cb,
            "final_category": category,
            "final_reasoning": reasoning,
            "audit_id": audit_id,
        }

    # ----------------------------------------------------------------------- #
    # Public Execution API
    # ----------------------------------------------------------------------- #
    async def evaluate_headline(
        self,
        headline: str,
        source: str = "MANUAL",
        use_live_llm: bool = True,
    ) -> RiskGateState:
        """Run the full LangGraph pipeline on an incoming market headline."""
        t_start = time.perf_counter()
        initial_state: RiskGateState = {
            "headline": headline,
            "source": source,
            "use_live_llm": use_live_llm,
            "market_context": {},
            "raw_assessment": None,
            "final_risk_score": 0.0,
            "final_risk_multiplier": 1.0,
            "final_circuit_breaker": False,
            "final_category": "NORMAL",
            "final_reasoning": "",
            "engine_used": "",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "audit_id": None,
        }

        output = await self.graph.ainvoke(initial_state)
        output["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
        return output
