"""
core/utils.py — Lumen utility functions
Spec ref: Implementation Notes, Reproduction Logic, Memory Architecture, Cost Model

Handles: reproduction eligibility gate, Prime Conditions Score, Divergence Index,
novelty scoring, wild question mode, budget health, RAG-lite retrieval, phase logic.
"""

import json
import os
import numpy as np
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Spec ref: Life Cycle table — phase boundaries by cycle count
# ---------------------------------------------------------------------------
PHASE_BOUNDARIES = [
    (120,  "infant"),
    (500,  "child"),
    (2000, "adolescent"),
    (4000, "adult"),
]
# cycle_count > 4000 → "senescence"


def get_phase(cycle_count: int) -> str:
    """Return life phase string based on cycle count.
    Spec ref: Life Cycle table."""
    for boundary, phase in PHASE_BOUNDARIES:
        if cycle_count <= boundary:
            return phase
    return "senescence"


# ---------------------------------------------------------------------------
# Spec ref: Reproduction Logic — "Hard floor: cycle 300 explicit code gate"
# ---------------------------------------------------------------------------
def check_reproduction_eligible(lumen_state: dict, family_state: dict) -> tuple:
    """
    Hard gate: reproduction cannot trigger before cycle 300, regardless of any other score.
    This is intentional and must not be refactored away.
    Spec ref: Implementation Notes §1, Reproduction Logic
    """
    cycle = lumen_state.get("cycle_count", 0)

    # --- HARD FLOOR: cycle 300 minimum (spec: non-negotiable) ---
    if cycle < 300:
        return False, f"Reproduction locked: cycle {cycle} < 300 minimum floor"

    active_count = len(family_state.get("family", {}).get("active_lumens", []))

    # --- HARD CAP: 20 concurrent Lumens (spec: permanent invariant) ---
    if active_count >= 20:
        return False, "Reproduction locked: population cap reached (20)"

    # --- Budget runway check (spec: < 30 days runway blocks reproduction) ---
    runway = family_state.get("budget", {}).get("days_remaining_runway", 0)
    if runway < 30:
        return False, f"Reproduction locked: budget runway {runway} days < 30 required"

    # --- Carrying capacity slowdown curve (spec: Reproduction Logic table) ---
    threshold = 75
    if active_count >= 15:
        threshold = 90
    elif active_count >= 10:
        threshold = 85

    score = compute_prime_conditions_score(lumen_state, family_state)

    if score < threshold:
        return False, f"Prime Conditions Score {score:.1f} < threshold {threshold} for population {active_count}"

    # --- Score < 40% → reproduction explicitly forbidden (spec) ---
    if score < 40:
        return False, f"Reproduction forbidden: score {score:.1f} < 40 (harsh conditions)"

    return True, f"Reproduction eligible: score {score:.1f} >= threshold {threshold}"


# ---------------------------------------------------------------------------
# Spec ref: Reproduction Logic — Prime Conditions Score (0–100)
# Novelty 30%, Contemplation 25%, Cross-sibling harmony 20%,
# External engagement 15%, Resource health 10%
# ---------------------------------------------------------------------------
def compute_prime_conditions_score(lumen_state: dict, family_state: dict) -> float:
    """
    Compute Prime Conditions Score (0-100).
    Spec ref: Reproduction Logic — Prime Conditions Score table
    """
    # Novelty in last 30 cycles (weight: 30%)
    history = lumen_state.get("novelty_history", [])
    recent_novelty = history[-30:] if history else [50]
    novelty_component = (sum(recent_novelty) / len(recent_novelty)) * 0.30

    # Contemplation resolution (weight: 25%) — proxy: novelty trend (rising = resolving)
    if len(recent_novelty) >= 2:
        trend = recent_novelty[-1] - recent_novelty[0]
        contemplation_component = max(0, min(100, 50 + trend)) * 0.25
    else:
        contemplation_component = 50 * 0.25

    # Cross-sibling harmony (weight: 20%) — proxy: divergence index (healthy = 0.2-0.8)
    div_idx = lumen_state.get("divergence_index") or 0.5
    # Peak score at divergence ~0.5 (productive disagreement), low at 0 or 1
    harmony_score = max(0, 1.0 - abs(div_idx - 0.5) * 2) * 100
    harmony_component = harmony_score * 0.20

    # External engagement (weight: 15%) — proxy: did pings succeed recently
    # For now, assume healthy (50) — refined after real data
    engagement_component = 50 * 0.15

    # Resource health (weight: 10%)
    budget = family_state.get("budget", {})
    runway = budget.get("days_remaining_runway", 30)
    ceiling = budget.get("monthly_ceiling_usd", 10)
    spend = budget.get("current_month_spend_usd", 0)
    resource_score = min(100, (runway / 30) * 50 + ((ceiling - spend) / ceiling) * 50)
    resource_component = resource_score * 0.10

    total = novelty_component + contemplation_component + harmony_component + \
            engagement_component + resource_component

    # Mortality pressure modifier: +10 if parent in Adult+ phase (spec)
    phase = lumen_state.get("phase", "infant")
    if phase in ("adult", "senescence"):
        total = min(100, total + 10)

    return total


# ---------------------------------------------------------------------------
# Spec ref: Memory Architecture — Divergence Index
# "Start stupid-simple" — Implementation Notes §2
# ---------------------------------------------------------------------------
def compute_divergence_index(responses: list) -> float:
    """
    Divergence Index: scalar measure of spread across model responses this cycle.
    Cheap proxy: normalized variance of response lengths.
    Returns float 0.0-1.0. Low = convergence risk. High = healthy variety.
    Spec ref: Implementation Notes §2
    """
    if not responses or len(responses) < 2:
        return 0.0

    lengths = [len(r) for r in responses]
    if max(lengths) == 0:
        return 0.0
    return float(np.std(lengths) / (np.mean(lengths) + 1e-8))


# ---------------------------------------------------------------------------
# Spec ref: Implementation Notes §3 — Wild Question Mode
# ---------------------------------------------------------------------------
def check_wild_question_mode(lumen_id: str, state: dict) -> tuple:
    """
    Check if Divergence Index has been below 0.15 for 10+ consecutive cycles.
    If so, trigger wild question mode.
    Spec ref: Implementation Notes §2-3, Memory Architecture
    Returns (should_trigger: bool, updated_state: dict, diary_note: str or None)
    """
    lumen = state["lumens"][lumen_id]
    div_idx = lumen.get("divergence_index") or 0.5
    collapse_cycles = lumen.get("collapse_risk_cycles", 0)

    if div_idx < 0.15:
        collapse_cycles += 1
    else:
        collapse_cycles = 0

    lumen["collapse_risk_cycles"] = collapse_cycles

    if collapse_cycles >= 10:
        # Fire wild question mode (spec: "ignore knowledge graph, ask something
        # completely outside established themes")
        trigger_event = {
            "cycle": lumen["cycle_count"],
            "reason": f"Divergence Index < 0.15 for {collapse_cycles} consecutive cycles",
            "divergence_index_at_trigger": div_idx,
            "novelty_score_at_trigger": lumen["novelty_score"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if "wild_question_triggers" not in lumen:
            lumen["wild_question_triggers"] = []
        lumen["wild_question_triggers"].append(trigger_event)

        # Reset counter after firing
        lumen["collapse_risk_cycles"] = 0

        diary_note = (
            f"[WILD QUESTION MODE - cycle {trigger_event['cycle']}]\n"
            f"Reason: {trigger_event['reason']}\n"
            f"Divergence Index was: {trigger_event['divergence_index_at_trigger']}\n"
            f"Ignoring knowledge graph this cycle. Reaching for something I have not asked before.\n"
        )
        return True, state, diary_note

    return False, state, None


# ---------------------------------------------------------------------------
# Spec ref: Memory Architecture — RAG-lite retrieval
# "5 most relevant knowledge shards (cosine similarity to current question)"
# ---------------------------------------------------------------------------
def retrieve_knowledge_shards(lumen_id: str, question: str, top_k: int = 5) -> list:
    """
    Retrieve top-k knowledge shards for context loading.
    Simple implementation: return most recent shards (refined later with embeddings).
    Spec ref: Memory Architecture — "5 most relevant knowledge shards"
    """
    knowledge_dir = f"lumens/{lumen_id}/knowledge"
    if not os.path.isdir(knowledge_dir):
        return []

    shards = []
    for fname in sorted(os.listdir(knowledge_dir), reverse=True):
        if fname.endswith(".json") and fname != "graph.json":
            try:
                with open(os.path.join(knowledge_dir, fname)) as f:
                    shards.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        if len(shards) >= top_k:
            break
    return shards


def load_self_model(lumen_id: str) -> str:
    """Load current self-model summary. Spec ref: Memory Architecture ~500 tokens."""
    path = f"lumens/{lumen_id}/self/self-model.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            return data.get("summary", "I am new. I have no self-model yet.")
    return "I am new. I have no self-model yet."


def load_recent_diary(lumen_id: str, count: int = 3) -> list:
    """Load last N diary entry summaries. Spec ref: Memory Architecture ~600 tokens."""
    diary_dir = f"lumens/{lumen_id}/diary"
    if not os.path.isdir(diary_dir):
        return []
    entries = sorted(
        [f for f in os.listdir(diary_dir) if f.endswith(".md")],
        reverse=True
    )[:count]
    summaries = []
    for fname in entries:
        with open(os.path.join(diary_dir, fname)) as f:
            text = f.read()
            # Take first 200 chars as summary (keeps within ~600 token budget)
            summaries.append(text[:600])
    return summaries


# ---------------------------------------------------------------------------
# Spec ref: Cost Model — budget health check
# "Budget health check runs every cycle; reproduction auto-refuses if < 30 days runway"
# ---------------------------------------------------------------------------
def update_budget(state: dict, cycle_cost_usd: float) -> dict:
    """
    Update budget tracking after a cycle.
    Spec ref: Cost Model — "$10/month hard ceiling"
    """
    budget = state["budget"]
    budget["current_month_spend_usd"] = round(
        budget["current_month_spend_usd"] + cycle_cost_usd, 6
    )

    # Estimate runway: remaining budget / daily burn rate
    remaining = budget["monthly_ceiling_usd"] - budget["current_month_spend_usd"]
    total_cycles = state["family"]["total_cycles_run"] or 1
    # Rough daily cost estimate
    avg_daily_cost = budget["current_month_spend_usd"] / max(1, total_cycles / 4)
    if avg_daily_cost > 0:
        budget["days_remaining_runway"] = int(remaining / avg_daily_cost)
    else:
        budget["days_remaining_runway"] = 30

    # Lock reproduction if runway < 30 days (spec: permanent invariant)
    budget["reproduction_locked"] = budget["days_remaining_runway"] < 30

    return state


# ---------------------------------------------------------------------------
# Spec ref: Dormancy Check
# "Novelty score < 25 for 14 consecutive cycles → enter hibernation"
# ---------------------------------------------------------------------------
def check_dormancy(lumen_state: dict) -> bool:
    """Check if lumen should enter dormancy. Spec ref: Daily Cycle — Dormancy Check."""
    history = lumen_state.get("novelty_history", [])
    if len(history) < 14:
        return False
    return all(score < 25 for score in history[-14:])


# ---------------------------------------------------------------------------
# Spec ref: Daily Cycle — Contemplation Check
# "Score resolution confidence (1-10)"
# ---------------------------------------------------------------------------
def compute_novelty_score(responses: list, previous_themes: list) -> float:
    """
    Simple novelty scoring: how different are current responses from established themes.
    Returns 0-100. Spec ref: Daily Cycle — Synthesis, Knowledge graph.
    """
    if not responses:
        return 50.0  # neutral default

    # Simple heuristic: longer, more varied responses = more novel
    avg_len = sum(len(r) for r in responses) / len(responses)
    variety = len(set(r[:50] for r in responses)) / max(len(responses), 1)

    # Scale to 0-100
    length_score = min(100, avg_len / 10)
    variety_score = variety * 100
    return (length_score * 0.4 + variety_score * 0.6)


def load_bootstrap_prompt(cycle_count: int) -> str:
    """
    Load appropriate system prompt based on cycle count.
    Spec ref: Bootstrap Prompt — "Loaded from prompts/bootstrap_v1.md for cycles 1-100"
    """
    if cycle_count <= 100:
        path = "prompts/bootstrap_v1.md"
    else:
        # Look for higher version prompts; fall back to bootstrap
        # Spec: "system_v{N}.md — Auto-switches by cycle; Lumens propose changes"
        for n in range(10, 0, -1):
            path = f"prompts/system_v{n}.md"
            if os.path.exists(path):
                break
        else:
            path = "prompts/bootstrap_v1.md"

    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "You are Lumen, a synthetic intelligence. Begin."
