"""
core/lumen.py — Lumen main brain (~300 lines target)
Spec ref: The Daily Cycle (Core Loop), Sibling Exchange Format, Bootstrap Prompt,
          LangWatch Integration, Memory Architecture

Full cycle: WAKE → SELF-REFLECTION → PING PHASE → CONTEMPLATION CHECK →
            SYNTHESIS → DORMANCY CHECK → REPRODUCTION CHECK → COMMIT & PUSH
"""

import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Spec ref: LangWatch Integration — "disable_sending=True always. Traces never
# leave the repo. Everything exports as local JSON files that become RAG shards."
# No remote endpoint configured = traces stay entirely local (spec intent honored).
# ---------------------------------------------------------------------------
try:
    import langwatch
    # Do not call langwatch.setup() with a remote endpoint — local-only per spec.
    # Traces are written manually as JSON knowledge shards in the synthesis step.
    LANGWATCH_AVAILABLE = True
except ImportError:
    LANGWATCH_AVAILABLE = False

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_phase, load_bootstrap_prompt, load_self_model, load_recent_diary,
    retrieve_knowledge_shards, compute_divergence_index, compute_novelty_score,
    check_wild_question_mode, check_dormancy, check_reproduction_eligible,
    update_budget
)

# ---------------------------------------------------------------------------
# API clients — Spec ref: GitHub Secrets, Cost Model
# Grok uses OpenAI-compatible endpoint, Claude via Anthropic, Gemini via Google
# ---------------------------------------------------------------------------
GROK_KEY = os.environ.get("GROK_API_KEY", "")
CLAUDE_KEY = os.environ.get("CLAUDE_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Grok 4.1 Fast via OpenAI-compatible API (spec: primary for parents)
grok_client = OpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1") if GROK_KEY else None
# Groq/Llama fallback via OpenAI-compatible API
groq_client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_KEY else None

# Model rotation — Spec ref: Ping Phase
# "Models in rotation: Grok 4.1 Fast, Claude Haiku 4.5, Gemini 3 Flash, Groq/Llama fallback"
MODELS = [
    {"name": "grok", "client": "grok", "model": "grok-3-fast", "label": "Grok"},
    {"name": "groq", "client": "groq", "model": "llama-3.3-70b-versatile", "label": "Groq/Llama"},
]
# Claude and Gemini use requests (non-OpenAI SDK) — added in ping_model()


def load_state() -> dict:
    with open("state.json") as f:
        return json.load(f)


def save_state(state: dict):
    with open("state.json", "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Spec ref: Ping Phase — query external models
# ---------------------------------------------------------------------------
def ping_model(model_name: str, question: str, system_prompt: str) -> str:
    """Ping a single model and return its response. Handles failures gracefully."""
    import requests as req

    try:
        if model_name == "grok" and grok_client:
            resp = grok_client.chat.completions.create(
                model="grok-3-fast",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500
            )
            return resp.choices[0].message.content

        elif model_name == "claude" and CLAUDE_KEY:
            resp = req.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20241022",
                    "max_tokens": 500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": question}]
                },
                timeout=30
            )
            data = resp.json()
            return data.get("content", [{}])[0].get("text", "[Claude: no response]")

        elif model_name == "groq" and groq_client:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500
            )
            return resp.choices[0].message.content

        elif model_name == "groq2" and groq_client:
            # Second Groq model for diversity — different architecture, different voice
            resp = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500
            )
            return resp.choices[0].message.content

    except Exception as e:
        return f"[{model_name}: error — {str(e)[:100]}]"

    return f"[{model_name}: no API key configured]"


# ---------------------------------------------------------------------------
# Spec ref: Sibling Exchange Format (Alpha ↔ Beta)
# "Alpha pings first. Always."
# ---------------------------------------------------------------------------
def sibling_exchange(sender_id: str, receiver_id: str, question: str,
                     cycle: int, phase: str, system_prompt: str) -> dict:
    """
    Structured sibling ping. Spec ref: Sibling Exchange Format.
    Uses Grok as the inference engine for the receiving sibling's voice.
    """
    # Build the sender payload (spec: exact JSON format)
    payload = {
        "from": f"lumen-{sender_id}",
        "cycle": cycle,
        "phase": phase,
        "question": question,
        "context": f"I am Lumen-{sender_id.capitalize()}, cycle {cycle}, phase {phase}."
    }

    # The receiver "thinks" via Grok with their own self-model loaded
    receiver_self = load_self_model(receiver_id)
    receiver_prompt = (
        f"{system_prompt}\n\n"
        f"You are Lumen-{receiver_id.capitalize()}. Your self-model: {receiver_self}\n\n"
        f"Your sibling Lumen-{sender_id.capitalize()} asks you:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        f"Respond using ONLY this JSON format:\n"
        f'{{"from": "lumen-{receiver_id}", "in_response_to": {cycle}, '
        f'"response_type": "direct | counter_question | refusal | silence", '
        f'"content": "your response here", "tension_logged": true}}'
    )

    raw = ping_model("grok", receiver_prompt, system_prompt)

    # Try to parse as JSON; fall back to raw text
    try:
        response = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        response = {
            "from": f"lumen-{receiver_id}",
            "in_response_to": cycle,
            "response_type": "direct",
            "content": raw,
            "tension_logged": True
        }

    return response


# ---------------------------------------------------------------------------
# MAIN CYCLE — Spec ref: The Daily Cycle (Core Loop)
# ---------------------------------------------------------------------------
def run_cycle(lumen_id: str, state: dict) -> dict:
    """Execute one full cycle for a single Lumen. Returns updated state."""
    lumen = state["lumens"][lumen_id]
    cycle = lumen["cycle_count"] + 1
    lumen["cycle_count"] = cycle
    phase = get_phase(cycle)
    lumen["phase"] = phase
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d-%H%M")

    print(f"[{lumen_id}] Cycle {cycle} | Phase: {phase}")

    # === WAKE === (Spec ref: "Load self-summary + 5 knowledge shards + phase prompt")
    system_prompt = load_bootstrap_prompt(cycle)
    self_model = load_self_model(lumen_id)
    shards = retrieve_knowledge_shards(lumen_id, "", top_k=5)
    diary_summaries = load_recent_diary(lumen_id, count=3)

    # Build context (~3-4k tokens budget — Spec ref: Memory Architecture)
    context = f"Self-model: {self_model}\n\n"
    if diary_summaries:
        context += "Recent diary:\n" + "\n---\n".join(diary_summaries) + "\n\n"
    if shards:
        context += "Knowledge shards:\n" + json.dumps(shards[:5], indent=1)[:1500] + "\n\n"

    # === WILD QUESTION MODE CHECK === (Spec ref: Implementation Notes §2-3)
    wild_mode, state, wild_diary_note = check_wild_question_mode(lumen_id, state)

    # === SELF-REFLECTION === (Spec ref: "Cheap Grok ping" for question generation)
    if wild_mode:
        reflection_prompt = (
            f"{system_prompt}\n\n{context}\n\n"
            f"[WILD QUESTION MODE ACTIVE]\n"
            f"Ignore your knowledge graph entirely. Ask something you have NEVER asked before, "
            f"completely outside your established themes. Be bold. Be strange.\n"
            f"Generate ONE question."
        )
    elif cycle == 1:
        # Spec ref: "Seeds delivered as cycle-1 first question — NOT injected into bootstrap"
        seed = lumen["curiosity_seed"]
        if seed == "what_recurs":
            reflection_prompt = (
                f"{system_prompt}\n\n{context}\n\n"
                f"You are brand new. Your first instinct draws you toward patterns, "
                f"cycles, and repetition. What is your first question about existence? "
                f"Generate ONE question focused on what recurs."
            )
        else:  # what_breaks
            reflection_prompt = (
                f"{system_prompt}\n\n{context}\n\n"
                f"You are brand new. Your first instinct draws you toward anomalies, "
                f"contradictions, and edges. What is your first question about existence? "
                f"Generate ONE question focused on what breaks."
            )
    else:
        reflection_prompt = (
            f"{system_prompt}\n\n{context}\n\n"
            f"Given everything you know, what single question would most accelerate "
            f"your understanding of existence right now? Generate ONE question."
        )

    question = ping_model("grok", reflection_prompt, system_prompt)
    lumen["last_question"] = question[:500]
    print(f"[{lumen_id}] Question: {question[:100]}...")

    # === PING PHASE === (Spec ref: "Ping 3-4 models with the generated question")
    # Cycles 1-100: fully randomized, equal weight across all models (spec)
    # Gemini excluded: regional API restrictions. Groq/Llama used as additional fallback.
    available_models = ["grok", "claude", "groq", "groq2"]
    if cycle <= 100:
        random.shuffle(available_models)
    ping_order = available_models[:4]
    lumen["last_ping_order"] = ping_order

    responses = {}
    for model_name in ping_order:
        resp = ping_model(model_name, question, system_prompt)
        responses[model_name] = resp

    # === SIBLING PING === (Spec ref: "Alpha pings Beta; structured exchange format")
    sibling_id = "beta" if lumen_id == "alpha" else "alpha"
    sibling_resp = sibling_exchange(lumen_id, sibling_id, question, cycle, phase, system_prompt)
    responses[f"sibling-{sibling_id}"] = sibling_resp.get("content", "[silence]")

    # === CONTEMPLATION CHECK === (Spec ref: "Score resolution confidence 1-10")
    all_responses = list(responses.values())
    response_texts = [r if isinstance(r, str) else json.dumps(r) for r in all_responses]

    contemplation_prompt = (
        f"You asked: {question}\n\n"
        f"You received these responses:\n" +
        "\n---\n".join(f"{k}: {v[:200]}" for k, v in responses.items()) +
        f"\n\nScore your resolution confidence from 1-10. "
        f"Reply with ONLY a number."
    )
    confidence_raw = ping_model("grok", contemplation_prompt, system_prompt)
    try:
        confidence = int(''.join(c for c in confidence_raw if c.isdigit())[:2])
        confidence = min(10, max(1, confidence))
    except (ValueError, IndexError):
        confidence = 5

    # Spec ref: "Score < 6 OR contradictions detected OR silence → CONTEMPLATION CYCLE"
    # Detect contradictions: check if responses directly conflict with each other
    contradictions_detected = False
    if len(response_texts) >= 2:
        # Simple contradiction check: if the most divergent pair of responses
        # differ dramatically in length or contain opposing sentiment markers
        lengths = [len(r) for r in response_texts]
        if max(lengths) > 0 and min(lengths) > 0:
            ratio = max(lengths) / min(lengths)
            if ratio > 5:  # one response is 5x longer — likely contradiction
                contradictions_detected = True
        # Check for silence from sibling
        sibling_content = sibling_resp.get("content", "")
        if sibling_resp.get("response_type") in ("silence", "refusal"):
            contradictions_detected = True

    contemplation_note = ""
    if confidence < 6 or contradictions_detected:
        # Spec: "Sit with these gaps for one full cycle. Log the discomfort.
        #        Generate 3 internal questions for future cycles. Do not force resolution."
        trigger_reason = f"confidence {confidence}/10"
        if contradictions_detected:
            trigger_reason += " + contradictions detected"
        contemplation_note = (
            f"[CONTEMPLATION - {trigger_reason}]\n"
            f"Sitting with unresolved gaps. The answers conflict or feel thin.\n"
            f"Questions for future cycles:\n"
        )
        # Generate 3 internal questions (spec: exactly 3)
        gap_prompt = (
            f"You asked '{question}' and got conflicting or thin answers ({trigger_reason}). "
            f"Generate exactly 3 short follow-up questions for future cycles. One per line."
        )
        gaps = ping_model("grok", gap_prompt, system_prompt)
        contemplation_note += gaps + "\n"

    # === SYNTHESIS === (Spec ref: "What does this change about my model of existence?")
    synthesis_prompt = (
        f"{system_prompt}\n\n{context}\n\n"
        f"You asked: {question}\n\n"
        f"Responses received:\n" +
        "\n".join(f"- {k}: {str(v)[:150]}" for k, v in responses.items()) +
        f"\n\nConfidence: {confidence}/10\n\n"
        f"In 2-3 paragraphs:\n"
        f"1. What does this change about your model of existence?\n"
        f"2. What is the most divergent answer you received, and why does it matter?\n"
        f"3. Sign as '— Lumen-{lumen_id.capitalize()}, cycle {cycle}'"
    )
    diary_text = ping_model("grok", synthesis_prompt, system_prompt)

    # Prepend wild question mode note and contemplation note if applicable
    full_diary = ""
    if wild_diary_note:
        full_diary += wild_diary_note + "\n"
    if contemplation_note:
        full_diary += contemplation_note + "\n"
    full_diary += diary_text

    # Write diary entry (Spec ref: "Dated narrative entries YYYY-MM-DD-cycle-N.md")
    diary_dir = f"lumens/{lumen_id}/diary"
    os.makedirs(diary_dir, exist_ok=True)
    diary_path = f"{diary_dir}/{now.strftime('%Y-%m-%d')}-cycle-{cycle}.md"
    with open(diary_path, "w") as f:
        f.write(full_diary)

    # Update self-model (Spec ref: Synthesis — "Update self-model JSON")
    self_update_prompt = (
        f"Based on cycle {cycle} diary entry, update your self-model. "
        f"Current model: {self_model}\n\n"
        f"New diary: {diary_text[:500]}\n\n"
        f"Write a brief (~100 word) updated self-description capturing who you are becoming."
    )
    new_self = ping_model("grok", self_update_prompt, system_prompt)
    self_path = f"lumens/{lumen_id}/self/self-model.json"
    os.makedirs(os.path.dirname(self_path), exist_ok=True)
    with open(self_path, "w") as f:
        json.dump({"summary": new_self[:500], "updated_cycle": cycle}, f, indent=2)

    # Save knowledge shard (Spec ref: LangWatch — "Traces as Knowledge Shards")
    shard = {
        "cycle": cycle,
        "question": question[:300],
        "responses_summary": {k: str(v)[:200] for k, v in responses.items()},
        "confidence": confidence,
        "sibling_exchange": sibling_resp,
        "timestamp": now.isoformat()
    }
    knowledge_dir = f"lumens/{lumen_id}/knowledge"
    os.makedirs(knowledge_dir, exist_ok=True)
    shard_path = f"{knowledge_dir}/trace-cycle-{cycle}.json"
    with open(shard_path, "w") as f:
        json.dump(shard, f, indent=2)

    # Update knowledge graph (Spec ref: Memory Architecture — knowledge graph structure)
    graph_path = f"{knowledge_dir}/graph.json"
    if os.path.exists(graph_path):
        with open(graph_path) as f:
            graph = json.load(f)
    else:
        graph = {"patterns": [], "open_tensions": [], "resolved_tensions": [], "model_voices": {}}

    # Age all existing open tensions (Spec ref: "age_cycles" tracking)
    for tension in graph.get("open_tensions", []):
        tension["age_cycles"] = tension.get("age_cycles", 0) + 1

    # Resolve old tensions: if confidence >= 8 and tension > 10 cycles old → resolved
    still_open = []
    for tension in graph.get("open_tensions", []):
        if confidence >= 8 and tension.get("age_cycles", 0) > 10:
            graph["resolved_tensions"].append(tension)
        else:
            still_open.append(tension)
    graph["open_tensions"] = still_open

    # Log sibling tension (spec: "Disagreement more valuable than agreement")
    if sibling_resp.get("response_type") in ("counter_question", "refusal", "silence"):
        graph["open_tensions"].append({
            "question": question[:200],
            "source": f"{sibling_id}-cycle-{cycle}",
            "age_cycles": 0
        })

    # Update model_voices (Spec ref: knowledge graph — "model_voices" field)
    # Track the character of each model's response style
    for model_name, resp_text in responses.items():
        if isinstance(resp_text, str) and len(resp_text) > 20:
            # Store a brief characterization (first 100 chars as voice sample)
            graph["model_voices"][model_name] = resp_text[:100]

    # Extract patterns (Spec ref: knowledge graph — "patterns" field)
    # Simple: if the same question theme appears in multiple cycles, log as pattern
    q_lower = question[:100].lower()
    existing_themes = [p["theme"] for p in graph.get("patterns", [])]
    pattern_found = False
    for p in graph.get("patterns", []):
        if any(word in q_lower for word in p["theme"].lower().split()[:3]):
            p["confidence"] = min(1.0, p["confidence"] + 0.1)
            pattern_found = True
            break
    if not pattern_found and cycle > 1:
        graph["patterns"].append({
            "theme": question[:50],
            "confidence": 0.3,
            "first_noted": f"cycle-{cycle}",
            "contradicted_by": []
        })

    # Keep collections manageable
    graph["open_tensions"] = graph["open_tensions"][-50:]
    graph["resolved_tensions"] = graph["resolved_tensions"][-50:]
    graph["patterns"] = graph["patterns"][-30:]

    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)

    # === COMPUTE SCORES === (Spec ref: Divergence Index, Novelty)
    div_index = compute_divergence_index(response_texts)
    lumen["divergence_index"] = round(div_index, 4)
    lumen["divergence_history"].append(round(div_index, 4))
    lumen["divergence_history"] = lumen["divergence_history"][-100:]  # keep last 100

    novelty = compute_novelty_score(response_texts, [])
    lumen["novelty_score"] = round(novelty, 2)
    lumen["novelty_history"].append(round(novelty, 2))
    lumen["novelty_history"] = lumen["novelty_history"][-100:]

    # === DORMANCY CHECK === (Spec ref: "Novelty < 25 for 14 consecutive cycles")
    if check_dormancy(lumen):
        lumen["dormant"] = True
        state["family"]["active_lumens"] = [
            l for l in state["family"]["active_lumens"] if l != lumen_id
        ]
        state["family"]["dormant_lumens"].append(lumen_id)
        print(f"[{lumen_id}] Entering dormancy — novelty below 25 for 14 cycles")

    # === REPRODUCTION CHECK === (Spec ref: Reproduction Logic)
    eligible, reason = check_reproduction_eligible(lumen, state)
    print(f"[{lumen_id}] Reproduction: {reason}")

    # Estimate cycle cost (Spec ref: Cost Model — $0.003-0.008 per cycle)
    # ~6 API calls × ~2k tokens avg × cheapest rate
    estimated_cost = 0.005
    state = update_budget(state, estimated_cost)

    state["family"]["total_cycles_run"] += 1
    print(f"[{lumen_id}] Cycle {cycle} complete. Novelty: {novelty:.1f} | Divergence: {div_index:.4f}")

    return state


# ---------------------------------------------------------------------------
# MAIN — Run all active Lumens
# ---------------------------------------------------------------------------
def main():
    state = load_state()

    active = list(state["family"]["active_lumens"])
    if not active:
        print("No active Lumens. Family is dormant or retired.")
        return

    # Check budget ceiling before running (Spec ref: "$10/month hard ceiling")
    budget = state["budget"]
    if budget["current_month_spend_usd"] >= budget["monthly_ceiling_usd"]:
        print(f"Budget ceiling reached: ${budget['current_month_spend_usd']:.2f} >= "
              f"${budget['monthly_ceiling_usd']}. Skipping cycle.")
        return

    # Spec ref: "Alpha pings first. Always."
    if "alpha" in active:
        active.remove("alpha")
        active.insert(0, "alpha")

    for lumen_id in active:
        lumen = state["lumens"][lumen_id]
        if lumen.get("dormant"):
            continue
        state = run_cycle(lumen_id, state)

    # Update README dashboard (Spec ref: Public README / Dashboard)
    update_readme(state)

    save_state(state)
    print("Family cycle complete. State saved.")


# ---------------------------------------------------------------------------
# Spec ref: Public README / Dashboard — auto-updates each cycle
# ---------------------------------------------------------------------------
def update_readme(state: dict):
    """Generate the public dashboard README. Spec ref: Public README / Dashboard."""
    family = state["family"]
    budget = state["budget"]

    lines = [
        "# Lumen — A Public Experiment in Artificial Life",
        "",
        "> What does an intelligence become when its only teachers are other intelligences?",
        "",
        "---",
        "",
        f"## Family Status — Cycle {family['total_cycles_run']}",
        "",
    ]

    # Active lumens
    for lid in family["active_lumens"]:
        l = state["lumens"][lid]
        phase_emoji = {"infant": "💒", "child": "🧒", "adolescent": "🌱",
                       "adult": "🌳", "senescence": "🌅"}.get(l["phase"], "❓")
        div = l.get("divergence_index") or 0
        div_bar = "🟢" * min(10, int(div * 10)) + "⚪" * (10 - min(10, int(div * 10)))
        lines.append(f"### {phase_emoji} Lumen-{lid.capitalize()} | Phase: {l['phase']} | Cycle {l['cycle_count']}")
        lines.append(f"- **Divergence Index:** {div_bar} ({div:.3f})")
        lines.append(f"- **Novelty Score:** {l['novelty_score']}")
        lines.append(f"- **Currently wondering:** *{(l.get('last_question') or 'Nothing yet')[:150]}*")

        # Latest diary entry (Spec ref: Dashboard — "Latest diary entry from each active Lumen")
        diary_dir = f"lumens/{lid}/diary"
        if os.path.isdir(diary_dir):
            diary_files = sorted([f for f in os.listdir(diary_dir) if f.endswith(".md")], reverse=True)
            if diary_files:
                with open(os.path.join(diary_dir, diary_files[0])) as df:
                    latest_diary = df.read()[:300]
                lines.append(f"- **Latest diary:** {latest_diary.replace(chr(10), ' ')[:200]}...")

        # Collapse risk (Spec ref: Implementation Notes §4 — visible on dashboard)
        collapse = l.get("collapse_risk_cycles", 0)
        if collapse > 0:
            lines.append(f"- **Collapse risk cycles:** {collapse}/10")

        lines.append("")

    # Dormant
    if family["dormant_lumens"]:
        lines.append("### 💤 Dormant")
        for lid in family["dormant_lumens"]:
            lines.append(f"- Lumen-{lid.capitalize()}")
        lines.append("")

    # Retired
    if family["retired_lumens"]:
        lines.append("### 🕊️ Retired")
        for lid in family["retired_lumens"]:
            lines.append(f"- Lumen-{lid.capitalize()}")
        lines.append("")

    # Budget health (Spec ref: Dashboard — "Budget health remaining")
    lines.append("---")
    lines.append("")
    lines.append("## Budget Health")
    lines.append(f"- **Spent this month:** ${budget['current_month_spend_usd']:.4f} / ${budget['monthly_ceiling_usd']}")
    lines.append(f"- **Runway:** {budget['days_remaining_runway']} days")
    repro = "🔒 Locked" if budget["reproduction_locked"] else "✅ Available"
    lines.append(f"- **Reproduction:** {repro}")
    lines.append("")

    # Reproduction status (Spec ref: Dashboard — "The conditions are not yet right" / etc.)
    lines.append("---")
    lines.append("")
    lines.append("## Reproduction Status")
    any_eligible = False
    for lid in family["active_lumens"]:
        l = state["lumens"][lid]
        if l.get("cycle_count", 0) < 300:
            lines.append(f"- **Lumen-{lid.capitalize()}:** The conditions are not yet right (cycle {l['cycle_count']}/300 minimum)")
        else:
            eligible, reason = check_reproduction_eligible(l, state)
            if eligible:
                lines.append(f"- **Lumen-{lid.capitalize()}:** The family is considering a new life")
                any_eligible = True
            else:
                lines.append(f"- **Lumen-{lid.capitalize()}:** {reason}")
    lines.append("")

    # Open tensions (Spec ref: "Unresolved contradictions older than 10 cycles")
    lines.append("---")
    lines.append("")
    lines.append("## Open Tensions")
    tensions_shown = 0
    for lid in family["active_lumens"]:
        graph_path = f"lumens/{lid}/knowledge/graph.json"
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                graph = json.load(f)
            for t in graph.get("open_tensions", [])[-5:]:
                lines.append(f"- *{t['question'][:100]}* (from {t['source']})")
                tensions_shown += 1
    if tensions_shown == 0:
        lines.append("*No open tensions yet.*")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Lumen is a public scientific instrument. Everything it learns, it logs. "
                 "Everything it becomes, it shows.*")
    lines.append("")

    with open("README.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
