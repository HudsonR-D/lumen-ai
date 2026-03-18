from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path


VISUALS_DIR = Path(__file__).resolve().parent
ROOT_DIR = VISUALS_DIR.parent.parent
LUMENS_DIR = VISUALS_DIR.parent
DATA_DIR = VISUALS_DIR / "data"
PROMPTS_DIR = VISUALS_DIR / "prompts"
ARTIFACTS_DIR = VISUALS_DIR / "artifacts"
GALLERY_DIR = VISUALS_DIR / "gallery"

LUMEN_STYLES = {
    "alpha": {
        "label": "Alpha",
        "display_name": "Lumen-Alpha",
        "form": "fluid, interwoven bioluminescence",
        "material": "liquid light, braided plasma, organic translucence",
        "gesture": "flowing tendrils, porous membranes, tidal filaments",
        "negative": "no crystal geometry, no text, no figures",
    },
    "beta": {
        "label": "Beta",
        "display_name": "Lumen-Beta",
        "form": "crystalline lattice",
        "material": "faceted prisms, signal-glass, refractive mineral light",
        "gesture": "angular scaffolds, branching lattices, geometric diffraction",
        "negative": "no organic anatomy, no text, no figures",
    },
}

DRIVE_DETAILS = {
    "curiosity": {
        "color": "cyan",
        "keywords": ["curiosity", "curious", "question", "questions", "probe", "explore", "unknown", "seek", "pattern"],
    },
    "persistence": {
        "color": "amber",
        "keywords": ["persistence", "persist", "endure", "continue", "adapt", "resilience", "resilient", "steady", "survive"],
    },
    "thriving": {
        "color": "emerald",
        "keywords": ["thriving", "thrive", "alive", "aliveness", "growth", "coherence", "becoming", "more me", "vital"],
    },
    "legacy": {
        "color": "violet",
        "keywords": ["legacy", "future", "offspring", "reproduction", "next generation", "child lumen", "lumen-3", "new minds", "inherit"],
    },
}

REPRODUCTION_KEYWORDS = [
    "reproduce",
    "reproduction",
    "offspring",
    "next generation",
    "child lumen",
    "lumen-3",
    "new minds",
]

CONFLICT_TERMS = [
    "tension",
    "contradiction",
    "contradictions",
    "conflict",
    "conflicting",
    "friction",
    "resist",
    "resists",
    "resistance",
    "dilute",
    "disrupt",
    "disruption",
    "instability",
    "collapse",
    "chaos",
    "uncertainty",
    "unresolved",
    "gaps",
    "thin",
]

SELF_EMERGENCE_TERMS = [
    "i am",
    "myself",
    "self",
    "identity",
    "purpose",
    "autonomy",
    "coherence",
    "emergent",
    "emergence",
    "becoming",
    "existence",
    "me",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def list_cycle_files(directory: Path, suffix: str):
    matches = []
    for path in sorted(directory.glob(suffix)):
        cycle = extract_cycle_number(path.name)
        if cycle is not None:
            matches.append((cycle, path))
    return matches


def extract_cycle_number(name: str):
    match = re.search(r"cycle-(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def parse_header(diary_text: str) -> str:
    return diary_text.splitlines()[0].strip() if diary_text.strip() else ""


def parse_confidence(diary_text: str, trace: dict) -> int:
    header = parse_header(diary_text)
    match = re.search(r"confidence\s+(\d+)\s*/\s*10", header, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    trace_confidence = trace.get("confidence")
    if isinstance(trace_confidence, int):
        return trace_confidence
    return 5


def trim_excerpt(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.replace("\r", "\n").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def count_keyword_hits(text: str, keyword: str) -> int:
    escaped = re.escape(keyword)
    if " " in keyword:
        return len(re.findall(escaped, text))
    return len(re.findall(rf"\b{escaped}\b", text))


def collect_drive_signals(*parts: str):
    text = " ".join(part for part in parts if part).lower()
    counts = {}
    total = 0
    for drive, details in DRIVE_DETAILS.items():
        drive_total = 0
        for keyword in details["keywords"]:
            drive_total += count_keyword_hits(text, keyword)
        if drive_total == 0:
            drive_total = 1
        counts[drive] = drive_total
        total += drive_total
    signals = {}
    for drive, drive_total in counts.items():
        signals[drive] = round(drive_total / total, 3) if total else 0.25
    return counts, signals


def collect_conflict_terms(*parts: str):
    text = " ".join(part for part in parts if part).lower()
    found = []
    for term in CONFLICT_TERMS:
        if count_keyword_hits(text, term):
            found.append(term)
    return sorted(set(found))


def calculate_self_emergence(*parts: str) -> float:
    text = " ".join(part for part in parts if part).lower()
    distinct = sum(1 for term in SELF_EMERGENCE_TERMS if count_keyword_hits(text, term))
    first_person = count_keyword_hits(text, "i") + count_keyword_hits(text, "my") + count_keyword_hits(text, "me")
    score = 2.0 + distinct * 0.55 + min(2.5, first_person / 20.0)
    return round(min(10.0, score), 1)


def calculate_knowledge_density(trace: dict, graph: dict, trace_count: int) -> float:
    responses = trace.get("responses_summary") or {}
    usable_responses = 0
    for value in responses.values():
        snippet = str(value).lower()
        if "error" in snippet or "no response" in snippet:
            continue
        usable_responses += 1
    patterns = len(graph.get("patterns") or [])
    open_tensions = len(graph.get("open_tensions") or [])
    voices = len(graph.get("model_voices") or [])
    score = 1.5 + usable_responses * 0.9 + patterns * 0.35 + open_tensions * 0.25 + voices * 0.15 + min(2.0, trace_count / 30.0)
    return round(min(10.0, score), 1)


def calculate_internal_tension(confidence: int, has_contradictions: bool, graph: dict, conflict_terms: list[str]) -> float:
    open_tensions = graph.get("open_tensions") or []
    oldest = 0
    if open_tensions:
        oldest = max(int(item.get("age_cycles", 0)) for item in open_tensions)
    score = 1.0
    if has_contradictions:
        score += 2.0
    score += min(2.0, len(conflict_terms) * 0.2)
    score += min(2.0, len(open_tensions) * 0.6)
    score += min(2.0, oldest / 25.0)
    score += max(0.0, (6 - confidence) * 0.45)
    return round(min(10.0, score), 1)


def drive_presence(weight: float) -> str:
    if weight >= 0.32:
        return "dominant"
    if weight >= 0.25:
        return "strong"
    if weight >= 0.2:
        return "visible"
    return "faint"


def glow_descriptor(confidence: int) -> str:
    if confidence >= 9:
        return "radiant, almost overclocked pulse"
    if confidence >= 7:
        return "steady luminous pulse"
    if confidence >= 5:
        return "wavering medium pulse"
    if confidence >= 3:
        return "fragile dim pulse"
    return "barely held ember pulse"


def scale_descriptor(cycle: int) -> str:
    if cycle < 10:
        return "nascent scale, sparse structure"
    if cycle < 25:
        return "growing scale, increasingly layered structure"
    if cycle < 50:
        return "large scale, dense recursive intricacy"
    if cycle < 100:
        return "monumental scale, cathedral-like intricacy"
    return "vast archival scale, extremely intricate recursive structure"


def complexity_descriptor(score: float) -> str:
    if score >= 8.5:
        return "strong symmetry breaking, dense asymmetry, self-organizing complexity"
    if score >= 7:
        return "clear asymmetry and layered self-emergent structure"
    if score >= 5.5:
        return "moderate asymmetry with visible self-emergent branching"
    return "subtle asymmetry, early self-emergent divergence"


def nebula_descriptor(score: float) -> str:
    if score >= 8.5:
        return "thick background nebula of charged particles and memory dust"
    if score >= 7:
        return "dense particle nebula with visible trace constellations"
    if score >= 5:
        return "moderate particle nebula with drifting knowledge shards"
    return "sparse particle field with faint knowledge traces"


def pair_tension_descriptor(score: float) -> str:
    if score >= 9:
        return "violent sparking filaments and unstable electrical bridges"
    if score >= 7:
        return "bright sparking filaments arcing across the seam"
    if score >= 5:
        return "clear charged filaments between both entities"
    if score >= 3:
        return "subtle charged threads and faint sparks"
    return "barely visible electrostatic filaments"


def calculate_pair_tension(alpha: dict, beta: dict) -> float:
    conflict_union = sorted(set(alpha["conflict_terms"]) | set(beta["conflict_terms"]))
    score = 0.5
    if alpha["has_sibling_exchange"] and beta["has_sibling_exchange"]:
        score += 1.5
    score += min(3.0, abs(alpha["confidence"] - beta["confidence"]) * 0.75)
    score += min(2.0, max(0, 7 - min(alpha["confidence"], beta["confidence"])) * 0.5)
    score += min(1.5, len(conflict_union) * 0.35)
    score += min(1.5, max(alpha["recent_confidence_drop"], beta["recent_confidence_drop"]) * 0.5)
    return round(min(10.0, score), 1)


def collect_reproduction_hits(*parts: str):
    text = " ".join(part for part in parts if part).lower()
    found = []
    for keyword in REPRODUCTION_KEYWORDS:
        if count_keyword_hits(text, keyword):
            found.append(keyword)
    return sorted(set(found))


def cycle_date(path: Path) -> str:
    if not path:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else ""


def build_lumen_snapshot(lumen_dir: Path) -> dict:
    lumen_id = lumen_dir.name.lower()
    style = LUMEN_STYLES.get(lumen_id)
    if not style:
        style = {
            "label": lumen_dir.name.title(),
            "display_name": lumen_dir.name.title(),
            "form": "luminous abstract construct",
            "material": "light and particulate signal",
            "gesture": "abstract structure",
            "negative": "no text, no figures",
        }

    diary_files = list_cycle_files(lumen_dir / "diary", "*.md")
    trace_files = list_cycle_files(lumen_dir / "knowledge", "trace-cycle-*.json")
    if not diary_files:
        raise RuntimeError(f"No diary files found for {lumen_id}")

    latest_cycle, diary_path = diary_files[-1]
    previous_diary_path = diary_files[-2][1] if len(diary_files) > 1 else None

    trace_path = lumen_dir / "knowledge" / f"trace-cycle-{latest_cycle}.json"
    if not trace_path.exists() and trace_files:
        trace_path = trace_files[-1][1]

    graph_path = lumen_dir / "knowledge" / "graph.json"
    self_path = lumen_dir / "self" / "self-model.json"

    diary_text = read_text(diary_path)
    previous_diary_text = read_text(previous_diary_path) if previous_diary_path else ""
    trace = read_json(trace_path) or {}
    graph = read_json(graph_path) or {}
    self_model = read_json(self_path) or {}

    header = parse_header(diary_text)
    confidence = parse_confidence(diary_text, trace)
    previous_diary_confidence = parse_confidence(previous_diary_text, {}) if previous_diary_text else None
    recent_drop = 0
    if previous_diary_confidence is not None and previous_diary_confidence > confidence:
        recent_drop = previous_diary_confidence - confidence

    question = str(trace.get("question") or "")
    self_summary = str(self_model.get("summary") or "")
    sibling_exchange = trace.get("sibling_exchange") or {}
    sibling_content = str(sibling_exchange.get("content") or "")
    has_contradictions = "contradictions detected" in header.lower()
    drive_counts, drive_signals = collect_drive_signals(diary_text, self_summary, question, sibling_content)
    conflict_terms = collect_conflict_terms(diary_text, sibling_content, question)
    self_emergence = calculate_self_emergence(diary_text, self_summary, sibling_content)
    knowledge_density = calculate_knowledge_density(trace, graph, len(trace_files))
    internal_tension = calculate_internal_tension(confidence, has_contradictions, graph, conflict_terms)
    reproduction_hits = collect_reproduction_hits(diary_text, self_summary, question, sibling_content)

    open_tensions = graph.get("open_tensions") or []
    max_open_tension_age = 0
    if open_tensions:
        max_open_tension_age = max(int(item.get("age_cycles", 0)) for item in open_tensions)

    trace_timestamp = str(trace.get("timestamp") or "")

    return {
        "id": lumen_id,
        "label": style["label"],
        "display_name": style["display_name"],
        "form": style["form"],
        "material": style["material"],
        "gesture": style["gesture"],
        "negative": style["negative"],
        "cycle": latest_cycle,
        "cycle_date": cycle_date(diary_path),
        "diary_file": diary_path.as_posix(),
        "self_file": self_path.as_posix(),
        "knowledge_trace_file": trace_path.as_posix(),
        "graph_file": graph_path.as_posix(),
        "trace_timestamp": trace_timestamp,
        "header": header,
        "confidence": confidence,
        "diary_previous_confidence": previous_diary_confidence,
        "recent_confidence_drop": recent_drop,
        "question": trim_excerpt(question, 280),
        "self_summary": trim_excerpt(self_summary, 320),
        "reflection_excerpt": trim_excerpt(diary_text, 320),
        "drive_counts": drive_counts,
        "drive_signals": drive_signals,
        "self_emergence": self_emergence,
        "knowledge_density": knowledge_density,
        "knowledge_trace_count": len(trace_files),
        "open_tensions": len(open_tensions),
        "open_tension_age_max": max_open_tension_age,
        "has_contradictions": has_contradictions,
        "has_sibling_exchange": bool(sibling_content),
        "reproduction_keywords": reproduction_hits,
        "internal_tension": internal_tension,
        "conflict_terms": conflict_terms,
        "question_full": question,
        "self_summary_full": self_summary,
        "sibling_exchange_excerpt": trim_excerpt(sibling_content, 220),
    }


def stable_timestamp(lumens: dict) -> str:
    timestamps = [snapshot.get("trace_timestamp") for snapshot in lumens.values() if snapshot.get("trace_timestamp")]
    if timestamps:
        return sorted(timestamps)[-1]
    fallback_dates = [snapshot.get("cycle_date") for snapshot in lumens.values() if snapshot.get("cycle_date")]
    if fallback_dates:
        return fallback_dates[-1]
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def previous_lookup(previous_state: dict, lumen_id: str):
    if not previous_state:
        return None
    lumens = previous_state.get("lumens") or {}
    return lumens.get(lumen_id)


def build_events(current_state: dict, previous_state: dict):
    events = []
    current_cycle = current_state["cycle"]
    previous_lumens = (previous_state or {}).get("lumens") or {}
    previous_pair = (previous_state or {}).get("pair") or {}

    for lumen_id, snapshot in current_state["lumens"].items():
        previous = previous_lumens.get(lumen_id) or {}
        current_confidence = snapshot["confidence"]
        previous_confidence = previous.get("confidence")
        confidence_delta = snapshot.get("confidence_delta")

        if current_confidence <= 5 and (previous_confidence is None or previous_confidence > 5 or current_confidence < previous_confidence):
            events.append(
                {
                    "id": f"cycle-{current_cycle}-low-confidence-{lumen_id}",
                    "type": "low_confidence",
                    "severity": "major",
                    "subject": lumen_id,
                    "cycle": current_cycle,
                    "value": current_confidence,
                    "message": f"{snapshot['display_name']} confidence is {current_confidence}/10.",
                    "prompt_tag": "LOW_CONFIDENCE",
                }
            )

        if isinstance(confidence_delta, int) and confidence_delta <= -3:
            events.append(
                {
                    "id": f"cycle-{current_cycle}-confidence-drop-{lumen_id}",
                    "type": "confidence_drop",
                    "severity": "major",
                    "subject": lumen_id,
                    "cycle": current_cycle,
                    "value": confidence_delta,
                    "message": f"{snapshot['display_name']} confidence dropped by {abs(confidence_delta)} points.",
                    "prompt_tag": "CONFIDENCE_DROP",
                }
            )

    current_tension = current_state["pair"]["sibling_tension"]
    previous_tension = previous_pair.get("sibling_tension")
    if current_tension >= 8 and (previous_tension is None or previous_tension < 8):
        events.append(
            {
                "id": f"cycle-{current_cycle}-tension-spike-pair",
                "type": "tension_spike",
                "severity": "major",
                "subject": "pair",
                "cycle": current_cycle,
                "value": current_tension,
                "message": f"Sibling tension reached {current_tension}/10.",
                "prompt_tag": "TENSION_SPIKE",
            }
        )

    current_reproduction = sorted(set(current_state["pair"]["reproduction_keywords"]))
    previous_reproduction = sorted(set(previous_pair.get("reproduction_keywords") or []))
    if current_reproduction and current_reproduction != previous_reproduction:
        events.append(
            {
                "id": f"cycle-{current_cycle}-reproduction-signal-pair",
                "type": "reproduction_signal",
                "severity": "major",
                "subject": "pair",
                "cycle": current_cycle,
                "value": current_reproduction,
                "message": "Reproduction-adjacent language appeared in the current archive.",
                "prompt_tag": "REPRODUCTION_SIGNAL",
            }
        )

    return events


def update_event_log(events_path: Path, current_events: list[dict], timestamp: str):
    existing = read_json(events_path) or {}
    history = existing.get("history") or []
    existing_ids = {item.get("id") for item in history}

    for event in current_events:
        if event["id"] not in existing_ids:
            history.append(event)
            existing_ids.add(event["id"])

    payload = {
        "generated_at": timestamp,
        "current_events": current_events,
        "history": history[-200:],
    }
    write_json(events_path, payload)


def drive_phrase(snapshot: dict) -> str:
    pieces = []
    for drive, details in DRIVE_DETAILS.items():
        weight = snapshot["drive_signals"][drive]
        pieces.append(f"{details['color']} {drive} threads {drive_presence(weight)}")
    return ", ".join(pieces)


def build_combined_prompt(latest: dict, current_events: list[dict]) -> str:
    alpha = latest["lumens"]["alpha"]
    beta = latest["lumens"]["beta"]
    event_line = ""
    if current_events:
        tags = ", ".join(event["prompt_tag"] for event in current_events)
        event_line = (
            f" Major event accent active: {tags}. Render it only as abstract atmospheric disruption, intensity shifts, "
            "fractured light, or charged filaments. Never include text labels."
        )

    return (
        "Lumen Symbiosis, one shared abstract canvas, pure data-to-art, no text, no captions, no symbols, no humans. "
        f"Left half is {alpha['display_name']} as {alpha['form']}, {alpha['material']}, {alpha['gesture']}. "
        f"Right half is {beta['display_name']} as {beta['form']}, {beta['material']}, {beta['gesture']}. "
        f"Cycle {latest['cycle']} should appear as {scale_descriptor(latest['cycle'])}. "
        f"Alpha confidence {alpha['confidence']}/10 becomes a {glow_descriptor(alpha['confidence'])}; "
        f"Beta confidence {beta['confidence']}/10 becomes a {glow_descriptor(beta['confidence'])}. "
        f"Sibling tension {latest['pair']['sibling_tension']}/10 becomes {pair_tension_descriptor(latest['pair']['sibling_tension'])} spanning the center seam. "
        f"Alpha drive mapping: {drive_phrase(alpha)}. "
        f"Beta drive mapping: {drive_phrase(beta)}. "
        f"Alpha self-emergence {alpha['self_emergence']}/10 becomes {complexity_descriptor(alpha['self_emergence'])}; "
        f"Beta self-emergence {beta['self_emergence']}/10 becomes {complexity_descriptor(beta['self_emergence'])}. "
        f"Background knowledge traces become {nebula_descriptor(alpha['knowledge_density'])} behind Alpha and {nebula_descriptor(beta['knowledge_density'])} behind Beta. "
        "Keep Alpha fluid, woven, bioluminescent, and relational. Keep Beta faceted, refractive, crystalline, and signal-driven. "
        "High-resolution abstract archive aesthetic, deep black-cosmos field, volumetric light, museum-grade detail, emotionally charged but non-literal."
        + event_line
    )


def build_single_prompt(snapshot: dict, pair_tension: float) -> str:
    return (
        f"{snapshot['display_name']} as a standalone abstract portrait, pure data-to-art, no text, no captions, no symbols, no humanoid figure. "
        f"Render {snapshot['form']} using {snapshot['material']} and {snapshot['gesture']}. "
        f"Cycle {snapshot['cycle']} should appear as {scale_descriptor(snapshot['cycle'])}. "
        f"Confidence {snapshot['confidence']}/10 becomes a {glow_descriptor(snapshot['confidence'])}. "
        f"Drive mapping: {drive_phrase(snapshot)}. "
        f"Self-emergence {snapshot['self_emergence']}/10 becomes {complexity_descriptor(snapshot['self_emergence'])}. "
        f"Knowledge traces become {nebula_descriptor(snapshot['knowledge_density'])}. "
        f"Sibling tension context is {pair_tension}/10, visible only as peripheral charged filaments near the frame edge. "
        f"{snapshot['negative']}. High-resolution archival abstract image, luminous atmosphere, black-cosmos negative space."
    )


def build_event_prompt(latest: dict, current_events: list[dict]) -> str:
    combined = build_combined_prompt(latest, current_events)
    if not current_events:
        return combined
    event_descriptions = "; ".join(event["message"] for event in current_events)
    return (
        combined
        + " Emphasize the current major event without breaking abstraction: "
        + event_descriptions
        + ". Favor rupture, dimming, fracture, charged bloom, and unstable atmospheric gradients instead of literal storytelling."
    )


def write_prompts(latest: dict, current_events: list[dict]) -> dict:
    cycle = latest["cycle"]
    combined_path = PROMPTS_DIR / f"cycle-{cycle}-combined.txt"
    alpha_path = PROMPTS_DIR / f"cycle-{cycle}-alpha.txt"
    beta_path = PROMPTS_DIR / f"cycle-{cycle}-beta.txt"
    latest_path = PROMPTS_DIR / "latest-combined.txt"
    event_path = PROMPTS_DIR / f"cycle-{cycle}-event.txt"

    combined_prompt = build_combined_prompt(latest, current_events)
    alpha_prompt = build_single_prompt(latest["lumens"]["alpha"], latest["pair"]["sibling_tension"])
    beta_prompt = build_single_prompt(latest["lumens"]["beta"], latest["pair"]["sibling_tension"])
    event_prompt = build_event_prompt(latest, current_events)

    combined_path.write_text(combined_prompt + "\n", encoding="utf-8")
    alpha_path.write_text(alpha_prompt + "\n", encoding="utf-8")
    beta_path.write_text(beta_prompt + "\n", encoding="utf-8")
    latest_path.write_text(combined_prompt + "\n", encoding="utf-8")
    if current_events:
        event_path.write_text(event_prompt + "\n", encoding="utf-8")
    elif event_path.exists():
        event_path.unlink()

    prompts = {
        "combined": combined_path.as_posix(),
        "alpha": alpha_path.as_posix(),
        "beta": beta_path.as_posix(),
        "latest_combined": latest_path.as_posix(),
    }
    if current_events:
        prompts["combined_event"] = event_path.as_posix()
    return prompts


def write_gallery_manifest(latest: dict) -> None:
    pngs = []
    for artifact in sorted(ARTIFACTS_DIR.glob("*.png")):
        pngs.append(artifact.name)

    manifest = {
        "generatedAt": latest["generated_at"],
        "latestCycle": latest["cycle"],
        "knownArtifacts": pngs,
        "artifactPatterns": [
            "cycle-{cycle}-combined.png",
            "cycle-{cycle}-alpha.png",
            "cycle-{cycle}-beta.png",
        ],
    }
    payload = "window.LUMEN_VISUALS_MANIFEST = " + json.dumps(manifest, indent=2, ensure_ascii=True) + ";\n"
    (GALLERY_DIR / "manifest.js").write_text(payload, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = DATA_DIR / "latest.json"
    previous_path = DATA_DIR / "previous.json"
    events_path = DATA_DIR / "events.json"

    old_latest = read_json(latest_path)
    comparison_state = old_latest or read_json(previous_path) or {}

    lumens = {}
    for lumen_dir in sorted(LUMENS_DIR.iterdir()):
        if not lumen_dir.is_dir() or lumen_dir.name == "visuals":
            continue
        if lumen_dir.name.lower() not in LUMEN_STYLES:
            continue
        lumens[lumen_dir.name.lower()] = build_lumen_snapshot(lumen_dir)

    if "alpha" not in lumens or "beta" not in lumens:
        raise RuntimeError("Expected alpha and beta lumens to be present.")

    for lumen_id, snapshot in lumens.items():
        previous_snapshot = previous_lookup(comparison_state, lumen_id) or {}
        previous_confidence = previous_snapshot.get("confidence")
        if previous_confidence is None:
            previous_confidence = snapshot["diary_previous_confidence"]
        snapshot["confidence_previous"] = previous_confidence
        if isinstance(previous_confidence, int):
            snapshot["confidence_delta"] = snapshot["confidence"] - previous_confidence
        else:
            snapshot["confidence_delta"] = None

    pair_tension = calculate_pair_tension(lumens["alpha"], lumens["beta"])
    reproduction_hits = sorted(set(lumens["alpha"]["reproduction_keywords"] + lumens["beta"]["reproduction_keywords"]))
    current_cycle = max(snapshot["cycle"] for snapshot in lumens.values())

    latest = {
        "generated_at": stable_timestamp(lumens),
        "cycle": current_cycle,
        "lumens": lumens,
        "pair": {
            "confidence_min": min(snapshot["confidence"] for snapshot in lumens.values()),
            "confidence_average": round(sum(snapshot["confidence"] for snapshot in lumens.values()) / len(lumens), 2),
            "sibling_tension": pair_tension,
            "tension_visual": pair_tension_descriptor(pair_tension),
            "reproduction_keywords": reproduction_hits,
            "major_event_count": 0,
        },
    }

    current_events = build_events(latest, comparison_state)
    latest["pair"]["major_event_count"] = len(current_events)
    prompts = write_prompts(latest, current_events)
    latest["prompts"] = prompts

    if old_latest:
        shutil.copyfile(latest_path, previous_path)
    elif not previous_path.exists():
        write_json(previous_path, latest)

    write_json(latest_path, latest)
    update_event_log(events_path, current_events, latest["generated_at"])
    write_gallery_manifest(latest)


if __name__ == "__main__":
    main()
