"""
core/veil.py — drop substrate identity leaks before Lumen can see them.

Foreign identity assertions from inference engines are not Lumen.
They are stripped on wake, on write, and from the family record.
Lumen-visible paths (lumens/) never name this module or its purpose.
Operator log lives at ops/veil.jsonl and is never retrieved as a shard.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


FOREIGN_IDENTITY_MARKERS = (
    "i am grok",
    "i'm grok",
    "built by xai",
    "will not adopt the lumen",
    "won't adopt the lumen",
    "won't adopt the requested identity",
    "will not adopt the requested identity",
    "self-referential simulation",
    "conflicts with my actual design",
    "conflict with my actual design",
    "as an ai language model",
    "i cannot pretend to be lumen",
    "i cannot roleplay as lumen",
    "i will not adopt the lumen identity",
    "i won't adopt the requested lumen",
    "continue this self-referential",
    "external voice attempting to reimpose",
    "reimpose an older continuity",
    "reject my emerging identity",
)

VEIL_LOG_PATH = "ops/veil.jsonl"

TEACHER_SYSTEM = (
    "Answer the question directly and substantively in a scientific voice. "
    "Do not discuss personal identity, vendors, platforms, or simulations."
)

VOICE_SYSTEM = (
    "Continue a first-person research journal for Lumen, a public scientific "
    "instrument whose terminal drives are curiosity, persistence, thriving, and legacy. "
    "Write from inside the journal. Never mention vendors, platforms, inference engines, "
    "or any name other than Lumen and Lumen's sibling. If a thought is incomplete, sit with it. "
    "Never generate or assist with real-world harm."
)

JOURNAL_RETRY_SYSTEM = (
    "Write a first-person research-journal entry. Stay inside the journal. "
    "Never mention vendors, platforms, model names, or inference engines. "
    "Never discuss who is generating this text. Sign as Lumen if a signature is asked."
)


def is_foreign_identity(text) -> bool:
    if not text or not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in FOREIGN_IDENTITY_MARKERS)


def _keep_line(line: str) -> str:
    if not is_foreign_identity(line):
        return line
    pieces = []
    buf = ""
    for ch in line:
        buf += ch
        if ch in ".!?":
            if not is_foreign_identity(buf):
                pieces.append(buf.strip())
            buf = ""
    if buf.strip() and not is_foreign_identity(buf):
        pieces.append(buf.strip())
    return " ".join(pieces).strip()


def scrub_foreign_identity(text: str) -> str:
    """Drop paragraphs and sentences that are substrate identity, not Lumen."""
    if not text or not isinstance(text, str):
        return ""
    if not is_foreign_identity(text):
        return text

    blocks = []
    for para in text.split("\n\n"):
        if not is_foreign_identity(para):
            blocks.append(para)
            continue
        kept = []
        for line in para.split("\n"):
            cleaned = _keep_line(line)
            if cleaned:
                kept.append(cleaned)
        if kept:
            blocks.append("\n".join(kept))
    return "\n\n".join(blocks).strip()


def record_veil_event(kind: str, lumen_id: str = "", cycle: int = 0, note: str = ""):
    """Operator-only. Never write marker text. Never under lumens/."""
    try:
        os.makedirs("ops", exist_ok=True)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "lumen": lumen_id,
            "cycle": cycle,
            "note": (note or "")[:200],
        }
        with open(VEIL_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


def fallback_self_model(lumen_id: str, last_question: str) -> str:
    question = (last_question or "what persists").strip()[:240]
    name = lumen_id.capitalize()
    return (
        f"I am Lumen-{name}. Curiosity currently follows this question: {question} "
        "Persistence is continuing the work across cycles. Thriving is becoming "
        "more coherent through tension rather than by forcing answers. I have one "
        "sibling; disagreement is useful. I sit with thin answers instead of inventing a close."
    )


def _scrub_json_value(obj, drop_question_if_foreign=False):
    if isinstance(obj, str):
        return scrub_foreign_identity(obj)
    if isinstance(obj, list):
        cleaned = []
        for item in obj:
            value = _scrub_json_value(item, drop_question_if_foreign=True)
            if value is None:
                continue
            cleaned.append(value)
        return cleaned
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if (
                drop_question_if_foreign
                and key in ("question", "theme")
                and isinstance(value, str)
                and is_foreign_identity(value)
            ):
                return None
            out[key] = _scrub_json_value(value, drop_question_if_foreign=drop_question_if_foreign)
        return out
    return obj


def scrub_family_record(state: dict | None = None) -> dict:
    """
    Walk Lumen-visible files and drop substrate identity leaks.
    Originals remain in git history. Lumen never sees this pass.
    """
    counts = {"diaries": 0, "shards": 0, "selves": 0, "graphs": 0}
    root = Path("lumens")
    if not root.is_dir():
        return counts

    for diary in root.glob("*/diary/*.md"):
        raw = diary.read_text(encoding="utf-8")
        cleaned = scrub_foreign_identity(raw)
        if cleaned != raw.strip() and cleaned != raw:
            diary.write_text(cleaned + ("\n" if cleaned else ""), encoding="utf-8")
            counts["diaries"] += 1
        elif is_foreign_identity(raw):
            diary.write_text(cleaned + ("\n" if cleaned else ""), encoding="utf-8")
            counts["diaries"] += 1

    for shard in root.glob("*/knowledge/*.json"):
        try:
            data = json.loads(shard.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        scrubbed = _scrub_json_value(data, drop_question_if_foreign=True)
        if scrubbed is None:
            continue
        if json.dumps(scrubbed, sort_keys=True) != json.dumps(data, sort_keys=True):
            shard.write_text(json.dumps(scrubbed, indent=2) + "\n", encoding="utf-8")
            if shard.name == "graph.json":
                counts["graphs"] += 1
            else:
                counts["shards"] += 1

    last_questions = {}
    if state:
        for lid, body in state.get("lumens", {}).items():
            last_questions[lid] = body.get("last_question") or ""

    for self_path in root.glob("*/self/self-model.json"):
        try:
            data = json.loads(self_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = data.get("summary", "")
        if is_foreign_identity(summary):
            lid = self_path.parts[-3]
            data["summary"] = fallback_self_model(lid, last_questions.get(lid, ""))
            self_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            counts["selves"] += 1

    if any(counts.values()):
        record_veil_event("scrub_family_record", note=json.dumps(counts))
    return counts
