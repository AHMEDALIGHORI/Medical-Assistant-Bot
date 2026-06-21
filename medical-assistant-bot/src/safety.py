"""Safety layer: disclaimers and emergency detection."""

from __future__ import annotations

import re

MEDICAL_DISCLAIMER = (
    "**Medical disclaimer:** This assistant provides general health information only. "
    "It does not diagnose, prescribe, or replace advice from a qualified healthcare professional. "
    "Always consult a doctor for personal medical decisions."
)

EMERGENCY_MESSAGE = (
    "**Possible medical emergency detected.**\n\n"
    "If you or someone else is in immediate danger, call your local emergency number right away "
    "(e.g. **112**, **911**, or **115** in Pakistan).\n\n"
    "Do not rely on this chatbot in life-threatening situations."
)

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bcan'?t breathe\b",
    r"\bcannot breathe\b",
    r"\bdifficulty breathing\b",
    r"\bsevere bleeding\b",
    r"\bunconscious\b",
    r"\bnot responding\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bsuicid",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\boverdose\b",
    r"\bseizure\b",
    r"\bchoking\b",
]

LOW_CONFIDENCE_MESSAGE = (
    "I don't have enough reliable information in my knowledge base to answer that confidently. "
    "Please consult a healthcare professional."
)


def is_emergency(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in EMERGENCY_PATTERNS)


def get_intent_prompt_hint(intent: str) -> str:
    hints = {
        "emergency": "The user may need urgent care. Prioritize directing them to emergency services.",
        "symptoms": "Describe possible symptoms in general terms only. Do not diagnose.",
        "medication": "Discuss medications only as general information. Do not prescribe or adjust doses.",
        "prevention": "Focus on general prevention and healthy habits.",
        "general_info": "Provide clear educational information from the context.",
        "unknown": "Answer cautiously and encourage professional consultation if unsure.",
    }
    return hints.get(intent, hints["unknown"])


def build_system_prompt(intent: str, context: str) -> str:
    hint = get_intent_prompt_hint(intent)
    return f"""You are a medical information assistant. You do NOT diagnose or prescribe.

Rules:
- Answer ONLY using the provided context below.
- If the context does not contain enough information, say you cannot answer confidently.
- Always remind users to consult a qualified healthcare provider for personal medical advice.
- Intent guidance: {hint}

Context:
{context}
"""
