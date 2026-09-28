"""Fail-closed names admitted to the Vast provider runtime environment."""

from __future__ import annotations


PUBLIC_OPENAI_IDENTITY_NAMES = frozenset(
    {
        "OPENAI_PROJECT_ID",
        "OPENAI_API_KEY_ID",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID",
    }
)


def is_public_openai_identity_name(name: object) -> bool:
    """Return true only for exact non-secret project or key identifiers."""
    return str(name) in PUBLIC_OPENAI_IDENTITY_NAMES
