"""Synthetic evidence for software tests only; never a real lead assessment."""
from datetime import timedelta

from tools.daily_research import verification


def assessment(candidate, now):
    return {"version": verification.VERSION, "candidate_digest": verification.digest(candidate),
            "assessed_at": now.isoformat(), "valid_until": (now + timedelta(days=7)).isoformat(),
            "claims": {name: {"status": "inference" if name == "plausible_fit" else "verified_fact",
                               "reason": "Synthetic fixture explicitly links this operator, physical site and human task; fit is a hypothesis",
                               "source_refs": ["synthetic-primary"]} for name in verification.CLAIMS},
            "sources": [{"id": "synthetic-primary", "url": "https://fixture.example/site-task",
                         "publisher": "Synthetic operator", "source_date": None, "event_date": None,
                         "checked_at": now.isoformat(), "retrieval": "static", "classification": "operator",
                         "quote": "Synthetic operator runs this exact physical site where humans perform this task.",
                         "freshness": "current", "freshness_reason": "Synthetic current-state fixture, not live proof"}],
            "counterevidence": {"status": "checked", "reason": "Synthetic bounded automation check; no physical/robot proof",
                                "source_refs": [], "searches": ["Synthetic task automation and incumbent-system search"]}}
