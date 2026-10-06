"""Synthetic contact-free qualified team export and exact pin. No live reads or credentials."""
from datetime import date

from tools.daily_research import team_universe as te

TODAY = date(2026, 10, 5)


def export(count=3):
    rows = []
    for i in range(count):
        key, domain = te.sha(str(i).encode()), f"synthetic-{i}.example"
        source = {"team_key": key, "domain": domain, "discovery_sha256": "b" * 64, "run_id": f"synthetic-run-{i}",
                  "result_sha256": "c" * 64, "evidence_sha256": "d" * 64}
        positive = i % 3 == 0
        quote = f"Synthetic Robotics {i} deploys its fixture arm robot for stacking rigid cases at warehouses."
        row = {"team_key": key, "domain": domain, "company": f"Synthetic Robotics {i}",
               "status": "capability_prospect" if positive else "reference_only" if i % 3 == 1 else "pending",
               "qualification": {"rule_version": te.RULE, "offering": "physical_robot_task" if positive else "reference_only" if i % 3 == 1 else "unknown",
                   "current_task_fit": positive, "evaluation_compatibility": "not_verified",
                   "robot_forms": ["fixed_arm"] if positive else [], "task_families": ["palletizing_depalletizing"] if positive else [],
                   "partner_intent": {"state": "unknown", "kind": "none"}, "willingness_to_work_with_blueprint": "unknown",
                   "relationship_implied": False, "promotion_allowed": False, "blockers": ["evaluation_compatibility_not_verified"]},
               "assessment": {"reason": "Synthetic reviewed current physical offering." if positive else "Held synthetic unresolved or adjacent evidence.",
                   "identified_hardware": "fixture arm robot" if positive else None, "physical_task": "stacking rigid cases" if positive else None,
                   "offering_relation": "deploys its fixture arm robot" if positive else None,
                   "as_of": TODAY.isoformat() if positive else None, "current_basis": "current_offering_page" if positive else None,
                   "proofs": [{"field": "robot_forms", "url": f"https://{domain}/product", "quote": quote,
                       "level": "verified_on_page", "quote_sha256": te.sha(quote.encode()), "page_sha256": "e" * 64}] if positive else []},
               "invitation_evidence": [], "source_binding": source, "screen_checked_on": TODAY.isoformat()}
        rows.append(row)
    rows.sort(key=lambda row: row["team_key"])
    return repack({"schema_version": te.EXPORT, "manifest": {"ranked_sha256": "1" * 64, "audit_sha256": "2" * 64,
                  "scope_sha256": "3" * 64, "assessed_on": TODAY.isoformat(), "rules": te.RULES,
                  "rows_sha256": "4" * 64, "counts": {}, "distribution": "internal_only"}, "teams": rows})


def repack(value):
    value["manifest"]["rows_sha256"] = te.sha(te.encoded(value["teams"]))
    value["manifest"]["scope_sha256"] = te.sha(te.encoded([r["source_binding"] for r in value["teams"]]))
    value["manifest"]["counts"] = {name: sum(r["status"] == name for r in value["teams"]) for name in sorted(te.STATUSES)}
    return te.encoded(value)


def pin(raw, generation="1001", version=1):
    value = te.load(raw, today=TODAY)
    return {"schema_version": te.PIN, "enabled": True, "version": version, "uri": te.object_uri(te.sha(raw)),
            "generation": generation, "sha256": te.sha(raw), "bytes": len(raw), "approval_reference": "synthetic-owner-agent-evidence",
            **{k: value["manifest"][k] for k in ("ranked_sha256", "audit_sha256", "scope_sha256", "assessed_on")}}
