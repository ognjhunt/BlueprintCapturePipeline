"""Contact-free, offline projection of the current reviewed rank and retained qualification audit."""
from datetime import datetime, timezone

from tools.daily_research import site_screen as ss
from tools.daily_research import team_universe as evidence
from tools.team_universe import qualification as tq
from tools.team_universe import rank
from tools.team_universe import universe as tu


def build(workspace, audit_raw, *, today=None, config_raw=None):
    """Revalidate every source/audit binding and qualification before projecting. Never reads a page/provider."""
    today = today or datetime.now(timezone.utc).date()
    config = rank.load_config(config_raw)
    with workspace.lock():
        states, _ = workspace.states()
        teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
        screens = tu.stage_records(workspace, states, "screen")
        audit = tq.load_audit(audit_raw, teams, screens, team_list_raw=workspace.teams_path().read_bytes(), today=today)
        raw = (workspace.root / tu.RANKED_NAME).read_bytes()
        ranked = ss._json(raw)
        scope = tq.scope_manifest(teams, screens)
        if (ranked.get("schema_version") != rank.RANKED or ranked.get("assessed_on") != today.isoformat()
                or ranked.get("scope_sha256") != scope["sha256"] or ranked.get("rules") != evidence.RULES
                or ranked.get("audit", {}).get("sha256") != audit["sha256"] or audit["sha256"] is None
                or ranked.get("config_sha256") != config["sha256"]):
            raise tu.TeamError("team_universe_export_snapshot_mismatch")
        listed = {team["site_key"]: team for team in teams}
        rows = {row["team_key"]: row for row in ranked["teams"]}
        if len(rows) != len(ranked["teams"]) or set(rows) != {item["team_key"] for item in scope["bindings"]}:
            raise tu.TeamError("team_universe_export_snapshot_mismatch")
        output = []
        values = ranked["weights"]["values"]
        maximum = max(values.values(), default=0)
        if not maximum > 0:
            raise tu.TeamError("team_universe_family_weights_invalid")
        weights = {"weights": values, "normalized": {family: values[family] / maximum for family in tu.TASK_FAMILIES}}
        for binding in scope["bindings"]:
            key = binding["team_key"]
            record, reviewed = screens.get(key), audit["rows"][key]
            retained = tu._evidence(workspace.path("screen", "evidence", key).read_bytes()) if record else {}
            qualified = tq.qualify(record, reviewed, retained, today=today)
            expected = rank.ranked_row(listed.get(key), record, config, weights, qualified)
            if ss.canonical(rows[key]) != ss.canonical(expected):
                raise tu.TeamError("team_universe_export_rank_binding_mismatch")
            details = {"reason": reviewed["capability_reason"], "identified_hardware": reviewed.get("identified_hardware"),
                       "physical_task": reviewed.get("physical_task"), "offering_relation": reviewed.get("offering_relation"),
                       "as_of": reviewed.get("capability_as_of"), "current_basis": reviewed.get("capability_current_basis"),
                       "proofs": reviewed["capability_evidence"] if rows[key]["status"] == "capability_prospect" else []}
            projected = {k: qualified[k] for k in evidence.QUAL_FIELDS}
            invitations = reviewed["published_partner_intent"]["evidence"] if qualified["partner_intent"]["state"] == "explicit_invitation" else []
            if any(p["field"] == "contact" for p in invitations):
                # A contact-free projection cannot substantiate a contact-page invitation.
                projected["partner_intent"] = {"state": "unknown", "kind": "none"}
                projected["blockers"] = [*projected["blockers"], "partner_invitation_not_exported"]
                invitations = []
            output.append({"team_key": key, "domain": binding["domain"], "company": rows[key]["company"] or binding["domain"],
                           "status": rows[key]["status"], "qualification": projected, "invitation_evidence": invitations,
                           "assessment": details, "source_binding": binding, "screen_checked_on": record["checked_on"] if record else None})
        document = {"schema_version": evidence.EXPORT, "manifest": {"ranked_sha256": ss._sha256(raw), "audit_sha256": audit["sha256"],
                    "scope_sha256": scope["sha256"], "assessed_on": today.isoformat(), "rules": evidence.RULES,
                    "rows_sha256": evidence.sha(evidence.encoded(output)), "counts": {name: sum(r["status"] == name for r in output) for name in sorted(evidence.STATUSES)},
                    "distribution": "internal_only"}, "teams": output}
        result = evidence.encoded(document)
        evidence.load(result, today=today)
        return result
