"""Hermetic host-owned site-screen admission (design section 3).

Synthetic out dirs from the site-screen fakes, the real private bridge and Store with in-memory Firestore, a fake
object store and a fake CRM sheet behind the real Publisher. Every operator, site, person and address is synthetic;
hosts use the reserved .example domain.
"""
import base64
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests import daily_research_site_screen_fixture as fixture
from tests.daily_research_site_screen_fixture import (
    KEY,
    OWNER,
    PERSON,
    SITE_STRINGS,
    TODAY,
    FakePages,
    contact_answers,
    pages_for,
    screen,
    screen_answers,
)
from tools.daily_research import outreach_ready, runner
from tools.daily_research import screen_admission as sa
from tools.daily_research import site_screen as ss
from tools.daily_research.firestore import Bridge
from tools.daily_research.runner import Refusal, canonical

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("outreach_ready_direction",
                                              ROOT / "tools/daily_research/operators/outreach-ready-direction.py")
direction_operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direction_operator)
site_spec = importlib.util.spec_from_file_location("site_screen_operator", ROOT / "tools/daily_research/operators/site-screen.py")
site_operator = importlib.util.module_from_spec(site_spec)
site_spec.loader.exec_module(site_operator)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)  # 10:00 America/Chicago, after the daily run.
DIRECTION_SHA, DIRECTION_GENERATION = "d" * 64, "1700000000000001"
APPROVAL = "owner-decision-screen-to-crm-synthetic"
HEADERS = ["Prospect ID", "Organization", "Prospect type", "Site / team", "Contact name", "Contact details",
           "Verification", "Contact source URL", "Robot-team fit", "Task evidence URL", "Stage", "Owner", "Next action",
           "Next action date", "Task / job", "Robot capability evidence URL", "Evidence maturity", "Geography",
           "Evidence checked date"]
NOBODY = {"person_name": "", "person_title": "", "person_url": "", "person_quote": "", "person_date": "", "email": "",
          "email_url": "", "email_quote": "", "channel_type": "none"}
TEAM = {"person_name": "", "person_title": "", "person_url": "", "person_quote": "", "person_date": "",
        "channel_type": "team_inbox"}
FOCUS_A, FOCUS_B = "fixed_arm_machine_tending", "kitting_assembly"
FOCUS_C, FOCUS_D = "palletizing_depalletizing", "sorting_pick_and_place"


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", (), raising=False)
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(sa.contact_lookup, "utc_today", lambda: TODAY)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"},
                        raising=False)


def team(number):
    mail = f"plant.team@operator-{number}.example"
    return {**TEAM, "email": mail, "email_url": f"https://operator-{number}.example/contact",
            "email_quote": f"Write to {mail} for plant questions any business day."}


def general(number):
    mail = f"info@operator-{number}.example"
    return {**TEAM, "email": mail, "email_url": f"https://operator-{number}.example/contact",
            "email_quote": f"General questions go to {mail} on any business day.", "channel_type": "general_inbox"}


def prepared(tmp_path, sites):
    """An out dir with one screened site universe row per (number, focus, contact changes or None[, screen changes])
    and one contact run for each outreach-ready site; None publishes no contact. Returns the workspace and the site
    keys in order."""
    records = [fixture.universe_row(site[0], screen_focus=site[1], capabilities=[site[1]]) for site in sites]
    answers = [screen_answers(site[0], **(site[3] if len(site) > 3 else {})) for site in sites]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    workspace, provider, _, _ = screen(tmp_path, records, answers, pages)
    keys = [record["site_id"] for record in records]
    for (number, _, changes, *_), key in zip(sites, keys):
        content = contact_answers(number, **(NOBODY if changes is None else changes))
        provider.contacts[key] = {"content": content, "basis": []}
        pages.update(pages_for(content))
    client = ss.TaskClient(KEY, transport=provider)
    ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="2", max_runs=100, apply=True)
    reader = FakePages(pages)
    ss.collect(workspace, client=client, reader=reader, today=TODAY, wait_seconds=0)
    ss.verify(workspace, reader=reader, today=TODAY)
    return workspace, keys


def built(workspace, **options):
    if "lookups" not in options:
        def read_synthetic(key):
            path = workspace.root / "synthetic-lookups" / (key + ".json")
            return json.loads(path.read_text()) if path.exists() else None
        options["lookups"] = read_synthetic
    return sa.build(workspace, direction_sha256=options.pop("sha", DIRECTION_SHA),
                    direction_generation=options.pop("generation", DIRECTION_GENERATION), **options)


def named(value):
    text = canonical(value)
    return [name for name in (*SITE_STRINGS, "@", "placeholder") if name in text]


# --- build, selection and the loader ------------------------------------------------------------------------------
def test_admit_takes_two_per_task_family_preferring_verified_recipients_in_the_owner_order(tmp_path):
    workspace, keys = prepared(tmp_path, [(1, FOCUS_A, None), (2, FOCUS_A, team(2)), (3, FOCUS_A, {}),
                                          (4, FOCUS_B, general(4)), (5, FOCUS_B, None), (6, FOCUS_B, None)])
    bundle, raw, report = built(workspace)
    # Family A: the person email (3) and the team inbox (2) beat the site without a contact (1). Family B: the
    # general inbox (4), then the first site without one (5). Results keep screen order.
    assert [result["site_key"] for result in bundle["results"]] == [keys[1], keys[2], keys[3], keys[4]]
    assert [(result["recipient"] or {}).get("route") for result in bundle["results"]] == [
        "published_team_inbox", "published_person_email", "published_general_inbox", None]
    assert report["by_route"] == {"published_team_inbox": 1, "published_person_email": 1, "published_general_inbox": 1,
                                  "none": 1}
    assert report["by_focus"] == {FOCUS_A: 2, FOCUS_B: 2} and report["eligible"] == 6 and report["records"] == 4
    assert report["admission_id"] == hashlib.sha256(raw).hexdigest() and raw == canonical(bundle).encode()
    assert report["state"] == "planned" and report["object_writes"] == 0 and report["sends_authorized"] is False
    assert named(report) == []
    assert sa.load_bundle(raw, report["admission_id"]) == bundle
    # Deterministic: the same out dir and options give the same admission.
    assert built(workspace)[1] == raw


def test_each_result_binds_its_digest_run_quotes_question_and_recipient_provenance(tmp_path):
    workspace, keys = prepared(tmp_path, [(1, FOCUS_A, {})])
    (result,) = built(workspace)[0]["results"]
    (record,) = workspace.records("screen")
    assert result["result_digest"] == runner.digest({"input": result["input"], "answers": result["answers"],
                                                     "checks": result["checks"]})
    assert (result["run_id"], result["result_sha256"]) == (record["run_id"], record["result_sha256"])
    assert result["checks"]["question"] == record["question"] == result["hypothesis"]["question"]
    assert result["hypothesis"] == {"tier": "outreach_ready", "label": "hypothesis", "rule_version": ss.SCREEN_RULE,
                                    "outreach_rule_version": sa.verification.OUTREACH_RULE_VERSION,
                                    "open_checks": ["freshness", "existing_automation", "fit", "interest"],
                                    "question_template": record["question_template"], "question": record["question"]}
    assert [proof["claim"] for proof in result["proofs"]] == ["operator", "physical_site", "site_task"]
    assert result["proofs"][1] == {"claim": "physical_site", "source_id": "government_record", "level": "government_record",
                                   "source_ids": ["epa_frs", "osha_ita"], "site_id": keys[0]}
    for proof in (result["proofs"][0], result["proofs"][2]):
        assert proof["quote_sha256"] == hashlib.sha256(proof["quote"].encode()).hexdigest()
        assert proof["level"] == "verified_on_page" and sa.HEX.fullmatch(proof["text_sha256"])
    assert result["candidate"] == {"organization": "Synthetic Operator 1", "site": "Synthetic Works 1",
                                   "location": "1 Example Road, Fixture City, TX 00001", "task": "CNC machine tending",
                                   "task_url": "https://operator-1.example/careers/lathe", "checked_on": "2026-10-05"}
    recipient = result["recipient"]
    assert recipient["route"] == "published_person_email" and recipient["rank"] == 1
    assert recipient["address"] == "avery.placeholder@operator-1.example" and recipient["address_source"] == "published"
    assert recipient["label"] == "published person email" and recipient["provider"] is None
    assert recipient["operator_domain"] == {"domain": "operator-1.example", "basis": "operator_quote",
                                            "url": "https://operator-1.example/about",
                                            "text_sha256": result["proofs"][0]["text_sha256"]}
    assert recipient["published"]["url"] == "https://operator-1.example/team"
    assert recipient["person"]["name"] == PERSON and recipient["person"]["source"] == "public_quote"
    assert result["contact"]["rule_version"] == "blueprint.site-contact-rule.v3"


def test_a_record_whose_stored_values_differ_is_refused(tmp_path):
    workspace, keys = prepared(tmp_path, [(1, FOCUS_A, {}), (2, FOCUS_A, {})])
    path = workspace.record_path("screen", keys[0])
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "question": "Is this edited by hand?"}, indent=1, sort_keys=True) + "\n")
    bundle, _, report = built(workspace)
    assert [result["site_key"] for result in bundle["results"]] == [keys[1]]
    assert report["refused"] == {"screen_admission_record_mismatch": 1}
    with pytest.raises(sa.AdmissionError, match="screen_admission_record_mismatch"):
        built(workspace, keys=[keys[0]])
    contact = workspace.record_path("contact", keys[1])
    contact.write_text(contact.read_text().replace("person_email", "team_inbox"))
    with pytest.raises(sa.AdmissionError, match="screen_admission_contact_record_mismatch"):
        built(workspace, keys=[keys[1]])


def test_explicit_keys_take_exactly_those_admissible_sites(tmp_path):
    workspace, keys = prepared(tmp_path, [(1, FOCUS_A, None), (2, FOCUS_A, {}), (3, FOCUS_A, None)])
    bundle, _, _ = built(workspace, keys=[keys[2], keys[0]])
    assert [result["site_key"] for result in bundle["results"]] == [keys[0], keys[2]]
    assert bundle["manifest"]["selection"] == {"per_focus": None, "keys": sorted([keys[0], keys[2]]), "max_records": 50}
    with pytest.raises(sa.AdmissionError, match="screen_admission_key_not_admissible"):
        built(workspace, keys=["e" * 64])
    with pytest.raises(sa.AdmissionError, match="screen_admission_batch_above_limit"):
        built(workspace, per_focus=3, max_records=2)
    with pytest.raises(Refusal, match="screen_admission_keys_invalid"):
        sa.parse_keys(f"{keys[0]},{keys[0]}")


def test_the_loader_refuses_a_tampered_bundle(tmp_path):
    workspace, _ = prepared(tmp_path, [(1, FOCUS_A, team(1))])
    bundle, raw, report = built(workspace)
    admission = report["admission_id"]
    flipped = raw.replace(b"Synthetic Works 1", b"Synthetic Works 9", 1)
    with pytest.raises(sa.AdmissionError, match="screen_admission_digest_mismatch"):
        sa.load_bundle(flipped, admission)
    with pytest.raises(sa.AdmissionError, match="screen_admission_bundle_not_canonical"):
        sa.load_bundle(json.dumps(bundle, indent=1).encode())

    def changed(edit):
        value = json.loads(raw)
        edit(value)
        return canonical(value).encode()

    def answers(value):
        value["results"][0]["answers"]["target_task"] = "Edited task"

    def candidate(value):
        value["results"][0]["candidate"]["organization"] = "Edited Operator"

    def rules(value):
        value["manifest"]["rules"]["contact"] = "blueprint.site-contact-rule.v2"

    def free_mail(value):
        value["results"][0]["recipient"]["address"] = "plant.team@gmail.com"

    def sends(value):
        value["manifest"]["sends_authorized"] = True

    for edit, code in ((answers, "screen_admission_result_digest_mismatch"), (rules, "screen_admission_rule_mismatch"),
                       (candidate, "screen_admission_result_invalid"), (free_mail, "screen_admission_recipient_invalid"),
                       (sends, "screen_admission_bundle_invalid")):
        with pytest.raises(sa.AdmissionError, match=code):
            sa.load_bundle(changed(edit))
    # Any edit also changes the bundle's SHA-256, which is the admission id the pin and the WebApp check.
    assert hashlib.sha256(changed(candidate)).hexdigest() != admission


class FakeObjects:
    """The owner's object store: create-only by name, generations, and a byte readback."""

    def __init__(self):
        self.objects, self.generation, self.creates = {}, 1700000000000100, 0

    def create(self, admission_id, path):
        self.creates += 1
        if admission_id not in self.objects:
            self.generation += 1
            self.objects[admission_id] = (str(self.generation), Path(path).read_bytes())
        generation, raw = self.objects[admission_id]
        return generation, len(raw)

    def read(self, admission_id, generation):
        stored, raw = self.objects[admission_id]
        assert generation == stored
        return raw


def test_admit_apply_keeps_the_bundle_write_once_uploads_create_only_and_reads_it_back(tmp_path):
    workspace, _ = prepared(tmp_path, [(1, FOCUS_A, {})])
    objects = FakeObjects()
    first = sa.admit(workspace, direction_sha256=DIRECTION_SHA, direction_generation=DIRECTION_GENERATION, apply=True,
                     objects=objects)
    assert first["state"] == "uploaded" and first["readback_verified"] is True and first["object_writes"] == 1
    kept = workspace.root / "admissions" / f"{first['admission_id']}.json"
    assert hashlib.sha256(kept.read_bytes()).hexdigest() == first["admission_id"]
    again = sa.admit(workspace, direction_sha256=DIRECTION_SHA, direction_generation=DIRECTION_GENERATION, apply=True,
                     objects=objects)
    assert again["generation"] == first["generation"] and len(objects.objects) == 1
    objects.objects[first["admission_id"]] = (first["generation"], b"{}")
    with pytest.raises(sa.AdmissionError, match="screen_admission_readback_failed"):
        sa.admit(workspace, direction_sha256=DIRECTION_SHA, direction_generation=DIRECTION_GENERATION, apply=True,
                 objects=objects)
    assert named(first) == []


def test_gcloud_upload_is_create_only_and_reads_back_the_stored_generation(tmp_path):
    calls, raw = [], b'{"synthetic":true}'
    admission = hashlib.sha256(raw).hexdigest()

    class Done:
        def __init__(self, returncode=0, stdout=""):
            self.returncode, self.stdout = returncode, stdout

    def run(command, **options):
        calls.append(command)
        if command[2] == "cp":
            return Done(1, "")  # Already there: create-only keeps it, and the caller reads it back.
        if command[2] == "objects":
            return Done(0, json.dumps({"generation": 1700000000000777, "size": len(raw)}))
        return Done(0, raw)

    objects = sa.GcloudObjects(run=run)
    path = tmp_path / "bundle.json"
    path.write_bytes(raw)
    assert objects.create(admission, path) == ("1700000000000777", len(raw))
    assert objects.read(admission, "1700000000000777") == raw
    assert calls[0] == ["gcloud", "storage", "cp", "--if-generation-match=0", "--content-type=application/json",
                        str(path), sa.uri(admission)]
    assert calls[2] == ["gcloud", "storage", "cat", sa.uri(admission) + "#1700000000000777"]
    assert sa.uri(admission) == ("gs://blueprint-8c1ca.appspot.com/operations/research/screen-admission/"
                                 f"{admission}/bundle.json")


def lookup(workspace, key, **changes):
    value = {"schema_version": "blueprint.site-contact-lookup.v1", "site_key": key, "address": "avery.placeholder@operator-1.example",
             "provider": {"name": "fullenrich", "status": "valid", "score": 96,
                          "checked_at": "2026-10-05T12:00:00Z", "request_digest": "c" * 64},
             "person": {"source": "public_quote", "name": PERSON, "title": "Plant Manager"}}
    value.update(changes)
    person = value["person"]
    provider = value["provider"]
    current = {"name": person.get("name"), "title": person.get("title"), "sourcing":
               "quoted_person" if person.get("source") == "public_quote" else "provider_sourced",
               "proof": {"source": "fullenrich_people_search", "request_digest": "d" * 64,
               "current_employment": {"field": "employment.current.is_current",
               "company_domain": value["address"].rpartition("@")[2], "start_at": None}}, "corroboration": {"corroborated": person.get("corroborated", False),
               **(person.get("corroboration") or {})}}
    if "site_responsibility" in person:
        current["site_responsibility"] = person["site_responsibility"]
    item = {"source": "provider_lookup", "usable": True, "address": value["address"], "person": current,
            "operator_domain": value["address"].rpartition("@")[2], "provider": {**provider,
            "status": "DELIVERABLE" if provider.get("status") == "valid" else "CATCH_ALL",
            "verification": provider.get("status"), "score": None}}
    value = {"schema_version": "blueprint.contact-lookup.v1" if value["schema_version"] == "blueprint.site-contact-lookup.v1" else "invalid",
             "rule_version": "blueprint.contact-lookup-rule.v2", "site_key": key, "lookups": [item]}
    folder = workspace.root / "synthetic-lookups"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.json").write_text(json.dumps(value))


def test_final_lookup_adds_provider_routes_only_with_full_provenance(tmp_path):
    # The contact stage verified the person but published only a team inbox.
    person_and_team = {**team(1), "person_name": PERSON, "person_title": "Plant Manager",
                       "person_url": "https://operator-1.example/team",
                       "person_quote": f"{PERSON} leads the machining plant as plant manager.", "person_date": "2026-06-01"}
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, person_and_team)])
    assert built(workspace)[0]["results"][0]["recipient"]["route"] == "published_team_inbox"
    lookup(workspace, key)
    recipient = built(workspace)[0]["results"][0]["recipient"]
    assert (recipient["route"], recipient["rank"], recipient["address_source"]) == (
        "quoted_person_looked_up_email", 2, "provider_lookup")
    assert recipient["label"] == "looked-up email, quoted person" and recipient["published"] is None
    assert recipient["provider"]["status"] == "valid" and recipient["person"]["url"] == "https://operator-1.example/team"
    assert recipient["person"]["quote"] == f"{PERSON} leads the machining plant as plant manager."
    assert sa.HEX.fullmatch(recipient["person"]["text_sha256"]) and recipient["person"]["source"] == "public_quote"
    corroboration = {"url": "https://operator-1.example/news", "quote": "Jordan Fixture is the Operations Lead who runs the plant.", "level": "verified_on_page",
                     "text_sha256": "a" * 64}
    for person, route in (({"source": "provider_sourced", "name": "Jordan Fixture", "title": "Operations Lead",
                            "corroborated": True, "corroboration": corroboration}, "provider_sourced_corroborated"),
                          ({"source": "provider_sourced", "name": "Jordan Fixture", "title": "Operations Lead",
                            "corroborated": False}, "provider_sourced_uncorroborated")):
        lookup(workspace, key, address="jordan.fixture@operator-1.example", person=person)
        assert built(workspace)[0]["results"][0]["recipient"]["route"] == route
    # Anything short of full provenance falls back to the published team inbox.
    for changes in ({"provider": {"name": "fullenrich", "status": "accept_all", "score": 96,
                                  "checked_at": "2026-10-05T12:00:00Z", "request_digest": "c" * 64}},
                    {"address": "sales@operator-1.example"}, {"address": "avery.placeholder@gmail.com"},
                    {"address": "avery.placeholder@another-operator.example"},
                    {"person": {"source": "public_quote", "name": "Someone Else", "title": "Plant Manager"}},
                    {"person": {"source": "provider_sourced", "name": "Jordan Fixture", "title": "Lead", "corroborated": True}},
                    {"schema_version": "blueprint.site-contact-lookup.v0"}):
        lookup(workspace, key, **changes)
        assert built(workspace)[0]["results"][0]["recipient"]["route"] == "published_team_inbox", changes


def test_unknown_site_responsibility_is_a_referral_in_bundle_and_sheet_question(tmp_path):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, None)])
    lookup(workspace, key, address="jordan.fixture@operator-1.example", person={
        "source": "provider_sourced", "name": "Jordan Fixture", "title": "Plant Manager", "corroborated": False})
    bundle, raw, _ = built(workspace)
    entry = bundle["results"][0]
    recipient = entry["recipient"]
    assert recipient["route"] == "provider_sourced_uncorroborated"
    assert recipient["provider"]["verification_status"] == "DELIVERABLE"
    assert recipient["person"]["site_responsibility"]["status"] == "unknown"
    assert entry["hypothesis"]["question"].startswith("Could you direct me to the person responsible for ")
    assert entry["candidate"]["site"] in entry["hypothesis"]["question"]
    assert sa.load_bundle(raw) == bundle
    assert "corporate referral, target-site responsibility unknown" in sa.contact_cells(recipient)["details"]
    changed = json.loads(raw)
    changed["results"][0]["hypothesis"]["question"] = entry["checks"]["question"]
    with pytest.raises(sa.AdmissionError, match="screen_admission_question_mismatch"):
        sa.load_bundle(canonical(changed).encode())
    changed = json.loads(json.dumps(recipient))
    changed["person"]["site_responsibility"] = {
        "site_key": key, "status": "verified", "route": "site_contact", "reason": None,
        "proof": {"url": "https://operator-1.example/team", "level": "verified_on_page", "text_sha256": "e" * 64,
                  "quote": "Jordan Fixture is Plant Manager at 2 Example Road, Fixture City, TX."}}
    assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"


@pytest.mark.parametrize("reason", ["target_site_responsibility_unproven", "target_site_location_mismatch"])
@pytest.mark.parametrize("title", ["Plant Manager", "Director"])
@pytest.mark.parametrize("source", ["public_quote", "provider_sourced"])
def test_loader_requalifies_quoted_referral_against_retained_facility(tmp_path, reason, title, source):
    changes = {**team(1), "person_name": PERSON, "person_title": title,
               "person_url": "https://operator-1.example/team",
               "person_quote": f"{PERSON} is {title} at Synthetic Operator 1.", "person_date": "2026-06-01"}
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, changes)])
    person = {"source": source, "name": PERSON, "title": title}
    if source == "provider_sourced":
        person.update(corroborated=True, corroboration={"url": "https://operator-1.example/team",
                      "quote": changes["person_quote"], "level": "verified_on_page", "text_sha256": "a" * 64})
    lookup(workspace, key, person=person)
    bundle, _, _ = built(workspace)
    entry = bundle["results"][0]
    assert entry["recipient"]["route"] == ("quoted_person_looked_up_email" if source == "public_quote"
                                           else "provider_sourced_corroborated")
    person = entry["recipient"]["person"]
    (person if source == "public_quote" else person["corroboration"])["quote"] = (
        f"{PERSON} is {title} at Rival and Sons, Fixture City, TX.")
    person["site_responsibility"]["reason"] = reason
    raw = canonical(bundle).encode()
    if title == "Director":
        assert sa.load_bundle(raw) == bundle
    else:
        with pytest.raises(sa.AdmissionError, match="screen_admission_recipient_invalid"):
            sa.load_bundle(raw)


@pytest.mark.parametrize("responsibility", [
    {"site_key": "a" * 64, "status": "unknown", "route": "corporate_referral",
     "reason": "target_site_responsibility_unproven", "proof": None},
    {"site_key": None, "status": "unknown", "route": "hold",
     "reason": "target_site_location_mismatch", "proof": None},
])
def test_held_or_another_site_lookup_cannot_replace_the_team_inbox(tmp_path, responsibility):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, team(1))])
    if responsibility["site_key"] is None:
        responsibility = {**responsibility, "site_key": key}
    lookup(workspace, key, address="jordan.fixture@operator-1.example", person={
        "source": "provider_sourced", "name": "Jordan Fixture", "title": "Plant Manager",
        "corroborated": False, "site_responsibility": responsibility})
    assert built(workspace)[0]["results"][0]["recipient"]["route"] == "published_team_inbox"


@pytest.mark.parametrize("display_location", ["Fixture City, TX", None])
@pytest.mark.parametrize("operator_prefix", ["", "Synthetic Operator 1's "])
def test_site_role_loader_uses_retained_physical_proof_when_display_omits_street(tmp_path, display_location, operator_prefix):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, None)])
    lookup(workspace, key, address="jordan.fixture@operator-1.example", person={
        "source": "provider_sourced", "name": "Jordan Fixture", "title": "Plant Manager", "corroborated": False})
    bundle, _, _ = built(workspace)
    entry = bundle["results"][0]
    if display_location is None:
        entry["input"].pop("location")
    else:
        entry["input"]["location"] = display_location
    entry["candidate"]["location"] = display_location or entry["candidate"]["site"]
    entry["result_digest"] = sa.result_digest(entry)
    entry["recipient"]["person"]["site_responsibility"] = {
        "site_key": key, "status": "verified", "route": "site_contact", "reason": None,
        "proof": {"url": "https://operator-1.example/team", "level": "verified_on_page", "text_sha256": "e" * 64,
                  "quote": f"Jordan Fixture is Plant Manager at {operator_prefix}1 Example Road, Fixture City, TX."}}
    entry["hypothesis"]["question"] = entry["checks"]["question"]
    assert sa.responsibility_target(entry)["address"]["street"] == "1 Example Road"
    assert sa.load_bundle(canonical(bundle).encode()) == bundle
    entry["checks"]["verification"]["site_identity"]["level"] = "unproven"
    entry["result_digest"] = sa.result_digest(entry)
    assert sa.recipient_problem(entry["recipient"], entry) == "screen_admission_recipient_invalid"


@pytest.mark.parametrize("found_state", ["TX", None])
def test_source_and_loader_reconstruct_the_same_proven_street_from_city_only_input(tmp_path, found_state):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, None)])
    states, _ = workspace.states()
    record = ss.stage_records(workspace, states, "screen")[key]
    contact = ss.stage_records(workspace, states, "contact")[key]
    record["address"] = {"city": "Fixture City", "state": "TX"}
    record["input"]["location"] = "Fixture City, TX"
    if found_state is None:
        for field in ("site_identity", "site_identity_quote"):
            record["answers"][field] = record["answers"][field].replace(", TX", "").replace(" TX", "")
    source = sa.contact_lookup.lookup_site(workspace, contact, record, today=TODAY)
    loader = sa.responsibility_target({"input": record["input"], "answers": record["answers"],
                                      "checks": {"verification": record["verification"]}})
    assert source["address"] == loader["address"] == {"street": "1 Example Road", "city": "Fixture City", "state": "TX"}
    assert source["operator"] == loader["operator"] == "Synthetic Operator 1"
    person = {"name": "Jordan Fixture", "title": "Plant Manager"}
    for target in (source, loader):
        assert not sa.contact_lookup.site_role_quote(
            target, person, "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City.")
        assert sa.contact_lookup.site_role_quote(
            target, person, "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX.")
    # A published verified person email outranks every looked-up address.
    workspace, (key,) = prepared(tmp_path / "published", [(1, FOCUS_A, {})])
    lookup(workspace, key)
    assert built(workspace)[0]["results"][0]["recipient"]["route"] == "published_person_email"


# --- the real bridge: pin, worker step, CRM rows and the WebApp snapshot ------------------------------------------
def bridge_script(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    script, files = tmp_path / "bridge.mjs", {name: str(tmp_path / f"{name}.json") for name in
                                              ("firestore", "bucket", "sheet", "faults", "puts", "reads", "clock")}
    uri = {name: json.dumps((ROOT / path).as_uri()) for name, path in (
        ("bridge", "tools/daily_research/firestore_bridge.mjs"), ("publisher", "tools/daily_research/publisher.mjs"),
        ("memory", "tests/fixtures/daily_research/firestore-memory.mjs"),
        ("bucket", "tests/fixtures/daily_research/fake-bucket.mjs"))}
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {readFileSync, writeFileSync, existsSync} from 'node:fs';",
        f"import {{Store, LeaseChannel}} from {uri['bridge']};",
        f"import {{Publisher, SHEET}} from {uri['publisher']};",
        f"import {{MemoryFirestore}} from {uri['memory']};",
        f"import {{FakeBucket}} from {uri['bucket']};",
        f"const files = {json.dumps(files)};",
        "const read = (name, fallback) => existsSync(files[name]) ? JSON.parse(readFileSync(files[name], 'utf8')) : fallback;",
        "const now = () => read('clock', null) ?? Date.now();",
        "const crmReader = async () => {",
        "  const reads = read('reads', 0) + 1, faults = read('faults', {});",
        "  writeFileSync(files.reads, JSON.stringify(reads));",
        "  if (faults.change_on_read === reads) writeFileSync(files.sheet, JSON.stringify([...read('sheet', []), faults.row]));",
        "  return {sheet_id: SHEET, complete: true, captured_at: new Date(now()).toISOString(), values: read('sheet', [])};",
        "};",
        "const google = async (method, path, body) => {",
        "  if (method === 'GET') return {sheets: []};",
        "  const faults = read('faults', {});",
        "  writeFileSync(files.puts, JSON.stringify(read('puts', 0) + 1));",
        "  if (faults.put === 'fail_before') throw new Error('synthetic sheets failure');",
        "  writeFileSync(files.sheet, JSON.stringify([...read('sheet', []), ...body.values]));",
        "  if (faults.put === 'lost_after') throw new Error('synthetic lost reply');",
        "  return {};",
        "};",
        "const publisher = new Publisher({crmReader, google, notion: async () => ({}), clock: now});",
        "const db = new MemoryFirestore(files.firestore), bucket = new FakeBucket(files.bucket);",
        "const channel = new LeaseChannel(new Store(db, now, undefined, crmReader, publisher, null, null, false, bucket));",
        "for await (const line of createInterface({input: process.stdin})) {",
        "try {const value = await channel.call(JSON.parse(line)); process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error) {process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}",
        "await channel.close();",
    ]))
    return script, files


class World:
    """One hermetic worker: the bridge, its files, an out dir and helpers."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.script, self.files = bridge_script(tmp_path)
        self.write("clock", int(NOW.timestamp() * 1000))
        self.write("sheet", [["Synthetic CRM"], [], [], [], HEADERS])
        self.bridge = Bridge(script=self.script)
        control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
        self.bridge.call("init", value=control)
        self.root = tmp_path / "work"
        self.root.mkdir()

    def write(self, name, value):
        Path(self.files[name]).write_text(json.dumps(value))

    def read(self, name, default=None):
        path = Path(self.files[name])
        return json.loads(path.read_text()) if path.exists() else default

    def direct(self, paths="daily_qa,site_screen", **changes):
        options = {"apply": True, "now": NOW - timedelta(hours=1), "paths": paths, "max_rows_per_batch": 50,
                   "approval_reference": APPROVAL, "approved_by": "owner", "reason": "Synthetic screen admission",
                   "sleep": lambda _: None, **changes}
        applied = direction_operator.set_direction(self.bridge, direction_operator.BridgeObjects(self.bridge), **options)
        return applied["next"]["sha256"], applied["object"]["generation"]

    def upload(self, raw):
        """The owner's create-only upload of one bundle, into the same fake bucket the worker reads."""
        bucket = self.read("bucket", {"objects": [], "generation": 1000})
        name = f"{sa.OBJECT_PREFIX}{hashlib.sha256(raw).hexdigest()}/{sa.OBJECT_NAME}"
        bucket["generation"] += 1
        bucket["objects"].append([name, {"raw": base64.b64encode(raw).decode(), "generation": bucket["generation"],
                                         "metadata": None}])
        self.write("bucket", bucket)
        # The fake object store loads its file when the bridge starts, as a new worker process would.
        self.bridge.close()
        self.bridge = Bridge(script=self.script)
        return str(bucket["generation"])

    def admission(self, workspace, **options):
        sha, generation = options.pop("direction", None) or self.direct()
        bundle, raw, report = built(workspace, sha=sha, generation=generation, **options)
        return bundle, report["admission_id"], self.upload(raw)

    def pin(self, admission_id, generation, **changes):
        options = {"admission_id": admission_id, "generation": generation, "approval_reference": APPROVAL,
                   "apply": True, "now": NOW, "sleep": lambda _: None, **changes}
        return sa.pin(self.bridge, **options)

    def step(self):
        return sa.step(self.bridge, now=NOW, root=self.root)

    def rows(self):
        return self.read("sheet")[5:]

    def communications_lap(self, record):
        """Inject a synthetic durable lap, then load it through the real bridge."""
        self.bridge.close()
        documents = dict(self.read("firestore", []))
        documents["blueprintCommunications/default/intakeState/workerLap"] = record
        self.write("firestore", list(documents.items()))
        self.bridge = Bridge(script=self.script)


@pytest.fixture
def world(tmp_path):
    value = World(tmp_path)
    yield value
    value.bridge.close()


def pinned(world, tmp_path, sites, **options):
    workspace, keys = prepared(tmp_path / "out", sites)
    bundle, admission_id, generation = world.admission(workspace, **options)
    assert world.pin(admission_id, generation)["state"] == "pinned"
    return workspace, keys, bundle, admission_id


def test_pin_needs_the_live_direction_for_site_screen_and_the_exact_uploaded_generation(world, tmp_path):
    workspace, _ = prepared(tmp_path / "out", [(1, FOCUS_A, {})])
    daily_only = world.direct(paths="daily_qa")
    _, admission_id, generation = world.admission(workspace, direction=daily_only)
    with pytest.raises(Refusal, match="screen_admission_path_not_directed"):
        world.pin(admission_id, generation)
    with pytest.raises(Refusal, match="screen_admission_object_missing"):
        world.pin(admission_id, str(int(generation) + 7))
    _, admission_id, generation = world.admission(workspace)
    planned = world.pin(admission_id, generation, apply=False)
    assert planned["state"] == "planned" and "screen_admission" not in world.bridge.call("control")
    with pytest.raises(Refusal, match="screen_admission_direction_expired"):
        world.pin(admission_id, generation, now=NOW + timedelta(days=60))
    with pytest.raises(Refusal, match="screen_admission_reference_invalid"):
        world.pin(admission_id, generation, approval_reference="PENDING owner")
    applied = world.pin(admission_id, generation)
    assert applied["state"] == "pinned" and applied["readback_verified"] is True
    assert world.bridge.call("control")["screen_admission"] == {"enabled": True, "current": {
        "admission_id": admission_id, "generation": generation, "bytes": applied["pin"]["bytes"],
        "uri": sa.uri(admission_id), "records": 1, "direction_sha256": applied["pin"]["direction_sha256"],
        "approval_reference": APPROVAL}}
    with pytest.raises(Refusal, match="screen_admission_control_conflict"):
        world.pin(admission_id, generation, expect="none")
    shown = sa.show(world.bridge, now=NOW)
    assert (shown["state"], shown["object_verified"], shown["direction"], shown["admission"]) == ("enabled", True, "ok", None)
    assert named(shown) == [] and named(applied) == []


def test_the_worker_writes_labelled_rows_once_with_markers_then_hands_off_to_the_webapp(world, tmp_path):
    _, _, bundle, admission_id = pinned(world, tmp_path, [(1, FOCUS_A, {}), (2, FOCUS_A, team(2))])
    first = world.step()
    assert first == {"state": "screen_admission_acknowledged", "rows": 2, "duplicates": 0}
    rows = world.rows()
    assert [row[0] for row in rows] == ["BP-000001", "BP-000002"]
    person, inbox = rows
    assert person[1:12] == ["Synthetic Operator 1", "Facility / site", "Synthetic Works 1", PERSON,
                            "avery.placeholder@operator-1.example (published person email)", "Hypothesis",
                            "https://operator-1.example/team", "unknown", "https://operator-1.example/careers/lathe",
                            "Research", ""]
    result = bundle["results"][0]
    assert person[12] == (f"First email asks: {result['hypothesis']['question']}\n"
                          f"[screen:{admission_id};{result['result_digest']}]")
    assert person[13:] == ["", "CNC machine tending", "", "Outreach-ready: operator, site, task proven",
                           "1 Example Road, Fixture City, TX 00001", "2026-10-05"]
    assert inbox[4:8] == ["", "plant.team@operator-2.example (published team inbox)", "Hypothesis",
                          "https://operator-2.example/contact"]
    assert world.read("puts") == 1
    # Idempotent: further passes and a repeated publish write nothing more.
    assert world.step()["state"] == "screen_admission_acknowledged"
    with pytest.raises(Refusal, match="firestore_lease_lost"):  # Publishing needs the worker lease.
        world.bridge.call("screen_admission_publish", admission_id=admission_id, generation="1", direction={})
    control = world.bridge.call("control")
    world.bridge.call("acquire", scope="research_release")
    try:
        repeated = world.bridge.call("screen_admission_publish", admission_id=admission_id,
                                     generation=control["screen_admission"]["current"]["generation"],
                                     direction=sa.direction_record(outreach_ready.current(control)[0]))
    finally:
        world.bridge.call("release")
    assert repeated == {"state": "acknowledged", "rows": 2, "duplicates": 0}
    assert world.read("puts") == 1 and len(world.rows()) == 2
    snapshot = world.bridge.call("screen_snapshot", admission_id=admission_id)
    assert snapshot["schema_version"] == "blueprint.site-screen-admission-snapshot.v1"
    assert base64.b64decode(snapshot["bundle"]) == canonical(bundle).encode()
    assert snapshot["work_item"]["stage"] == "completed" and snapshot["work_item"]["sends_authorized"] is False
    assert snapshot["state"]["receipt"]["reference"] == f"sheets:{runner.SHEET}:Prospects:BP-000001,BP-000002"
    assert snapshot["state"]["direction"]["paths"] == ["daily_qa", "site_screen"]
    assert snapshot["state"]["plan"]["sheet_rows"] == rows


def test_the_worker_waits_for_idle_daily_work_and_never_writes_under_an_active_row(world, tmp_path):
    pinned(world, tmp_path, [(1, FOCUS_A, {})])
    world.bridge.call("acquire")
    try:
        world.bridge.call("import_run", row={"date": "2026-10-06", "run_key": "blueprint-researcher:2026-10-06",
                                             "state": "running", "metadata": {}, "cleanup_required": False})
    finally:
        world.bridge.call("release")
    assert world.step() == {"state": "screen_admission_waiting", "error": "screen_admission_daily_work_active"}
    assert sa.retry(world.step()) is True and world.rows() == [] and world.read("puts") is None


def test_expired_communications_lap_blocks_rows_until_explicit_full_completion(world, tmp_path):
    pinned(world, tmp_path, [(1, FOCUS_A, {})])
    at = int(NOW.timestamp() * 1000)
    record = {"schema_version": "blueprint.communications-worker-lap.v1", "phase": "active",
              "lease": {"owner": "communications-worker-lap:synthetic", "generation": 1, "until": at - 1},
              "startedAt": at - 1000, "renewedAt": at - 500, "completedAt": None}
    world.communications_lap(record)
    assert world.step() == {"state": "screen_admission_waiting", "error": "screen_admission_daily_work_active"}
    assert sa.retry(world.step()) and world.rows() == [] and world.read("puts") is None
    # Merely changing phase/expiry is insufficient: the full producer record
    # must contain the explicit integer completion time.
    record.update(phase="complete", lease={**record["lease"], "until": 0})
    world.communications_lap(record)
    assert not sa.idle(world.bridge) and world.read("puts") is None
    record["completedAt"] = at
    world.communications_lap(record)
    assert sa.idle(world.bridge)
    assert world.step() == {"state": "screen_admission_acknowledged", "rows": 1, "duplicates": 0}
    assert world.read("puts") == 1


def test_a_lap_admitted_between_idle_and_acquire_is_a_retryable_refusal(world, tmp_path):
    pinned(world, tmp_path, [(1, FOCUS_A, {})])
    actual = world.bridge
    at = int(NOW.timestamp() * 1000)
    record = {"schema_version": "blueprint.communications-worker-lap.v1", "phase": "active",
              "lease": {"owner": "communications-worker-lap:synthetic", "generation": 1, "until": at + 1000},
              "startedAt": at, "renewedAt": at, "completedAt": None}

    class RacingBridge:
        def call(self, op, **fields):
            if op == "acquire":
                assert fields == {"scope": "research_release"}
                world.communications_lap(record)
                return world.bridge.call(op, **fields)
            return actual.call(op, **fields)

    result = sa.step(RacingBridge(), now=NOW, root=world.root)
    assert result == {"state": "screen_admission_blocked", "error": "communications_worker_lap_active"}
    assert sa.retry(result) and world.rows() == [] and world.read("puts") is None


def test_crm_duplicates_and_earlier_admissions_are_never_written_again(world, tmp_path):
    held = ["BP-000007", "Synthetic Operator 2", "Facility / site", "Synthetic Works 2", "", "", "Needs recheck", "",
            "unknown", "https://operator-2.example/careers/lathe", "Research", "", "Earlier row", "",
            "CNC machine tending", "", "Unverified", "2 Example Road, Fixture City, TX 00002", "2026-10-01"]
    world.write("sheet", [["Synthetic CRM"], [], [], [], HEADERS, held])
    _, keys, _, first = pinned(world, tmp_path, [(1, FOCUS_A, {}), (2, FOCUS_A, {}), (3, FOCUS_B, {})],
                                       keys=None, per_focus=2)
    assert world.step() == {"state": "screen_admission_acknowledged", "rows": 2, "duplicates": 1}
    assert [row[0] for row in world.rows()] == ["BP-000007", "BP-000008", "BP-000009"]
    state = world.bridge.call("screen_snapshot", admission_id=first)["state"]
    assert state["payload"]["duplicates"] == [{"site_key": keys[1], "code": "screen_admission_crm_duplicate"}]
    # A later admission naming an admitted site writes only the new one.
    more = prepared(tmp_path / "more", [(1, FOCUS_A, {}), (4, FOCUS_B, {})])[0]
    _, second, generation = world.admission(more, direction=(state["direction"]["sha256"],
                                                             state["direction"]["generation"]))
    world.pin(second, generation)
    assert world.step() == {"state": "screen_admission_acknowledged", "rows": 1, "duplicates": 1}
    duplicates = world.bridge.call("screen_snapshot", admission_id=second)["state"]["payload"]["duplicates"]
    assert duplicates == [{"site_key": keys[0], "code": "screen_admission_site_already_admitted"}]
    assert [row[1] for row in world.rows()] == ["Synthetic Operator 2", "Synthetic Operator 1", "Synthetic Operator 3",
                                                "Synthetic Operator 4"]


def test_an_uncertain_write_is_reconciled_by_reading_only_and_never_repeated(world, tmp_path):
    _, _, _, admission_id = pinned(world, tmp_path, [(1, FOCUS_A, {})])
    world.write("faults", {"put": "lost_after"})
    assert world.step() == {"state": "screen_admission_acknowledged", "rows": 1, "duplicates": 0}  # Read back at once.
    assert world.read("puts") == 1
    other = World(tmp_path / "lost")
    try:
        _, _, _, admission_id = pinned(other, tmp_path / "lost", [(1, FOCUS_A, {})])
        other.write("faults", {"put": "fail_before"})
        assert other.step() == {"state": "screen_admission_write_uncertain", "rows": 1, "duplicates": 0}
        other.write("faults", {})
        for _ in range(2):
            assert other.step()["state"] == "screen_admission_write_uncertain"
        assert other.read("puts") == 1 and other.rows() == []
        assert other.bridge.call("screen_admission_get", admission_id=admission_id)["state"] == "claimed"
        # A new pin cannot take over the uncertain one without the owner's explicit flag.
        more = prepared(tmp_path / "lost-more", [(4, FOCUS_B, {})])[0]
        _, newer, generation = other.admission(more, direction=other.direct())
        with pytest.raises(Refusal, match="screen_admission_direction_changed|screen_admission_current_write_uncertain"):
            other.pin(newer, generation)
    finally:
        other.bridge.close()


def test_a_crm_change_before_the_claim_makes_a_fresh_plan_and_nothing_is_written_twice(world, tmp_path):
    pinned(world, tmp_path, [(1, FOCUS_A, {})])
    unrelated = ["BP-000003", "Unrelated Operator", "Facility / site", "North", "", "", "Needs recheck", "", "unknown",
                 "https://unrelated.example/jobs", "Research", "", "", "", "Packing", "", "Unverified", "Elsewhere",
                 "2026-10-01"]
    # CRM reads of a first pass: the worker's refresh, the plan's own read, the readback, then the pre-claim check.
    # Someone adds a row between the plan and the claim.
    world.write("faults", {"change_on_read": 3, "row": unrelated})
    assert world.step() == {"state": "screen_admission_replan_required", "rows": 1, "duplicates": 0}
    assert world.read("puts") is None
    assert world.step() == {"state": "screen_admission_acknowledged", "rows": 1, "duplicates": 0}
    assert world.read("puts") == 1 and [row[0] for row in world.rows()] == ["BP-000003", "BP-000004"]


def test_the_brakes_stop_the_admission_before_any_write(world, tmp_path):
    _, _, _, admission_id = pinned(world, tmp_path, [(1, FOCUS_A, {})])
    assert sa.disable(world.bridge, apply=True, sleep=lambda _: None)["state"] == "disabled"
    assert world.step() == {"state": "screen_admission_disabled"} and world.read("puts") is None
    current = world.bridge.call("control")["screen_admission"]["current"]
    world.bridge.call("acquire", scope="research_release")
    try:
        world.bridge.call("screen_admission_set", expected_admission_id=admission_id,
                          value={"enabled": True, "current": current}, supersede_uncertain=False)
    finally:
        world.bridge.call("release")
    direction_operator.disable(world.bridge, apply=True, sleep=lambda _: None)
    assert world.step() == {"state": "screen_admission_blocked", "error": "screen_admission_direction_disabled"}
    assert world.read("puts") is None and not sa.retry(world.step())


def test_configure_keeps_the_pin_and_init_refuses_one(world, tmp_path):
    pinned(world, tmp_path, [(1, FOCUS_A, {})])
    control = world.bridge.call("control")
    replacement = {key: value for key, value in control.items() if key not in {"lease", "screen_admission"}}
    world.bridge.call("acquire")
    try:
        world.bridge.call("configure", value=replacement)
        with pytest.raises(Refusal, match="screen_admission_requires_pin_operation"):
            world.bridge.call("configure", value={**replacement, "screen_admission": None})
    finally:
        world.bridge.call("release")
    assert world.bridge.call("control")["screen_admission"] == control["screen_admission"]
    fresh = Bridge(script=bridge_script(tmp_path / "fresh")[0])
    try:
        with pytest.raises(Refusal, match="screen_admission_requires_pin_operation"):
            initial = {key: value for key, value in replacement.items() if key != "outreach_ready"}
            fresh.call("init", value={**initial, "screen_admission": {}})
    finally:
        fresh.close()


def test_operator_commands_print_counts_only(world, tmp_path, capsys):
    workspace, _ = prepared(tmp_path / "out", [(1, FOCUS_A, {})])
    sha, generation = world.direct()
    out = str(workspace.root)
    planned = site_operator.main(["admit", "--out", out, "--direction-sha256", sha, "--direction-generation", generation])
    uploaded = site_operator.main(["admit", "--out", out, "--direction-sha256", sha, "--direction-generation", generation,
                                   "--apply"], objects=FakeObjects())
    assert planned["state"] == "planned" and uploaded["state"] == "uploaded"
    generation = world.upload((workspace.root / "admissions" / f"{uploaded['admission_id']}.json").read_bytes())
    factory, clock = (lambda: Bridge(script=world.script)), (lambda: NOW)
    pinned_result = site_operator.main(["admission-pin", "--admission-id", uploaded["admission_id"], "--generation",
                                        generation, "--approval-reference", APPROVAL, "--apply"],
                                       bridge_factory=factory, clock=clock, sleep=lambda _: None)
    assert pinned_result["state"] == "pinned"
    assert site_operator.main(["admission-show"], bridge_factory=factory, clock=clock)["state"] == "enabled"
    assert site_operator.main(["admission-disable", "--apply"], bridge_factory=factory, clock=clock,
                              sleep=lambda _: None)["state"] == "disabled"
    output = capsys.readouterr().out
    assert not any(value in output for value in (*SITE_STRINGS, "placeholder", "@"))
    assert site_operator.error_code(Refusal("screen_admission_direction_expired")) == "screen_admission_direction_expired"
    assert site_operator.error_code(ValueError("upstream text")) == "site_screen_operation_unavailable"


# --- the WebApp golden snapshot ----------------------------------------------------------------------------------
GOLDEN = ROOT / "tests/fixtures/daily_research/screen-admission-snapshot.json"
GOVERNMENT = "https://records.synthetic.gov/facility/3"


def golden(tmp_path, monkeypatch):
    """One acknowledged admission as the WebApp reads it (Store.screenSnapshot), with the CRM sheet and the control
    pins, covering every recipient route: a published person email, a team inbox, a government-proven operator whose
    domain contact rule v3 proves from the website (general inbox), a site with no contact, and the three provider
    lookup routes (quoted person, provider-sourced corroborated and not). Fixed clocks only."""
    monkeypatch.setattr(ss, "_now", lambda: "2026-10-05T20:00:00+00:00")
    world = World(tmp_path)
    try:
        quoted = {**NOBODY, "person_name": PERSON, "person_title": "Plant Manager",
                  "person_url": "https://operator-5.example/team",
                  "person_quote": f"{PERSON} leads the machining plant as plant manager.", "person_date": "2026-06-01"}
        workspace, keys = prepared(tmp_path / "out", [
            (1, FOCUS_A, {}), (2, FOCUS_A, team(2)), (3, FOCUS_B, general(3), {"operator_identity_url": GOVERNMENT}),
            (4, FOCUS_B, None), (5, FOCUS_C, quoted), (6, FOCUS_C, None), (7, FOCUS_D, None)])
        provider = {"name": "fullenrich", "status": "valid", "score": 97, "checked_at": "2026-10-05T12:00:00+00:00",
                    "request_digest": "c" * 64}
        lookup(workspace, keys[4], address="avery.placeholder@operator-5.example", provider=provider,
               person={"source": "public_quote", "name": PERSON, "title": "Plant Manager"})
        lookup(workspace, keys[5], address="jordan.fixture@operator-6.example", provider=provider,
               person={"source": "provider_sourced", "name": "Jordan Fixture", "title": "Operations Lead",
                       "corroborated": True, "corroboration": {
                           "url": "https://operator-6.example/news", "level": "verified_on_page", "text_sha256": "a" * 64,
                           "quote": "Jordan Fixture, operations lead, runs the Fixture City plant every day."}})
        lookup(workspace, keys[6], address="riley.example@operator-7.example", provider=provider,
               person={"source": "provider_sourced", "name": "Riley Example", "title": "Plant Superintendent",
                       "corroborated": False})
        _, admission_id, generation = world.admission(workspace)
        assert world.pin(admission_id, generation)["state"] == "pinned"
        assert world.step() == {"state": "screen_admission_acknowledged", "rows": 7, "duplicates": 0}
        control = world.bridge.call("control")
        return {"admission_id": admission_id, "site_keys": keys,
                "snapshot": world.bridge.call("screen_snapshot", admission_id=admission_id),
                "sheet": world.read("sheet"),
                "control": {name: control[name] for name in ("outreach_ready", "screen_admission")}}
    finally:
        world.bridge.close()


def test_the_webapp_golden_snapshot_is_this_package_output(tmp_path, monkeypatch):
    """Blueprint-WebApp server/tests/fixtures/screen-admission-snapshot.json is a copy of this file. Regenerate it with
    BLUEPRINT_REGENERATE_SCREEN_GOLDEN=1 when the admission format changes, and copy it to the WebApp."""
    value = golden(tmp_path, monkeypatch)
    routes = [(result["recipient"] or {}).get("route") for result in json.loads(base64.b64decode(
        value["snapshot"]["bundle"]))["results"]]
    assert routes == ["published_person_email", "published_team_inbox", "published_general_inbox", None,
                      "quoted_person_looked_up_email", "provider_sourced_corroborated", "provider_sourced_uncorroborated"]
    if os.environ.get("BLUEPRINT_REGENERATE_SCREEN_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    assert json.loads(GOLDEN.read_text()) == value
    assert named({key: item for key, item in value.items() if key != "snapshot"}) != []  # Synthetic names only.


def test_default_admission_recomputes_lookup_records_from_current_states(tmp_path, monkeypatch):
    workspace, _ = prepared(tmp_path, [(1, FOCUS_A, {})])
    load = sa.contact_lookup.load
    seen = []
    def read(current, *, states, today):
        seen.append((states, today))
        return load(current, states=states, today=today)
    monkeypatch.setattr(sa.contact_lookup, "load", read)
    bundle, _, _ = sa.build(workspace, direction_sha256=DIRECTION_SHA, direction_generation=DIRECTION_GENERATION)
    assert len(seen) == 1 and seen[0][0]["contact"] and seen[0][1] == TODAY
    assert bundle["results"][0]["recipient"]["route"] == "published_person_email"


def test_final_provider_loader_keeps_status_employment_and_refuses_tampering(tmp_path):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, team(1))])
    lookup(workspace, key, address="jordan.fixture@operator-1.example", person={"source": "provider_sourced",
           "name": "Jordan Fixture", "title": "Operations Manager", "corroborated": False})
    entry = built(workspace)[0]["results"][0]
    recipient = entry["recipient"]
    assert recipient["provider"]["verification_status"] == "DELIVERABLE"
    assert recipient["provider"]["score"] is None
    assert recipient["person"]["proof"]["current_employment"]["company_domain"] == "operator-1.example"
    for field, value in (("verification_status", None), ("name", "other-provider")):
        changed = json.loads(json.dumps(recipient))
        changed["provider"][field] = value
        assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"
    changed = json.loads(json.dumps(recipient))
    changed["person"]["proof"]["current_employment"]["field"] = "employment.all.end_at_absent"
    assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"
    changed = json.loads(json.dumps(recipient))
    changed["address"] = "sales@operator-1.example"
    assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"


def test_published_person_loader_refuses_role_or_noncurrent_proof(tmp_path):
    workspace, _ = prepared(tmp_path, [(1, FOCUS_A, {})])
    entry = built(workspace)[0]["results"][0]
    recipient = entry["recipient"]
    changed = json.loads(json.dumps(recipient))
    changed["address"] = "plant.team@operator-1.example"
    changed["published"]["quote"] = "Business inquiries: plant.team@operator-1.example"
    assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"
    changed = json.loads(json.dumps(recipient))
    changed["person"]["current"] = False
    assert sa.recipient_problem(changed, entry) == "screen_admission_recipient_invalid"


def test_admission_rechecks_published_person_date_and_allows_provider_fallback(tmp_path, monkeypatch):
    workspace, (key,) = prepared(tmp_path, [(1, FOCUS_A, {})])
    monkeypatch.setattr(sa.contact_lookup, "utc_today", lambda: TODAY + timedelta(days=549))
    assert built(workspace)[0]["results"][0]["recipient"] is None
    lookup(workspace, key, address="jordan.fixture@operator-1.example", person={"source": "provider_sourced",
           "name": "Jordan Fixture", "title": "Operations Manager", "corroborated": False})
    assert built(workspace)[0]["results"][0]["recipient"]["route"] == "provider_sourced_uncorroborated"
