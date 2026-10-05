"""Hermetic fixtures for the site screen tests: a fake Parallel Task API and a fake page reader.

Every company, site, person, address and email here is synthetic. Hosts use the reserved ``.example``
top-level domain, and person names carry a fixture marker word.
"""
import json
import re
from datetime import date

from tests.test_daily_research_site_universe import site as universe_site
from tools.daily_research import site_screen as ss
from tools.daily_research.search import ToolFailure

KEY = "synthetic-parallel-key-7f3a9c41"  # Never a real key; the tests check it never leaves the client.
OWNER = "owner-decision-synthetic-20261005"  # The owner reference the first --apply pins.
TODAY = date(2026, 10, 5)
PERSON = "Avery Placeholder"
OTHER_PERSON = "Jordan Fixture"
PEOPLE = (PERSON, OTHER_PERSON)
# Strings that identify a synthetic site; command output must never contain them.
SITE_STRINGS = ("Synthetic Works", "Synthetic Operator", "Example Road", "Fixture City", ".example", PERSON)


class FakeProvider:
    """The Parallel Task API behind TaskClient's transport seam. It records every request, creates runs,
    reports scripted statuses and returns scripted outputs keyed by the run's site_key metadata."""

    def __init__(self):
        self.calls, self.runs = [], {}
        self.outputs, self.contacts = {}, {}  # site_key -> {"content": ..., "basis": [...]} per form
        self.final, self.progress = {}, {}  # site_key -> terminal status / statuses reported before it
        self.create_answers, self.read_answers = [], []  # Scripted answers used first: (status, body) or an error.

    def __call__(self, method, path, *, headers, body, timeout):
        self.calls.append({"method": method, "path": path, "headers": dict(headers), "body": body, "timeout": timeout})
        scripted = self.create_answers if method == "POST" else self.read_answers
        if scripted:
            answer = scripted.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        if method == "POST":
            assert path == ss.RUNS_PATH and headers["Content-Type"] == "application/json"
            request = json.loads(body)
            run_id = f"trun_{len(self.runs) + 1:04d}"
            self.runs[run_id] = request
            return 202, json.dumps({"run_id": run_id, "status": "queued", "is_active": True}).encode()
        match = re.fullmatch(r"/v1/tasks/runs/(trun_\d{4})(/result\?timeout=30)?", path)
        request = self.runs[match.group(1)]
        key, form = request["metadata"]["site_key"], request["metadata"]["form"]
        waiting = self.progress.get(key)
        status = waiting.pop(0) if waiting and not match.group(2) else self.final.get(key, "completed")
        run = {"run_id": match.group(1), "status": status, "is_active": status not in ss.TERMINAL}
        if not match.group(2):
            return 200, json.dumps(run).encode()
        if status != "completed":
            return 404, b'{"detail": "Run failed or run id not found"}'
        output = (self.outputs if form == ss.SCREEN else self.contacts)[key]
        return 200, json.dumps({"run": run, "output": {"type": "json", **output}}).encode()

    def creates(self, form=None):
        return [json.loads(call["body"]) for call in self.calls if call["method"] == "POST"
                and (form is None or json.loads(call["body"])["metadata"]["form"] == form)]


class FakePages:
    """Public pages by URL. A missing URL fails like a read the site blocks; every request is recorded."""

    def __init__(self, pages=None):
        self.pages, self.requested = dict(pages or {}), []

    def __call__(self, url):
        self.requested.append(url)
        if url not in self.pages:
            raise ToolFailure("source_http_failure")
        return {"text": self.pages[url]}


class Clock:
    """A monotonic clock that only sleeping moves."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def inventory_record(number, **changes):
    record = {"operator": f"Synthetic Operator {number}", "site": f"Synthetic Works {number}",
              "location": "Fixture City, TX", "task_hypothesis": "CNC machine tending",
              "source_urls": [f"https://operator-{number}.example/plant"], "evidence_gap": "Task volume is unknown.",
              "disposition": "candidate"}
    record.update(changes)
    return record


def universe_row(number, sources=("epa_frs", "osha_ita"), **changes):
    row = universe_site(number)
    row.update(sources=sorted(sources), **changes)
    return row


def screen_answers(number, **changes):
    """A complete screen form answer for site ``number``: every proof quoted on its own page."""
    site = f"https://operator-{number}.example"
    value = {
        "website": site,
        "operator_identity": f"Synthetic Operator {number}",
        "operator_identity_url": f"{site}/about",
        "operator_identity_quote": f"Synthetic Operator {number} runs the machining plant on {number} Example Road.",
        "site_identity": f"{number} Example Road, Fixture City, TX",
        "site_identity_url": f"{site}/contact",
        "site_identity_quote": f"Our plant is at {number} Example Road, Fixture City, TX.",
        "operating_now": "yes", "operating_now_url": f"{site}/news",
        "operating_now_quote": "The plant added a second shift this spring.", "operating_now_date": "2026-05-01",
        "target_task": "CNC machine tending", "target_task_found": "yes", "target_task_url": f"{site}/capabilities",
        "target_task_quote": "Operators load and unload twelve CNC lathes on every shift.",
        "target_task_date": "2026-04-02",
        "manual_today": "yes", "manual_today_url": f"{site}/careers",
        "manual_today_quote": "Machine operators load bar stock and unload finished parts by hand.",
        "manual_today_date": "2026-09-01",
        "existing_automation": "unknown", "existing_automation_url": "", "existing_automation_quote": "",
        "existing_automation_date": "",
        "notes": "No public evidence on robot use was found.",
    }
    value.update(changes)
    return value


def contact_answers(number, **changes):
    """A complete contact form answer for site ``number``: a named person and their published address."""
    site = f"https://operator-{number}.example"
    value = {
        "decision_role": "Plant manager", "person_name": PERSON, "person_title": "Plant Manager",
        "person_url": f"{site}/team", "person_quote": f"{PERSON} leads the machining plant as plant manager.",
        "person_date": "2026-06-01",
        "email": f"plant.lead@operator-{number}.example", "email_url": f"{site}/team",
        "email_quote": f"Write to plant.lead@operator-{number}.example for plant questions.",
        "channel_type": "person_email", "channel_url": "", "notes": "No direct phone line is published.",
    }
    value.update(changes)
    return value


def pages_for(answers, stems=None):
    """Pages that hold each listed proof's quote among other text (default: every proof of the answers' form)."""
    if stems is None:
        stems = tuple(ss.SCREEN_PROOFS) if "target_task" in answers else ("person", "email")
    pages = {}
    for stem in stems:
        url, quote = answers.get(stem + "_url"), answers.get(stem + "_quote")
        if url and quote:
            pages[url] = pages.get(url, "Synthetic page header.") + " " + quote + " Synthetic footer."
    return pages


def raw_input(records):
    return json.dumps(records).encode()


def screen(tmp_path, records, answers, pages, *, provider=None, reader=None, basis=None):
    """Run, collect and verify one input list end to end with the fakes. Returns the workspace, provider,
    reader and screen records by site key."""
    workspace = ss.Workspace(tmp_path / "out", create=True)
    provider = provider or FakeProvider()
    sites, _ = ss.load_sites(raw_input(records))
    for site, content in zip(sites, answers):
        provider.outputs[site["site_key"]] = {"content": content, "basis": (basis or {}).get(site["site_key"], [])}
    client = ss.TaskClient(KEY, transport=provider)
    ss.run(raw_input(records), workspace, client=client, owner_reference=OWNER, ceiling_usd="5", max_runs=100,
           apply=True)
    ss.collect(workspace, client=client, wait_seconds=0)
    reader = reader or FakePages(pages)
    ss.verify(workspace, reader=reader, today=TODAY)
    return workspace, provider, reader, {record["site_key"]: record for record in workspace.records("screen")}
