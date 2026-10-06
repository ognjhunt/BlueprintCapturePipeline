"""Hermetic fixtures for the team universe tests: a fake Parallel Task API, synthetic robot teams and their
public pages, and synthetic query sets and family weights.

Every company, host and address here is synthetic. Hosts use the reserved ``.example`` top-level domain and
every company name carries the fixture marker word ``Synthbot``. No real company appears in this file.
"""
import json
import re
from datetime import date

from tests.daily_research_site_screen_fixture import Clock, FakePages
from tools.daily_research import site_screen as ss
from tools.team_universe import rank as tr
from tools.team_universe import universe as tu

__all__ = ["Clock", "FakePages"]  # Re-exported from the site screen fixture.

KEY = "synthetic-parallel-key-5d2e8b17"  # Never a real key; the tests check it never leaves the client.
OWNER = "owner-decision-synthetic-team-universe-20261005"
SPEND = {"owner_reference": OWNER, "ceiling_usd": "5", "max_runs": 200}
TODAY = date(2026, 10, 5)
# Strings that identify a synthetic team; command output and summary.json must never contain them.
FIXTURE_STRINGS = ("Synthbot", "synthbot", ".example")


class FakeProvider:
    """The Parallel Task API behind TaskClient's transport seam. It records every request, creates runs,
    reports scripted statuses and returns the scripted output of the run's subject key."""

    def __init__(self):
        self.calls, self.runs, self.outputs = [], {}, {}  # outputs: site_key -> {"content": ..., "basis": [...]}
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
            run_id = f"trun_{len(self.runs) + 1:04d}"
            self.runs[run_id] = json.loads(body)
            return 202, json.dumps({"run_id": run_id, "status": "queued", "is_active": True}).encode()
        match = re.fullmatch(r"/v1/tasks/runs/(trun_\d{4})(/result\?timeout=30)?", path)
        key = self.runs[match.group(1)]["metadata"]["site_key"]
        waiting = self.progress.get(key)
        status = waiting.pop(0) if waiting and not match.group(2) else self.final.get(key, "completed")
        run = {"run_id": match.group(1), "status": status, "is_active": status not in ss.TERMINAL}
        if not match.group(2):
            return 200, json.dumps(run).encode()
        if status != "completed":
            return 404, b'{"detail": "Run failed or run id not found"}'
        return 200, json.dumps({"run": run, "output": {"type": "json", **self.outputs[key]}}).encode()

    def creates(self, form=None):
        bodies = [json.loads(call["body"]) for call in self.calls if call["method"] == "POST"]
        return [body for body in bodies if form is None or body["metadata"]["form"] == form]


# --- discovery --------------------------------------------------------------------------------
def query(number, family="funding_by_task", task_focus="palletizing_depalletizing", robot_form=None, **changes):
    value = {"id": f"synthetic-query-{number}", "family": family, "task_focus": task_focus, "robot_form": robot_form,
             "query": f"Synthetic robot companies for query {number} that stack cases on pallets."}
    value.update(changes)
    return value


def query_document(*queries, **changes):
    value = {"schema_version": tu.QUERY_SET, "query_set": "synthetic-team-queries", "reviewed_on": "2026-10-05",
             "form": tu.DISCOVERY, "since": "2024-01-01", "max_companies": 25,
             "queries": list(queries) or [query(1), query(2, family="stealth", task_focus=None),
                                          query(3, family="robot_learning", task_focus=None)]}
    value.update(changes)
    return value


def query_set(*queries, **changes):
    """A loaded synthetic query set (structure checked, no review pin)."""
    return tu.load_queries(json.dumps(query_document(*queries, **changes)).encode())


def company(number, **changes):
    """One discovered company: its own site, and a funding story that names it."""
    value = {"name": f"Synthbot Robotics {number}", "website": f"https://synthbot-{number}.example",
             "robot_form": "mobile_manipulator", "task_focus": "palletizing and case picking",
             "source_url": f"https://news.example/funding/{number}",
             "source_quote": f"Synthbot Robotics {number} raised a seed round to build palletizing robots.",
             "source_date": "2026-03-01"}
    value.update(changes)
    return value


def discovery_output(companies, notes="Synthetic notes on what could not be found."):
    return {"content": {"companies": list(companies), "notes": notes}, "basis": []}


def discovery_pages(*groups):
    """Pages that hold each company's source quote among other text."""
    pages = {}
    for companies in groups:
        for item in companies:
            url, quote = item.get("source_url"), item.get("source_quote")
            if url and quote:
                pages[url] = pages.get(url, "Synthetic news header.") + " " + quote + " Synthetic footer."
    return pages


# --- team screen ------------------------------------------------------------------------------
def screen_answers(number, **changes):
    """A complete team screen answer for team ``number``: every proof quoted on its own page, naming its answer."""
    site, name = f"https://synthbot-{number}.example", f"Synthbot Robotics {number}"
    value = {
        "company": name, "company_url": f"{site}/about",
        "company_quote": f"{name} builds mobile manipulators that palletize cases in warehouses.",
        "robot_forms": "mobile_manipulator", "robot_forms_url": f"{site}/product",
        "robot_forms_quote": "Our mobile manipulator drives to each pallet and stacks cases by itself.",
        "task_families": "palletizing_depalletizing", "task_evidence": "pilot",
        "task_evidence_url": f"{site}/news/pilot",
        "task_evidence_quote": f"{name} began a palletizing pilot with a regional grocery distributor this spring.",
        "task_evidence_date": "2026-05-01",
        "stage": "seed", "funding_amount": "USD 8 million", "funding_date": "2026-03-01",
        "funding_url": f"https://news.example/funding/{number}",
        "funding_quote": f"{name} raised a seed round to build palletizing robots.",
        "hq": "Fixture City, United States", "hq_url": f"{site}/about",
        "hq_quote": f"{name} is headquartered in Fixture City with its own test warehouse.",
        "deployment_geography": "United States", "deployment_geography_url": f"{site}/customers",
        "deployment_geography_quote": "Our robots run at customer warehouses across the United States today.",
        "design_partners": "yes", "design_partners_url": f"{site}/partners",
        "design_partners_quote": "We are accepting design partners for our palletizing pilot program now.",
        "simulation": "yes", "simulation_url": f"{site}/blog/simulation",
        "simulation_quote": "We train every policy in Isaac Sim before it reaches a customer site.",
        "learned_policy": "yes", "learned_policy_url": f"{site}/blog/learning",
        "learned_policy_quote": "Our learned policy handles mixed cases without any manual programming.",
        "shares_policy": "no", "shares_policy_url": "", "shares_policy_quote": "",
        "api_sdk": "yes", "api_sdk_url": f"{site}/developers",
        "api_sdk_quote": "The robot SDK lets integrators call every robot from their own software.",
        "seeking_partners": "yes", "seeking_partners_url": f"{site}/partners",
        "seeking_partners_quote": "We are looking for warehouse sites to join our next pilot cohort.",
        "seeking_partners_date": "2026-08-01",
        "contact_email": f"partnerships@synthbot-{number}.example", "contact_url": f"{site}/contact",
        "contact_quote": f"Write to partnerships@synthbot-{number}.example to discuss a pilot at your site.",
        "notes": "No public price list was found.",
    }
    value.update(changes)
    return value


def screen_pages(answers, stems=None):
    """Pages that hold each listed proof's quote among other text (default: every quoted answer)."""
    pages = {}
    for stem in tu.PROOFS if stems is None else stems:
        url, quote = answers.get(stem + "_url"), answers.get(stem + "_quote")
        if url and quote:
            pages[url] = pages.get(url, "Synthetic page header.") + " " + quote + " Synthetic footer."
    return pages


def weights(**values):
    """A synthetic family weights file: counts a private site-screen report could hold."""
    return json.dumps({"schema_version": "blueprint.team-family-weights.v1", "reference": "synthetic-weights-20261005",
                       "weights": values or {"palletizing_depalletizing": 6, "sorting_pick_and_place": 3}}).encode()


# --- end to end with the fakes (the tests' autouse fixture makes tmp_path an accepted out dir) -----------
def setup(tmp_path, provider=None):
    provider = provider or FakeProvider()
    return tu.TeamWorkspace(tmp_path / "out", create=True), provider, ss.TaskClient(KEY, transport=provider)


def discovered(tmp_path, outputs, *, pages=None, reader=None, spend=None):
    """Discover, collect and verify one synthetic query per company list in ``outputs``."""
    workspace, provider, client = setup(tmp_path)
    queries = query_set(*[query(number + 1) for number in range(len(outputs))])
    for item, companies in zip(queries["queries"], outputs):
        provider.outputs[tu.query_subject(queries, item)["site_key"]] = discovery_output(companies)
    tu.discover(workspace, queries, client=client, apply=True, **(spend or SPEND))
    ss.collect(workspace, client=client, wait_seconds=0)
    reader = reader or FakePages(discovery_pages(*outputs) if pages is None else pages)
    tu.verify(workspace, reader=reader, today=TODAY)
    return workspace, provider, client, reader


def pipeline(tmp_path, answers, *, pages=None, discovery=None, basis=None, weights_raw=None, spend=None):
    """Discover one synthetic team per screen answer, screen every team, then collect and verify. Returns the
    workspace, provider, reader and the screen records by team number."""
    numbers = sorted(answers)
    companies = discovery or [company(number) for number in numbers]
    workspace, provider, client, _ = discovered(tmp_path, [companies], spend=spend)
    for number in numbers:
        key = tu.team_key(f"synthbot-{number}.example")
        provider.outputs[key] = {"content": answers[number], "basis": (basis or {}).get(number, [])}
    loaded = tr.load_weights(weights_raw) if weights_raw else None
    tu.screen(workspace, client=client, apply=True, weights=loaded, **(spend or SPEND))
    if pages is None:
        pages = {url: text for number in numbers for url, text in screen_pages(answers[number]).items()}
    reader = FakePages(pages)
    ss.collect(workspace, client=client, reader=reader, today=TODAY, wait_seconds=0)
    tu.verify(workspace, reader=reader, today=TODAY)
    records = {record["domain"]: record for record in workspace.records("screen")}
    return workspace, provider, reader, {number: records.get(f"synthbot-{number}.example") for number in numbers}


def one(tmp_path, answers, **options):
    """The screen record of team 1 with these answers."""
    return pipeline(tmp_path, {1: answers}, **options)[3][1]
