"""Per-site research line ("site screen") through the Parallel Task API. Standard library only.

Owner decision 2026-10-05, after a successful 50-site pilot: each site gets one Parallel Task run
(processor ``core``, $0.025 per completed run; failed runs are not billed) that fills the versioned
``blueprint.site-screen.v1`` form, with a URL and an exact quote for every answer. Nothing here sends,
drafts or writes a CRM.

Inputs are daily-run discovery inventory records or site universe export rows (``load_sites``). On a
site universe row with an OSHA ITA or EPA FRS source, that government record is the primary source
for the operator and the exact site. A web-found site must prove both with its own quotes.

Spend. Each stage keeps an append-only JSONL ledger in the out dir, and an ``intent`` line is fsynced
before every create. A site with a stored run id, or with a create whose outcome is unknown (an
ambiguous answer, or an interruption after the intent), is never submitted again. Before every
create, the price of every run that may be billed (completed, in flight, cancelled or of unknown
outcome) plus the new one must stay at or below the ceiling, and the stage's runs at or below ``max_runs``;
otherwise the create is refused with a stable code. Only a run observed ``failed`` frees its price.

Verification reads each cited page once with the daily agent's own reader (``search.source``) under
its wall-time alarm. A quote is ``verified_on_page``; else
``in_citation_excerpt`` when one of the provider's citation excerpts for that field holds it (our read
failed, or the page shows it only after scripts run); else ``unverified``, or
``unverified_page_unreachable`` when our read failed. Matching is exact after normalization; a quote
of 40 or more characters also matches on its leading or trailing 90 %.

The API key is given to ``TaskClient`` and held only in its request headers: it is never read from a
module global, logged or written. Failures leave as stable ``site_screen_*`` codes, never upstream text.
"""
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import ssl
import threading
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

SCREEN = "blueprint.site-screen.v1"
INPUT = "blueprint.site-screen.input.v1"
LEDGER = "blueprint.site-screen.ledger.v1"
SUMMARY = "blueprint.site-screen.summary.v1"
STAGES = ("screen",)
API_HOST, RUNS_PATH = "api.parallel.ai", "/v1/tasks/runs"
API_KEY_ENV = "PARALLEL_API_KEY"  # The operator reads it; this module only receives the value.
DEFAULT_PROCESSOR = "core"
PRICES_USD = {"core": Decimal("0.025")}  # Per completed run. Add a processor only with its published price.
MAX_CEILING_USD = Decimal("100")  # A typo guard on --ceiling-usd; the owner's ceiling is the control.
MAX_RUNS = 5000  # Per stage and out dir: the site universe export's row bound.
REQUEST_TIMEOUT_SECONDS = 60  # Per socket operation. Nothing is retried.
RESULT_WAIT_SECONDS = 30  # The result endpoint's own long-poll bound; it is read only after completion.
COLLECT_WAIT_SECONDS, POLL_SECONDS, MAX_WAIT_SECONDS = 1800, 15, 7200
PAGE_READ_SECONDS = 45  # The search.bounded_request alarm for one page read, redirects included.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
FRESH_DAYS = 548  # The forms ask for sources from the last 18 months.
NEAR_EXACT_CHARACTERS, NEAR_EXACT_SHARE = 40, 0.9
GOVERNMENT_SOURCES = frozenset({"epa_frs", "osha_ita"})  # Site universe source ids of EPA FRS and OSHA ITA.
EXCLUDED_DISPOSITIONS = frozenset({"duplicate", "learning", "rejected"})  # Never new opportunities.
INVENTORY_VERSION = "blueprint.discovery-inventory.v1"  # discovery.INVENTORY_VERSION
INVENTORY_FIELDS = ("operator", "site", "location", "task_hypothesis", "source_urls")
PROVEN = frozenset({"verified_on_page", "in_citation_excerpt"})
STATUSES = frozenset({"queued", "action_required", "running", "completed", "failed", "cancelling", "cancelled"})
TERMINAL = frozenset({"completed", "failed", "cancelled"})
TIERS = ("outreach_ready", "screened")
SHA = re.compile(r"[0-9a-f]{64}")
ATTEMPT = re.compile(r"[0-9a-f]{16}")
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
API_KEY = re.compile(r"[\x21-\x7e]{8,512}")
SOURCE_ID = re.compile(r"[a-z0-9_]{1,64}")
CODE = re.compile(r"[a-z][a-z0-9_]{2,80}")
CHOICE = re.compile(r"(yes|no|unknown)\b")
DAY = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")
SINGLE_QUOTES = re.compile("[\u2018\u2019\u201b']")
DOUBLE_QUOTES = re.compile('[\u201c\u201d\u201f"]')
DASHES = re.compile("[\u2010-\u2015-]")
WHITESPACE = re.compile(r"\s+")
CODE_ROOT = Path(__file__).resolve().parents[2]  # The repository, or the release directory holding this copy.


def canonical(value):
    """Sorted, compact JSON with non-ASCII escaped: the ledger and command output format."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _s(description):
    return {"type": "string", "description": description}


def _form(description, properties):
    return {"type": "object", "description": description, "properties": properties,
            "required": list(properties), "additionalProperties": False}


ANSWER = "One of: yes, no, unknown. Use unknown unless a source supports yes or no."
URL = "The single best public source URL supporting this answer, or empty when unknown."
QUOTE = "An exact sentence copied verbatim from that source supporting the answer, or empty."
DATE = "The source's publication or update date as YYYY-MM-DD if shown, else empty."
PRIMARY = ("Use a primary source that owns the fact: the operator's own website, careers or job pages, a "
           "government record, or the operator's own press release or filing. Never use a directory, data broker, "
           "map listing or aggregator.")
# The pilot's form, with the operator and the exact site added as quoted primary-source answers.
SCREEN_SCHEMA = _form(
    "Research ONE specific physical site (not the company in general) for a fixed-arm robot design partnership. "
    "Answer only from public sources about this exact site or its operator, quote sources exactly, and prefer "
    "sources from the last 18 months.",
    {"website": _s("The operator's official website URL, or empty."),
     "operator_identity": _s("The name of the company that operates this exact site, as a primary source states it, "
                             "or empty. " + PRIMARY),
     "operator_identity_url": _s("The primary source URL that names this operator at this site, or empty."),
     "operator_identity_quote": _s("An exact sentence copied verbatim from that source that names the operator at "
                                   "this site, or empty."),
     "site_identity": _s("The exact street address of this physical site (not a headquarters elsewhere), as a "
                         "primary source states it, or empty. " + PRIMARY),
     "site_identity_url": _s("The primary source URL that states this address, or empty."),
     "site_identity_quote": _s("An exact sentence copied verbatim from that source that states this address, "
                               "or empty."),
     "operating_now": _s("Is this site operating now? " + ANSWER),
     "operating_now_url": _s(URL), "operating_now_quote": _s(QUOTE), "operating_now_date": _s(DATE),
     "target_task": _s("Short name of a repetitive physical task done at THIS site that a fixed robot arm could take "
                       "on, for example CNC machine tending, molding press unloading, case palletizing or kitting. "
                       "Empty if none found."),
     "target_task_found": _s("Is that task evidenced at this site? " + ANSWER),
     "target_task_url": _s(URL), "target_task_quote": _s(QUOTE), "target_task_date": _s(DATE),
     "manual_today": _s("Do people do this task by hand at this site now, for example a job post for operators, "
                        "loaders or packers listing that duty? " + ANSWER),
     "manual_today_url": _s(URL), "manual_today_quote": _s(QUOTE), "manual_today_date": _s(DATE),
     "existing_automation": _s("Is there public evidence this site already uses robots or automation for this task, "
                               "for example a vendor case study, press release or robot technician job post? "
                               + ANSWER),
     "existing_automation_url": _s(URL), "existing_automation_quote": _s(QUOTE),
     "existing_automation_date": _s(DATE),
     "notes": _s("One or two sentences on what could not be established.")})
FORMS = {stage: {"version": version, "json_schema": schema,
                 "sha256": hashlib.sha256(canonical(schema).encode()).hexdigest()}
         for stage, version, schema in (("screen", SCREEN, SCREEN_SCHEMA),)}
# Quoted screen answers: name -> (answer field, date field). Each also has <name>_url and <name>_quote.
SCREEN_PROOFS = {"operator_identity": ("operator_identity", None), "site_identity": ("site_identity", None),
                 "operating_now": ("operating_now", "operating_now_date"),
                 "target_task": ("target_task_found", "target_task_date"),
                 "manual_today": ("manual_today", "manual_today_date"),
                 "existing_automation": ("existing_automation", "existing_automation_date")}
SCREEN_QUESTIONS = {
    "manual_workflow": "Do people still do this task by hand at the site today?",
    "existing_automation": "Does the site already use robots or other automation for this task?",
    "freshness": "Is the site operating now, and is the task evidence still current?",
    "fit": "Do the parts, cycle times and cell layout suit a fixed robot arm?",
    "interest": "Would the site take part in a fixed-arm robot design partnership?",
}


class ScreenError(ValueError):
    """A stable site_screen_* code; never upstream text or a credential."""


class ProviderRefused(ScreenError):
    """The provider refused a create, or was never reached: no run exists, so the site may be tried again."""

    def __init__(self, code, *, status=None, stop=False):
        super().__init__(code)
        self.status, self.stop = status, stop


class OutcomeUnknown(ScreenError):
    """A create may have reached the provider; its site is never submitted again."""


class TransportError(ScreenError):
    """One request failed below HTTP. ``sent`` is False only when no request byte left this host."""

    def __init__(self, code, *, sent):
        super().__init__(code)
        self.sent = sent


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError, RecursionError):  # UnicodeDecodeError is a ValueError.
        return None


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


# --- provider client ------------------------------------------------------------------------
def https(method, path, *, headers, body=None, timeout=REQUEST_TIMEOUT_SECONDS):
    """One HTTPS request to the provider host; no redirect, retry or proxy. Returns (status, bytes)."""
    connection = http.client.HTTPSConnection(API_HOST, 443, timeout=timeout, context=ssl.create_default_context())
    try:
        try:
            connection.connect()
        except (OSError, http.client.HTTPException):
            raise TransportError("site_screen_provider_unreachable", sent=False) from None
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raise TransportError("site_screen_provider_connection_lost", sent=True) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TransportError("site_screen_provider_response_too_large", sent=True)
        return response.status, raw
    finally:
        connection.close()


# Documented create refusals that stop a run, because every later create would meet the same answer.
# A 5xx answer proves nothing about the run, so it is an unknown outcome.
STOPPING = {401: "site_screen_provider_auth_refused", 402: "site_screen_provider_credit_exhausted",
            403: "site_screen_processor_refused", 429: "site_screen_provider_rate_limited"}


class TaskClient:
    """Create, status and result calls with one key, which only this client's request headers hold."""

    def __init__(self, api_key, *, transport=https, timeout=REQUEST_TIMEOUT_SECONDS):
        if not isinstance(api_key, str) or not API_KEY.fullmatch(api_key):
            raise ScreenError("site_screen_api_key_invalid")
        self._headers = {"x-api-key": api_key, "Accept": "application/json", "User-Agent": "BlueprintSiteScreen/1"}
        self._transport, self._timeout = transport, timeout

    def __repr__(self):
        return "TaskClient(<key withheld>)"

    def _send(self, method, path, body=None):
        headers = dict(self._headers, **({"Content-Type": "application/json"} if body is not None else {}))
        return self._transport(method, path, headers=headers, body=body, timeout=self._timeout)

    def create(self, body):
        """Create one run. ProviderRefused means no run exists; OutcomeUnknown means one may."""
        try:
            status, raw = self._send("POST", RUNS_PATH, canonical(body).encode())
        except TransportError as error:
            if error.sent:
                raise OutcomeUnknown("site_screen_create_outcome_unknown") from None
            raise ProviderRefused(str(error), stop=True) from None
        if 200 <= status < 300:
            run = _json(raw)
            if isinstance(run, dict) and isinstance(run.get("run_id"), str) and RUN_ID.fullmatch(run["run_id"]):
                return run
            raise OutcomeUnknown("site_screen_create_response_invalid")
        if status in STOPPING:
            raise ProviderRefused(STOPPING[status], status=status, stop=True)
        if 300 <= status < 400:  # Never followed: the key would travel with the redirect.
            raise ProviderRefused("site_screen_provider_redirect_refused", status=status, stop=True)
        if 400 <= status < 500:
            raise ProviderRefused("site_screen_create_rejected", status=status)
        raise OutcomeUnknown("site_screen_create_outcome_unknown")

    def _read(self, path, unavailable):
        try:
            status, raw = self._send("GET", path)
        except TransportError:
            raise ScreenError(unavailable) from None
        if status == 401:
            raise ScreenError("site_screen_provider_auth_refused")
        if status == 404:
            raise ScreenError("site_screen_run_not_found")
        value = _json(raw) if status == 200 else None
        if not isinstance(value, dict):
            raise ScreenError(unavailable)
        return value, raw

    def status(self, run_id):
        """The run as the provider reports it now, and the response bytes. Status reads are not billed."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ScreenError("site_screen_run_id_invalid")
        run, raw = self._read(f"{RUNS_PATH}/{run_id}", "site_screen_status_unavailable")
        if run.get("run_id") != run_id or run.get("status") not in STATUSES:
            raise ScreenError("site_screen_status_invalid")
        return run, raw

    def result(self, run_id):
        """The completed run's result, bound to that run, and the response bytes."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ScreenError("site_screen_run_id_invalid")
        value, raw = self._read(f"{RUNS_PATH}/{run_id}/result?timeout={RESULT_WAIT_SECONDS}",
                                "site_screen_result_unavailable")
        run = value.get("run")
        if (not isinstance(run, dict) or run.get("run_id") != run_id or run.get("status") != "completed"
                or not isinstance(value.get("output"), dict)):
            raise ScreenError("site_screen_result_invalid")
        return value, raw


def create_body(stage, site, processor):
    """The exact create request: the site's task input, its stable key as metadata and the stage's form."""
    form = FORMS[stage]
    return {"processor": processor, "input": site["task_input"],
            "metadata": {"site_key": site["site_key"], "form": form["version"]},
            "task_spec": {"output_schema": {"type": "json", "json_schema": form["json_schema"]}}}


# --- inputs ---------------------------------------------------------------------------------
def _clean(value, limit=2000):
    """A single-spaced non-empty string, or None for null or blank; any other value is invalid input."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ScreenError("site_screen_input_record_invalid")
    return " ".join(value.split()) or None


def _compact(value):
    return {key: item for key, item in value.items() if item not in (None, "", [])}


def _words(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value or "").lower()))


def _public_url(value):
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
    except (TypeError, ValueError, AttributeError):
        return False
    return (isinstance(value, str) and len(value) <= 2000 and parts.scheme in {"http", "https"} and "." in host
            and not parts.username and not parts.password)


def from_inventory(record):
    """A web-found site from one daily-run discovery inventory record. Its operator and exact site must
    still be proven by the screen's own quotes."""
    if not isinstance(record, dict) or not set(INVENTORY_FIELDS) <= set(record):
        raise ScreenError("site_screen_input_record_invalid")
    disposition = record.get("disposition")
    if isinstance(disposition, str) and disposition in EXCLUDED_DISPOSITIONS:
        raise ScreenError("site_screen_input_" + disposition)
    operator, site, location, task = (_clean(record[field]) for field in INVENTORY_FIELDS[:4])
    urls = record["source_urls"]
    if not isinstance(urls, list) or len(urls) > 12 or not all(_public_url(url) for url in urls):
        raise ScreenError("site_screen_input_source_urls_invalid")
    if not (operator or site):
        raise ScreenError("site_screen_input_identity_missing")
    if not location:
        raise ScreenError("site_screen_input_location_missing")
    # A record about a site universe site shares that site's key, so the site is never screened twice.
    universe = record.get("site_universe_id")
    key = universe if isinstance(universe, str) and SHA.fullmatch(universe) else _sha256(
        canonical(["discovery_inventory", _words(operator), _words(site), _words(location)]).encode())
    return {"schema_version": INPUT, "site_key": key, "origin": "discovery_inventory", "identity": {},
            "task_input": _compact({"site_name": site, "operator": operator, "location": location,
                                    "task_hint": task, "known_source_urls": list(urls)})}


def from_site_universe(row):
    """A site from one site universe export row. With an OSHA ITA or EPA FRS source, that government record
    is the primary source for the operator and, when it gives a street, the exact site."""
    if not isinstance(row, dict) or not isinstance(row.get("site_id"), str) or not SHA.fullmatch(row["site_id"]):
        raise ScreenError("site_screen_input_record_invalid")
    name, operator, street, city, state, postal, naics, lead = (_clean(row.get(field), 500) for field in (
        "name", "operator", "street", "city", "state", "postal_code", "naics", "lead_capability"))
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and SOURCE_ID.fullmatch(source) for source in sources):
        raise ScreenError("site_screen_input_record_invalid")
    if not (name or operator):
        raise ScreenError("site_screen_input_identity_missing")
    address = ", ".join(part for part in (street, city, " ".join(part for part in (state, postal) if part)) if part)
    if not address:
        raise ScreenError("site_screen_input_location_missing")
    identity = {}
    if GOVERNMENT_SOURCES & set(sources):
        record = {"source": "government_record", "source_ids": sorted(set(sources)), "site_id": row["site_id"]}
        identity["operator"] = {**record, "answer": operator or name}
        if street and city and state:
            identity["physical_site"] = {**record, "answer": address}
    return {"schema_version": INPUT, "site_key": row["site_id"], "origin": "site_universe", "identity": identity,
            "task_input": _compact({"site_name": name, "operator": operator, "location": address,
                                    "task_hint": lead.replace("_", " ") if lead else None, "naics": naics})}


def load_sites(raw):
    """The site inputs of one input file, in file order, and refusal counts by code.

    The file is a site universe export (``backlog.v1.json.gz``, checked by its runtime loader), a
    discovery inventory page, or a JSON list of inventory records and export rows. A site listed twice
    is refused the second time."""
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ScreenError("site_screen_input_invalid")
    if len(raw) > MAX_INPUT_BYTES:
        raise ScreenError("site_screen_input_too_large")
    if raw[:2] == b"\x1f\x8b":
        from tools.daily_research import site_universe  # Standard library only, like this module.
        try:
            rows = site_universe.load_export(bytes(raw))["rows"]
        except site_universe.SiteUniverseError:
            raise ScreenError("site_screen_input_export_invalid") from None
        records = [(from_site_universe, row) for row in rows]
    else:
        document = _json(raw)
        if (isinstance(document, dict) and document.get("version") == INVENTORY_VERSION
                and isinstance(document.get("records"), list)):
            records = [(from_inventory, record) for record in document["records"]]
        elif isinstance(document, list):
            records = [(from_site_universe if isinstance(item, dict) and "site_id" in item else from_inventory, item)
                       for item in document]
        else:
            raise ScreenError("site_screen_input_format_unknown")
    sites, refused, seen = [], Counter(), set()
    for build, record in records:
        try:
            site = build(record)
        except ScreenError as error:
            refused[str(error)] += 1
            continue
        if site["site_key"] in seen:
            refused["site_screen_input_duplicate_site"] += 1
            continue
        seen.add(site["site_key"])
        sites.append(site)
    return sites, refused


def read_input(path):
    """The bytes of one input file, bounded."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
    except OSError:
        raise ScreenError("site_screen_input_unreadable") from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ScreenError("site_screen_input_too_large")
    return raw


def price_of(processor):
    """The reviewed per-run price; a processor without one is refused."""
    if processor not in PRICES_USD:
        raise ScreenError("site_screen_processor_price_unknown")
    return PRICES_USD[processor]


def plan(raw, *, processor=DEFAULT_PROCESSOR):
    """What one input file holds and what screening all of it would cost. Reads nothing else."""
    sites, refused = load_sites(raw)
    price = price_of(processor)
    return {"command": "plan", "state": "planned", "form": SCREEN, "sites": len(sites),
            "by_origin": dict(Counter(site["origin"] for site in sites)),
            "government_record": {name: sum(name in site["identity"] for site in sites)
                                  for name in ("operator", "physical_site")},
            "input_refused": dict(refused), "processor": processor, "price_usd": str(price),
            "estimated_cost_usd": str(price * len(sites)), "provider_calls": 0}


# --- out dir, ledger and spend admission ----------------------------------------------------
def guard_out_dir(path, code_root=CODE_ROOT):
    """The resolved out dir. Outputs hold prospect data, so one inside this code tree or any Git work tree
    is refused."""
    resolved = Path(path).expanduser().resolve()
    root = Path(code_root).resolve()
    if resolved == root or root in resolved.parents or any(
            (folder / ".git").exists() for folder in (resolved, *resolved.parents)):
        raise ScreenError("site_screen_out_dir_inside_repository")
    return resolved


def _write_once(path, data):
    """Create ``path`` (mode 0600) holding ``data``, atomically. An existing file is kept."""
    if path.exists():
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


EVENT_FIELDS = {"intent": ("attempt", "processor", "price_usd", "ceiling_usd", "max_runs", "form_sha256", "input"),
                "created": ("attempt", "run_id", "status"), "refused": ("attempt", "code"),
                "uncertain": ("attempt", "code"), "observed": ("run_id", "status", "result_sha256"),
                "sealed": ("line", "sha256")}


def _check_event(event, stage):
    """One ledger line's shape; anything else is damage."""
    kind = event.get("event") if isinstance(event, dict) else None
    if (kind not in EVENT_FIELDS or event.get("schema_version") != LEDGER or event.get("stage") != stage
            or not all(field in event for field in EVENT_FIELDS[kind])):
        raise ScreenError("site_screen_ledger_invalid")
    if kind == "sealed":
        valid = type(event["line"]) is int and isinstance(event["sha256"], str) and SHA.fullmatch(event["sha256"])
    else:
        valid = (isinstance(event.get("site_key"), str) and SHA.fullmatch(event["site_key"])
                 and ("attempt" not in event or isinstance(event["attempt"], str) and ATTEMPT.fullmatch(event["attempt"]))
                 and ("run_id" not in event or isinstance(event["run_id"], str) and RUN_ID.fullmatch(event["run_id"]))
                 and ("status" not in event or event["status"] is None or event["status"] in STATUSES))
        if valid and kind == "intent":
            try:
                valid = (Decimal(event["price_usd"]).is_finite() and isinstance(event["input"], dict)
                         and event["input"].get("site_key") == event["site_key"])
            except (InvalidOperation, TypeError, ValueError):
                valid = False
    if not valid:
        raise ScreenError("site_screen_ledger_invalid")


class Ledger:
    """One stage's append-only JSONL run ledger. Each line is written with one write and fsynced."""

    def __init__(self, path, stage):
        self.path, self.stage = Path(path), stage

    def events(self):
        """Every event in order. A torn final line (a write cut short by a crash) is skipped: no step after
        it ran. Any other damaged line refuses, unless the line after it seals it."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return []
        lines = raw.split(b"\n")[:-1]  # The last item is b"" or a torn tail.
        parsed = []
        for line in lines:
            try:
                event = json.loads(line)
                _check_event(event, self.stage)
            except (ValueError, ScreenError):
                event = None
            parsed.append(event)
        events = []
        for number, (line, event) in enumerate(zip(lines, parsed), 1):
            seal = parsed[number] if number < len(parsed) else None
            if event is None and not (seal and seal["event"] == "sealed" and seal["line"] == number
                                      and seal["sha256"] == _sha256(line)):
                raise ScreenError("site_screen_ledger_invalid")
            if event is not None and event["event"] != "sealed":
                events.append(event)
        return events

    def append(self, event):
        """Append one event and fsync it. A torn tail left by a crash is first sealed on its own line, so it
        can never merge with a new line."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = (canonical(event) + "\n").encode()
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                raw = os.pread(fd, size, 0)
                seal = {"schema_version": LEDGER, "stage": self.stage, "event": "sealed", "line": raw.count(b"\n") + 1,
                        "sha256": _sha256(raw[raw.rfind(b"\n") + 1:]), "recorded_at": _now()}
                data = b"\n" + (canonical(seal) + "\n").encode() + data
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)


def _event(stage, kind, key, **fields):
    return {"schema_version": LEDGER, "stage": stage, "event": kind, "site_key": key, "recorded_at": _now(), **fields}


def fold(events):
    """Each site's state in one stage's ledger: ``created`` (it has a run id), ``unknown`` (a create that may
    exist: an intent without an answer, or an ambiguous answer) or ``refused`` (no run; it may be retried)."""
    sites = {}
    for event in events:
        kind, key = event["event"], event["site_key"]
        site = sites.get(key)
        if kind == "intent":
            if site is not None and site["state"] != "refused":
                raise ScreenError("site_screen_ledger_invalid")
            sites[key] = {"state": "unknown", "attempt": event["attempt"], "price_usd": event["price_usd"],
                          "processor": event["processor"], "input": event["input"], "answered": False}
        elif kind in ("created", "refused", "uncertain"):
            if site is None or site["attempt"] != event["attempt"] or site["answered"]:
                raise ScreenError("site_screen_ledger_invalid")
            site["answered"] = True
            if kind == "created":
                site.update(state="created", run_id=event["run_id"], status=event["status"], observed=False)
            elif kind == "refused":
                site.update(state="refused", code=event["code"])
            else:
                site["code"] = event["code"]
        elif site is None or site["state"] != "created" or site["run_id"] != event["run_id"] or site["observed"]:
            raise ScreenError("site_screen_ledger_invalid")
        else:
            site.update(status=event["status"], observed=True, result_sha256=event["result_sha256"])
    return sites


def _billable(site):
    return site["state"] == "unknown" or site["state"] == "created" and site.get("status") != "failed"


def committed_usd(sites):
    """The price of every run that may be billed: created and not seen failed, or of unknown outcome."""
    return sum((Decimal(site["price_usd"]) for site in sites.values() if _billable(site)), Decimal("0"))


def run_count(sites):
    return sum(site["state"] in ("created", "unknown") for site in sites.values())


def parse_ceiling(value):
    """The owner's spend ceiling in USD: finite, above 0 and at most MAX_CEILING_USD."""
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ScreenError("site_screen_ceiling_invalid") from None
    if not amount.is_finite() or not Decimal("0") < amount <= MAX_CEILING_USD:
        raise ScreenError("site_screen_ceiling_invalid")
    return amount


def parse_max_runs(value):
    if type(value) is not int or not 1 <= value <= MAX_RUNS:
        raise ScreenError("site_screen_max_runs_invalid")
    return value


def admit(sites, *, price, ceiling, max_runs):
    """Refuse a create that would pass ``max_runs`` or the spend ceiling. Checked before every create."""
    if run_count(sites) + 1 > max_runs:
        raise ScreenError("site_screen_max_runs_reached")
    if committed_usd(sites) + price > ceiling:
        raise ScreenError("site_screen_spend_ceiling_reached")


class Workspace:
    """One out dir: for each stage a ledger, raw provider responses and verified records. Outputs hold
    prospect data, so the out dir is never inside this code tree or any Git work tree."""

    def __init__(self, path, *, create=False, code_root=CODE_ROOT):
        self.root = guard_out_dir(path, code_root)
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ScreenError("site_screen_out_dir_missing")

    def ledger(self, stage):
        return Ledger(self.root / stage / "runs.jsonl", stage)

    def path(self, stage, kind, key):
        return self.root / stage / kind / (key + ".json")

    def records(self, stage):
        folder = self.root / stage / "records"
        return [_json(path.read_bytes()) for path in sorted(folder.glob("*.json"))] if folder.is_dir() else []

    @contextmanager
    def lock(self):
        """Held for a whole command; a second command on the same out dir refuses."""
        import fcntl  # POSIX, like the worker; imported here so this module imports anywhere.
        fd = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ScreenError("site_screen_out_dir_busy") from None
            yield
        finally:
            os.close(fd)


def _submit(workspace, stage, sites, *, client, ceiling_usd, max_runs, processor, apply, input_sha256=None):
    """Create one run per site that has none, in order, each admitted first. Without ``apply`` the same
    admission runs and nothing is sent. The caller holds the out dir lock."""
    price, ceiling, limit = price_of(processor), parse_ceiling(ceiling_usd), parse_max_runs(max_runs)
    if apply and client is None:
        raise ScreenError("site_screen_client_missing")
    ledger = workspace.ledger(stage)
    state = fold(ledger.events())
    counts = Counter({name: 0 for name in ("created", "would_create", "already_created", "outcome_unknown",
                                           "outcome_unknown_kept_out")})
    refused, stop = Counter(), None
    for site in sites:
        key = site["site_key"]
        known = state.get(key, {}).get("state")
        if known in ("created", "unknown"):
            counts["already_created" if known == "created" else "outcome_unknown_kept_out"] += 1
            continue
        try:
            admit(state, price=price, ceiling=ceiling, max_runs=limit)
        except ScreenError as refusal:
            stop = str(refusal)
            break
        if not apply:
            state[key] = {"state": "unknown", "price_usd": str(price)}  # What this create would commit.
            counts["would_create"] += 1
            continue
        attempt = secrets.token_hex(8)
        ledger.append(_event(stage, "intent", key, attempt=attempt, processor=processor, price_usd=str(price),
                             ceiling_usd=str(ceiling), max_runs=limit, form_sha256=FORMS[stage]["sha256"],
                             input=site, input_sha256=input_sha256))
        state[key] = {"state": "unknown", "price_usd": str(price)}
        try:
            run = client.create(create_body(stage, site, processor))
        except ProviderRefused as error:
            ledger.append(_event(stage, "refused", key, attempt=attempt, code=str(error), http_status=error.status))
            state[key]["state"] = "refused"
            refused[str(error)] += 1
            if error.stop:
                stop = str(error)
                break
            continue
        except OutcomeUnknown as error:
            ledger.append(_event(stage, "uncertain", key, attempt=attempt, code=str(error)))
            counts["outcome_unknown"] += 1
            stop = str(error)
            break
        status = run.get("status") if run.get("status") in STATUSES else None
        ledger.append(_event(stage, "created", key, attempt=attempt, run_id=run["run_id"], status=status))
        state[key].update(state="created", status=status)
        counts["created"] += 1
    return {"command": "run", "stage": stage, "form": FORMS[stage]["version"],
            "apply": apply, "state": "stopped" if stop else "complete" if apply else "planned", "stop": stop,
            "sites": len(sites), **counts, "refused": dict(refused), "runs": run_count(state),
            "committed_usd": str(committed_usd(state)), "ceiling_usd": str(ceiling), "max_runs": limit,
            "processor": processor, "price_usd": str(price)}


def run(raw, workspace, *, client, ceiling_usd, max_runs, processor=DEFAULT_PROCESSOR, apply=False):
    """Screen each site of one input file that has no run yet, within the ceiling and ``max_runs``."""
    sites, refused = load_sites(raw)
    with workspace.lock():
        result = _submit(workspace, "screen", sites, client=client, ceiling_usd=ceiling_usd, max_runs=max_runs,
                         processor=processor, apply=apply, input_sha256=_sha256(bytes(raw)))
    return {**result, "input_refused": dict(refused)}


def _stored_status(raw):
    value = _json(raw) or {}
    run = value.get("run") if isinstance(value.get("run"), dict) else value
    return run.get("status") if run.get("status") in TERMINAL else None


def collect(workspace, *, client, wait_seconds=COLLECT_WAIT_SECONDS, poll_seconds=POLL_SECONDS,
            monotonic=time.monotonic, sleep=time.sleep):
    """Store each created run's terminal response once, then record the observation. Reads are not billed;
    a failed read is tried again on the next pass until ``wait_seconds`` ends."""
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
        raise ScreenError("site_screen_wait_invalid")
    observed, read_errors, pending = Counter(), Counter(), {}
    with workspace.lock():
        deadline = monotonic() + wait_seconds
        for stage in STAGES:
            ledger = workspace.ledger(stage)
            for key, site in fold(ledger.events()).items():
                if site["state"] != "created" or site["observed"]:
                    continue
                path = workspace.path(stage, "results", key)
                status = _stored_status(path.read_bytes()) if path.exists() else None
                if status:  # Stored before an interruption; only the observation is missing.
                    ledger.append(_event(stage, "observed", key, run_id=site["run_id"], status=status,
                                         result_sha256=_sha256(path.read_bytes())))
                    observed[f"{stage}_{status}"] += 1
                    continue
                pending[(stage, key)] = site["run_id"]
        while pending:
            for (stage, key), run_id in list(pending.items()):
                try:
                    run_value, raw = client.status(run_id)
                    if run_value["status"] == "completed":
                        _, raw = client.result(run_id)
                    elif run_value["status"] not in TERMINAL:
                        continue
                except ScreenError as error:
                    if str(error) == "site_screen_provider_auth_refused":
                        raise
                    read_errors[str(error)] += 1
                    continue
                _write_once(workspace.path(stage, "results", key), raw)
                workspace.ledger(stage).append(_event(stage, "observed", key, run_id=run_id,
                                                      status=run_value["status"], result_sha256=_sha256(raw)))
                observed[f"{stage}_{run_value['status']}"] += 1
                del pending[(stage, key)]
            if not pending or monotonic() + poll_seconds > deadline:
                break
            sleep(poll_seconds)
    return {"command": "collect", "state": "pending" if pending else "complete", "observed": dict(observed),
            "still_running": len(pending), "read_errors": dict(read_errors)}


# --- verification ---------------------------------------------------------------------------
def normalize(text):
    """The pilot's quote normalization: NFKC, lower case, plain quotes and dashes, single spaces."""
    text = unicodedata.normalize("NFKC", text if isinstance(text, str) else "").lower()
    text = DASHES.sub("-", DOUBLE_QUOTES.sub('"', SINGLE_QUOTES.sub("'", text)))
    return WHITESPACE.sub(" ", text).strip()


def contains(quote, text):
    """True when the normalized quote is in the normalized text, or, for a quote of 40 or more characters,
    its leading or trailing 90 % is."""
    if not quote or not text:
        return False
    if quote in text:
        return True
    span = math.ceil(len(quote) * NEAR_EXACT_SHARE)
    return len(quote) >= NEAR_EXACT_CHARACTERS and (quote[:span] in text or quote[-span:] in text)


def _host(url):
    """The host of an http(s) URL (a bare host is read as https), or None."""
    try:
        parts = urlsplit(url if "://" in url else "https://" + url)
        host = (parts.hostname or "").rstrip(".")
    except (TypeError, ValueError, AttributeError):
        return None
    return host if parts.scheme in {"http", "https"} and host else None


def public_page_reader(seconds=PAGE_READ_SECONDS):
    """The daily agent's own public-page reader, ``search.source``, under its wall-time alarm."""
    if threading.current_thread() is not threading.main_thread():
        raise ScreenError("site_screen_page_reader_needs_main_thread")  # The alarm is a main-thread signal.
    from tools.daily_research import search  # Standard library only, like this module.

    def read(url):
        with search.bounded_request(seconds):
            page = search.source({"url": url})
        return {"text": page["text"]}
    return read


class Pages:
    """The page reads of one verify pass: each URL is read once."""

    def __init__(self, reader):
        self.reader, self.cache, self.reads = reader, {}, 0

    def __call__(self, url):
        if url not in self.cache:
            self.cache[url] = self._read(url)
        return self.cache[url]

    def _read(self, url):
        if _host(url) is None:
            return "unreachable", "site_screen_page_url_invalid"
        self.reads += 1
        try:
            page = self.reader(url)
        except Exception as error:  # noqa: BLE001 - a failed read is an observation, kept as a stable code
            code = str(error)
            return "unreachable", code if CODE.fullmatch(code) else "site_screen_page_read_failed"
        if not isinstance(page, dict) or not isinstance(page.get("text"), str):
            return "unreachable", "site_screen_page_read_failed"
        return "ok", normalize(page["text"])


def output_of(result):
    """The form answers and field basis of one stored result; anything malformed reads as empty."""
    output = result.get("output") if isinstance(result, dict) else None
    output = output if isinstance(output, dict) else {}
    content = output.get("content")
    if isinstance(content, str):
        content = _json(content)
    basis = output.get("basis")
    return (content if isinstance(content, dict) else {},
            [entry for entry in basis if isinstance(entry, dict)] if isinstance(basis, list) else [])


def excerpts(basis, fields):
    """The normalized citation excerpts of the basis entries for ``fields``. Each is matched on its own."""
    found = []
    for entry in basis:
        citations = entry.get("citations") if entry.get("field") in fields else None
        for citation in citations if isinstance(citations, list) else ():
            if not isinstance(citation, dict):
                continue
            notes = citation.get("excerpts")
            found += [normalize(note) for note in notes if isinstance(note, str) and note.strip()] if isinstance(
                notes, list) else []
    return found


def quote_level(url, quote, notes, pages):
    """One quoted answer's level, and the read's code when our read failed."""
    if not url or not quote:
        return "no_quote", None
    quote = normalize(quote)
    state, text = pages(url)
    if state == "ok" and contains(quote, text):
        return "verified_on_page", None
    if any(contains(quote, note) for note in notes):
        return "in_citation_excerpt", None
    return ("unverified", None) if state == "ok" else ("unverified_page_unreachable", text)


def _string(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _choice(value):
    """yes, no or unknown from a form answer; a blank answer is blank and anything else is other."""
    text = normalize(value)
    match = CHOICE.match(text)
    return match.group(1) if match else "other" if text else "blank"


def _fresh(value, today):
    """True when a form date (YYYY-MM-DD, YYYY-MM or YYYY, read as its first day) is no later than today and
    at most FRESH_DAYS before it."""
    match = DAY.fullmatch((value or "").strip())
    if not match:
        return False
    try:
        day = date(int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1))
    except ValueError:
        return False
    return 0 <= (today - day).days <= FRESH_DAYS


def screen_checks(identity, answers, levels, choices):
    """The owner's outreach-ready rule: every check must pass."""
    def proven(name, proof):
        if identity.get(name):
            return {"passed": True, "basis": "government_record"}
        if answers[proof] and levels[proof] in PROVEN:
            return {"passed": True, "basis": "web_quote", "level": levels[proof]}
        return {"passed": False, "basis": None, "level": levels[proof]}
    return {"operator": proven("operator", "operator_identity"),
            "physical_site": proven("physical_site", "site_identity"),
            "site_task": {"passed": bool(answers["target_task"]) and choices["target_task"] == "yes"
                          and levels["target_task"] in PROVEN, "level": levels["target_task"]},
            "not_closed": {"passed": choices["operating_now"] in {"yes", "unknown", "blank"},
                           "answer": choices["operating_now"]},
            "no_existing_automation": {"passed": choices["existing_automation"] in {"no", "unknown", "blank"},
                                       "answer": choices["existing_automation"]}}


def screen_questions(answers, levels, choices, today):
    """The checks the outreach-ready rule leaves unproven, as questions for the first email."""
    def proven(proof, answer):
        return choices[proof] == answer and levels[proof] in PROVEN
    fresh = (proven("operating_now", "yes") and _fresh(answers["operating_now_date"], today)
             and _fresh(answers["target_task_date"], today))
    unproven = {"manual_workflow": not proven("manual_today", "yes"),
                "existing_automation": not proven("existing_automation", "no"),
                "freshness": not fresh, "fit": True, "interest": True}
    return [{"check": name, "question": SCREEN_QUESTIONS[name]} for name, open_ in unproven.items() if open_]


def screen_record(site, run_id, result, pages, today):
    """One site's verified screen: input, answers, quote levels, checks, tier and open questions."""
    content, basis = output_of(result)
    answers = {field: _string(content.get(field)) for field in SCREEN_SCHEMA["properties"]}
    verification = {}
    for proof, (answer, dated) in SCREEN_PROOFS.items():
        fields = {proof, answer, proof + "_url", proof + "_quote", *((dated,) if dated else ())}
        level, code = quote_level(answers[proof + "_url"], answers[proof + "_quote"], excerpts(basis, fields), pages)
        verification[proof] = {"level": level, **({"read": code} if code else {})}
    levels = {proof: item["level"] for proof, item in verification.items()}
    choices = {proof: _choice(answers[answer]) for proof, (answer, dated) in SCREEN_PROOFS.items() if dated}
    checks = screen_checks(site["identity"], answers, levels, choices)
    return {"schema_version": SCREEN, "site_key": site["site_key"], "origin": site["origin"],
            "input": site["task_input"], "identity": site["identity"], "run_id": run_id, "answers": answers,
            "verification": verification, "choices": choices, "checks": checks,
            "tier": "outreach_ready" if all(check["passed"] for check in checks.values()) else "screened",
            "open_questions": screen_questions(answers, levels, choices, today), "verified_on": today.isoformat()}


def verify(workspace, *, reader=None, today=None):
    """Write the verified record of each stored completed result once. Counts only."""
    today = today or datetime.now(timezone.utc).date()
    pages = Pages(reader if reader is not None else public_page_reader())
    counts = {stage: Counter({"written": 0, "kept": 0}) for stage in STAGES}
    with workspace.lock():
        for stage in STAGES:
            for key, site in fold(workspace.ledger(stage).events()).items():
                if site.get("status") != "completed" or not site.get("observed"):
                    continue
                path = workspace.path(stage, "records", key)
                if path.exists():
                    counts[stage]["kept"] += 1
                    continue
                result = _json(workspace.path(stage, "results", key).read_bytes())
                record = screen_record(site["input"], site["run_id"], result, pages, today)
                _write_once(path, (json.dumps(record, indent=1, sort_keys=True) + "\n").encode())
                counts[stage]["written"] += 1
                counts[stage][record["tier"]] += 1
    return {"command": "verify", "state": "complete", "screen": dict(counts["screen"]), "page_reads": pages.reads}


# --- summary --------------------------------------------------------------------------------
def _runs(sites):
    observed = Counter(site["status"] for site in sites.values() if site.get("observed"))
    return {"sites": len(sites), "created": sum(site["state"] == "created" for site in sites.values()),
            "outcome_unknown": sum(site["state"] == "unknown" for site in sites.values()),
            "refused_not_created": sum(site["state"] == "refused" for site in sites.values()),
            "completed": observed["completed"], "failed": observed["failed"], "cancelled": observed["cancelled"],
            "in_flight": sum(site["state"] == "created" and not site["observed"] for site in sites.values())}


def _screen_counts(records):
    return {"records": len(records), "tiers": {tier: sum(r["tier"] == tier for r in records) for tier in TIERS},
            "tiers_by_origin": {origin: {tier: sum(r["tier"] == tier for r in records if r["origin"] == origin)
                                         for tier in TIERS} for origin in sorted({r["origin"] for r in records})},
            "fields": {proof: {"answers": dict(Counter(r["choices"].get(proof) or (
                "present" if r["answers"][proof] else "blank") for r in records)),
                "levels": dict(Counter(r["verification"][proof]["level"] for r in records))}
                for proof in SCREEN_PROOFS},
            "checks": {name: sum(r["checks"][name]["passed"] for r in records) for name in (
                "operator", "physical_site", "site_task", "not_closed", "no_existing_automation")},
            "identity_basis": {name: dict(Counter(r["checks"][name]["basis"] or "unproven" for r in records))
                               for name in ("operator", "physical_site")},
            "open_questions": dict(Counter(q["check"] for r in records for q in r["open_questions"]))}


def summary(workspace):
    """Counts by field, check and tier, and cost; also written to summary.json. Never names, addresses or
    quotes."""
    report = {"schema_version": SUMMARY, "command": "summary", "state": "complete"}
    billed = committed = Decimal("0")
    for stage in STAGES:
        sites = fold(workspace.ledger(stage).events())
        records = [record for record in workspace.records(stage) if isinstance(record, dict)]
        stage_billed = sum((Decimal(site["price_usd"]) for site in sites.values()
                            if site.get("observed") and site["status"] == "completed"), Decimal("0"))
        stage_committed = committed_usd(sites)
        billed, committed = billed + stage_billed, committed + stage_committed
        report[stage] = {"form": FORMS[stage]["version"], "runs": _runs(sites),
                         **_screen_counts(records),
                         "estimated_cost_usd": str(stage_billed), "committed_usd": str(stage_committed)}
    report.update(estimated_cost_usd=str(billed), committed_usd=str(committed))
    path = workspace.root / "summary.json"
    path.unlink(missing_ok=True)  # A derived file: each summary replaces the last.
    _write_once(path, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode())
    return report
