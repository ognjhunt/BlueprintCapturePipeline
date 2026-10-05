"""Per-site research line ("site screen") through the Parallel Task API. Standard library only.

Owner decision 2026-10-05, after a successful 50-site pilot: each site gets one Parallel Task run
(processor ``core``, $0.025 per completed run; failed runs are not billed) that fills the versioned
``blueprint.site-screen.v2`` form, with a URL and an exact quote for every answer. A site whose screen
is outreach-ready may then get one run of the ``blueprint.site-contact.v1`` form: the deciding role,
a named person and a published business address (owner decision 2026-10-05,
``owner-decision-contact-sources-20261005.json``). Nothing here sends, drafts or writes a CRM.

Inputs are daily-run discovery inventory records or site universe export rows (``load_sites``); a batch
may add about one calibration site in ten from below the ranked cut (``select_batch``). On a site
universe row with an OSHA ITA or EPA FRS source, that government record is the primary source for the
exact site address only. The operator, and the site of a web-found input, need their own quotes.

Spend. The first ``--apply`` pins the owner's ceiling, run limit and owner reference in
``owner_ceiling.json``, which is created once; a later invocation may only lower them. Every event is
fsynced to the out dir's spend journal (``spend.jsonl``) and then to its stage's ledger, and an
``intent`` (with the code's commit and whether its tree was dirty) precedes every create. Each command
first checks that the pin, the journal and the ledgers agree, and that every kept result and page read
has its run in the ledger, and refuses otherwise. That catches damage to any one of those files at a
time; matching edits to two of them, or the loss of a whole out dir, can still reset spend, so the out
dir must be durable and never edited by hand. A site with a stored run id, or with a create whose
outcome is unknown (an ambiguous answer, or an interruption after the intent), is never submitted again. Before every create, the price of every run in any stage that may be
billed (completed, in flight, cancelled or of unknown outcome) plus the new one must stay at or below
the ceiling, and all runs at or below ``max_runs``; otherwise the create is refused with a stable code.
Only a run observed ``failed`` frees its price. On the daily worker, ``run`` and ``contact`` refuse
until this line uses ``paid_resource_admission`` (see ``operators/README.md``).

Verification reads each cited page once with the daily agent's own reader (``search.source``) under
its wall-time alarm, and keeps the text. It never reads, and never accepts as evidence, a LinkedIn URL,
including one inside an archive or redirect wrapper. A quote of at least five words is
``verified_on_page`` when our read of its own URL holds every word of it, in order and as whole words
(case, spacing and punctuation may differ); else ``in_citation_excerpt`` when a provider excerpt cited
for that same URL does; else ``unverified``, or ``unverified_page_unreachable`` when our read failed.
A proven quote must also name its answer: the operator's name (which must match the input's operator),
a site anchor (``site_anchors``: the street, the city followed by its state, or the site name with the
city; a city alone never counts) or a provider-found address in the input's city, and a word of the task.
The task quote, its page or a same-URL excerpt must name a site anchor; otherwise the task is only
``company_level_task``. A contact email's domain comes only from the page that proved the operator.

A contact email counts only when our own read of its cited page holds its quote and the whole address, on
the operator's own domain and never free mail; the recipient kind comes from the address itself. Every
other address is removed from the stored result and page reads before they are written (``seal_contact``).

Records are recomputed from the stored raw results and page reads under ``SCREEN_RULE`` (``screen_gates``
and ``outreach_tier``, which mirror the in-flight ``verification.outreach_gates`` and ``outreach_tier``),
so a later rule needs no paid re-run, and each derived file carries its rule version in its name.

The API key is given to ``TaskClient`` and held only in its request headers: it is never read from a
module global, logged or written. Failures leave as stable ``site_screen_*`` codes, never upstream text.
"""
import hashlib
import http.client
import json
import os
import re
import secrets
import ssl
import subprocess
import threading
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.daily_research.verification import normalized  # Standard library only, like this module.

SCREEN = "blueprint.site-screen.v2"
CONTACT = "blueprint.site-contact.v1"
INPUT = "blueprint.site-screen.input.v2"
LEDGER = "blueprint.site-screen.ledger.v1"
SUMMARY = "blueprint.site-screen.summary.v2"
OWNER_CEILING = "blueprint.site-screen.owner-ceiling.v1"
EVIDENCE = "blueprint.site-screen.evidence.v1"
# The rules each stage's records are recomputed under; a derived file's name carries its rule version.
SCREEN_RULE = "blueprint.site-screen-rule.v2"
CONTACT_RULE = "blueprint.site-contact-rule.v2"
STAGES = ("screen", "contact")
API_HOST, RUNS_PATH = "api.parallel.ai", "/v1/tasks/runs"
API_KEY_ENV = "PARALLEL_API_KEY"  # The operator reads it; this module only receives the value.
DEFAULT_PROCESSOR = "core"
PRICES_USD = {"core": Decimal("0.025")}  # Per completed run. Add a processor only with its published price.
MAX_CEILING_USD = Decimal(100)  # A typo guard on --ceiling-usd; the owner's ceiling is the control.
MAX_RUNS = 5000  # All stages of one out dir; a site universe export has at most 5,000 rows.
WORKER_FLAG = "BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED"  # Set (to any value) only on the daily worker.
# Storage the system prunes: macOS tmp_cleaner, per-user temporary folders, tmpfs. A lost ledger means paying again.
VOLATILE_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders",
                  "/dev/shm")
REQUEST_TIMEOUT_SECONDS = 60  # Per socket operation. Nothing is retried.
RESULT_WAIT_SECONDS = 30  # The result endpoint's own long-poll bound; it is read only after completion.
COLLECT_WAIT_SECONDS, POLL_SECONDS, MAX_WAIT_SECONDS = 1800, 15, 7200
PAGE_READ_SECONDS = 45  # The search.bounded_request alarm for one page read, redirects included.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
FRESH_DAYS = 548  # The forms ask for sources from the last 18 months.
MIN_QUOTE_WORDS = 5  # A shorter quote proves too little: a few words appear on almost any page.
CALIBRATION_EVERY = 10  # About one batch site in ten is a calibration site from below the ranked cut.
DEFAULT_SEED = "site-screen-calibration-v1"
GOVERNMENT_SOURCES = frozenset({"epa_frs", "osha_ita"})  # Site universe source ids of EPA FRS and OSHA ITA.
EXCLUDED_DISPOSITIONS = frozenset({"duplicate", "learning", "rejected"})  # Never new opportunities.
INVENTORY_VERSION = "blueprint.discovery-inventory.v1"  # discovery.INVENTORY_VERSION
INVENTORY_FIELDS = ("operator", "site", "location", "task_hypothesis", "source_urls")
# LinkedIn's terms ban automated access: never read, never evidence, and excluded from the provider's search.
NEVER_FETCH = ("linkedin.com", "lnkd.in")
SOURCE_POLICY = {"exclude_domains": list(NEVER_FETCH)}  # Task API source_policy; an apex covers its subdomains.
PROVEN = frozenset({"verified_on_page", "in_citation_excerpt"})
STATUSES = frozenset({"queued", "action_required", "running", "completed", "failed", "cancelling", "cancelled"})
TERMINAL = frozenset({"completed", "failed", "cancelled"})
TIERS = ("outreach_ready", "screened")
RECIPIENT_PREFERENCE = ("person_email", "team_inbox", "general_inbox")  # The owner's order; else none.
CHANNELS = (*RECIPIENT_PREFERENCE, "contact_form", "phone", "none")
FREE_MAIL = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "ymail.com", "hotmail.com", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com", "proton.me", "protonmail.com", "gmx.com", "gmx.net", "mail.com",
    "yandex.com", "zoho.com", "fastmail.com", "hey.com", "tutanota.com", "comcast.net", "att.net", "verizon.net",
    "sbcglobal.net", "bellsouth.net", "charter.net", "cox.net", "earthlink.net"})
# Words in an address's local part that name what it reaches (design section 4: a press inbox is general).
TEAM_INBOX = frozenset({"sales", "operations", "ops", "plant", "engineering", "manufacturing", "production",
                        "purchasing", "procurement", "quality", "maintenance", "automation", "innovation", "projects",
                        "service", "orders", "business", "partnerships"})
GENERAL_INBOX = frozenset({"info", "contact", "contacts", "hello", "office", "mail", "enquiries", "enquiry",
                           "inquiries", "inquiry", "general", "admin", "reception", "press", "media", "pr", "news",
                           "communications"})
REFUSED_INBOX = frozenset({"careers", "career", "jobs", "job", "hr", "recruiting", "recruitment", "talent", "hiring",
                           "legal", "privacy", "support", "help", "helpdesk", "noreply", "donotreply", "billing",
                           "accounts", "invoices", "webmaster", "abuse", "security", "unsubscribe"})
REDACTED = "[redacted-email]"
# The lead-verification claims and the three a quote must prove (verification.CLAIMS and PROVEN_FACTS).
CLAIMS = ("operator", "physical_site", "site_task", "human_workflow", "plausible_fit")
PROVEN_FACTS = ("operator", "physical_site", "site_task")
LEGAL_WORDS = frozenset({"inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "co", "corp", "corporation",
                         "company", "plc", "pllc", "the", "and", "of"})
# Words a company or site name shares with many others (distinctive, site_anchors).
GENERIC_COMPANY_WORDS = frozenset({"manufacturing", "mfg", "industries", "industrial", "holdings", "group",
                                   "international", "intl", "usa", "america", "enterprises", "products", "systems",
                                   "solutions", "services", "technologies", "technology", "global", "worldwide"})
GENERIC_SITE_WORDS = frozenset({"plant", "site", "facility", "factory", "warehouse", "main", "location", "campus",
                                "building", "center", "centre", "distribution", "dc", "office", "headquarters", "hq"})
ABBREVIATIONS = frozenset({"st", "ft", "mt", "pt", "dr", "rd", "ave", "blvd", "hwy", "pkwy", "ln", "ct", "ste", "inc",
                           "co", "corp", "ltd", "llc", "no", "jr", "sr", "mr", "mrs", "ms", "dept", "vs", "etc"})
# Hosts whose pages never give the operator's own domain: directories, data brokers, job boards and applicant
# tracking hosts, social networks, maps and encyclopedias (government hosts are refused by suffix).
NOT_OPERATOR_DOMAINS = frozenset({
    "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com", "careerbuilder.com", "simplyhired.com",
    "snagajob.com", "myworkdayjobs.com", "workday.com", "icims.com", "taleo.net", "greenhouse.io", "lever.co",
    "smartrecruiters.com", "jobvite.com", "bamboohr.com", "ultipro.com", "paylocity.com", "adp.com",
    "applytojob.com", "workable.com", "recruitee.com", "breezy.hr", "jazzhr.com", "yelp.com", "yellowpages.com",
    "superpages.com", "manta.com", "dnb.com", "zoominfo.com", "bbb.org", "bizapedia.com", "buzzfile.com",
    "opencorporates.com", "crunchbase.com", "thomasnet.com", "kompass.com", "chamberofcommerce.com",
    "rocketreach.co", "apollo.io", "signalhire.com", "lusha.com", "owler.com", "craft.co", "pitchbook.com",
    "bloomberg.com", "globalspec.com", "industrynet.com", "iqsdirectory.com", "macraesbluebook.com",
    "mapquest.com", "google.com", "bing.com", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "youtube.com", "tiktok.com", "pinterest.com", "reddit.com", "medium.com", "wikipedia.org",
    "prnewswire.com", "businesswire.com", "globenewswire.com", "accesswire.com", "einpresswire.com"})
TASK_STOPWORDS = frozenset({"a", "an", "and", "or", "of", "the", "to", "for", "in", "on", "at", "by", "with", "from"})
# USPS street suffixes, directions and ordinals (tools/site_universe/normalize.py), and city abbreviations.
STREET_WORDS = {"street": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "boulevard": "blvd",
                "highway": "hwy", "parkway": "pkwy", "lane": "ln", "court": "ct", "place": "pl", "circle": "cir",
                "terrace": "ter", "trail": "trl", "square": "sq", "freeway": "fwy", "expressway": "expy",
                "center": "ctr", "route": "rte", "suite": "ste", "north": "n", "south": "s", "east": "e", "west": "w",
                "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "first": "1st",
                "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th", "seventh": "7th",
                "eighth": "8th", "ninth": "9th", "tenth": "10th"}
CITY_WORDS = {"ft": "fort", "mt": "mount", "st": "saint", "pt": "port"}
STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "CT": "connecticut", "DE": "delaware", "DC": "district of columbia", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan",
    "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "PR": "puerto rico", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming"}
STATE_CODES = {name: code for code, name in STATE_NAMES.items()}
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
ATTEMPT = re.compile(r"[0-9a-f]{16}")
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
API_KEY = re.compile(r"[\x21-\x7e]{8,512}")
REFERENCE = re.compile(r"[\x20-\x7e]{1,200}")
SOURCE_ID = re.compile(r"[a-z0-9_]{1,64}")
CODE = re.compile(r"[a-z][a-z0-9_]{2,80}")
DAY = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")
EMAIL = re.compile(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+")
EMAIL_ANY = re.compile(EMAIL.pattern, re.IGNORECASE)
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
QUOTE = ("An exact sentence of at least five words copied verbatim from that source, supporting the answer, or "
         "empty.")
DATE = "The source's publication or update date as YYYY-MM-DD if shown, else empty."
PRIMARY = ("Use a primary source that owns the fact: the operator's own website, careers or job pages, a "
           "government record, or the operator's own press release or filing. Never use a directory, data broker, "
           "map listing or aggregator.")
PERSON_SOURCES = ("Use only a reputable public source: the operator's own pages, a press release, local or trade "
                  "news, a job post, or a conference or association page.")
NO_LINKEDIN = ("Do not use LinkedIn as the source; a LinkedIn profile may only point you to a confirming press "
               "release, news story, job post or company page.")
# The pilot's form, with the operator, the exact site, the facility and its variability added as quoted answers.
SCREEN_SCHEMA = _form(
    "Research ONE specific physical site (not the company in general) for a fixed-arm robot design partnership. "
    "Answer only from public sources about this exact site or its operator, quote sources exactly, and prefer "
    "sources from the last 18 months.",
    {"website": _s("The operator's official website URL, or empty."),
     "operator_identity": _s("The name of the company that runs the physical work at this exact site, as a primary "
                             "source states it, or empty. " + PRIMARY),
     "operator_identity_url": _s("The primary source URL that names this operator at this site, preferably a page "
                                 "on the operator's own website, or empty."),
     "operator_identity_quote": _s("An exact sentence of at least five words copied verbatim from that source that "
                                   "names the operator, or empty."),
     "site_identity": _s("The exact street address of this physical site (not a headquarters elsewhere), as a "
                         "primary source states it, or empty. " + PRIMARY),
     "site_identity_url": _s("The primary source URL that states this address, or empty."),
     "site_identity_quote": _s("An exact sentence of at least five words copied verbatim from that source that "
                               "states this street address, or this site's city and state, or empty."),
     "facility_type": _s("What is at this address. One of: operations (a plant, shop, warehouse, lab or other site "
                         "where physical work is done), office (offices only), mailing (a mailing, registered or "
                         "headquarters address with no operations), unknown."),
     "facility_type_url": _s(URL), "facility_type_quote": _s(QUOTE),
     "facility_operator": _s("Who runs the physical work at this site. One of: self (the company in operator_identity "
                             "runs it for itself), contractor (that company runs it as a contractor or third-party "
                             "provider for another company), tenant (another company runs the work there, or that "
                             "company is a tenant in another company's site), unknown."),
     "facility_operator_url": _s(URL), "facility_operator_quote": _s(QUOTE),
     "operating_now": _s("Is this site operating now? " + ANSWER),
     "operating_now_url": _s(URL), "operating_now_quote": _s(QUOTE), "operating_now_date": _s(DATE),
     "target_task": _s("Short name of a repetitive physical task done at THIS site that a fixed robot arm could take "
                       "on, for example CNC machine tending, molding press unloading, case palletizing or kitting. "
                       "Empty if none found."),
     "target_task_found": _s("Is that task evidenced at this site? " + ANSWER),
     "target_task_url": _s("The best public source URL for this task at THIS site: a job post for work at this site, "
                           "or a page about this site that names its city or street. Empty when unknown."),
     "target_task_quote": _s(QUOTE), "target_task_date": _s(DATE),
     "manual_today": _s("Do people do this task by hand at this site now, for example a job post for operators, "
                        "loaders or packers listing that duty? " + ANSWER),
     "manual_today_url": _s(URL), "manual_today_quote": _s(QUOTE), "manual_today_date": _s(DATE),
     "existing_automation": _s("Does this site already automate this task? One of: full (a source shows this exact "
                               "task at this site fully done by robots or automation), partial (a source shows some of "
                               "this task at this site automated, or automation of other tasks or at other sites of "
                               "the company), no (a source shows this task done without automation), unknown."),
     "existing_automation_url": _s(URL), "existing_automation_quote": _s(QUOTE),
     "existing_automation_date": _s(DATE),
     "variability_signals": _s("A short phrase naming what a source shows about the work at this site: high-mix or "
                               "low-volume work, frequent changeovers, many SKUs or part types, or irregular items. "
                               "Empty when none is found."),
     "variability_signals_url": _s(URL), "variability_signals_quote": _s(QUOTE),
     "notes": _s("One or two sentences on what could not be established.")})
CONTACT_SCHEMA = _form(
    "For ONE specific physical site, find who would decide on a fixed-arm robot pilot for the named task, and a "
    "published business route to reach them. Answer only from public sources, quote them exactly, and never "
    "guess a name or an address.",
    {"decision_role": _s("The role that would decide on a robot pilot for this task at this site, for example plant "
                         "manager, general manager, operations or manufacturing engineering manager, or the owner or "
                         "president at a small firm. For a large multi-site operator, also name the corporate "
                         "automation or innovation team."),
     "person_name": _s("The full name of a named current person in that role, or empty. " + PERSON_SOURCES + " "
                       + NO_LINKEDIN),
     "person_title": _s("That person's title exactly as the source states it, or empty."),
     "person_url": _s("The page that names this person in this role, or empty. " + NO_LINKEDIN),
     "person_quote": _s("An exact sentence copied verbatim from that page that contains the person's full name, or "
                        "empty."),
     "person_date": _s(DATE),
     "email": _s("A business email address that is itself published verbatim on a public page: the person's "
                 "published business address, a team inbox, a press or media contact, or a general business inbox. "
                 "Prefer the person's address, then a team inbox, then a general inbox. Never guess an address or "
                 "derive one from a name pattern; leave it empty when none is published. " + NO_LINKEDIN),
     "email_url": _s("The page that publishes this exact address, or empty. " + NO_LINKEDIN),
     "email_quote": _s("An exact sentence copied verbatim from that page that contains the exact address, or empty."),
     "channel_type": _s("One of: person_email, team_inbox, general_inbox, contact_form, phone, none. Use "
                        "person_email, team_inbox or general_inbox only for the published address in email; "
                        "otherwise contact_form or phone when the operator publishes one, else none."),
     "channel_url": _s("When no email address is published, the URL of the operator's contact form or phone page; "
                       "else empty."),
     "notes": _s("One or two sentences on what could not be established.")})
FORMS = {stage: {"version": version, "json_schema": schema,
                 "sha256": hashlib.sha256(canonical(schema).encode()).hexdigest()}
         for stage, version, schema in (("screen", SCREEN, SCREEN_SCHEMA), ("contact", CONTACT, CONTACT_SCHEMA))}
# Quoted screen answers: name -> (answer field, date field). Each also has <name>_url and <name>_quote.
SCREEN_PROOFS = {"operator_identity": ("operator_identity", None), "site_identity": ("site_identity", None),
                 "facility_type": ("facility_type", None), "facility_operator": ("facility_operator", None),
                 "operating_now": ("operating_now", "operating_now_date"),
                 "target_task": ("target_task_found", "target_task_date"),
                 "manual_today": ("manual_today", "manual_today_date"),
                 "existing_automation": ("existing_automation", "existing_automation_date"),
                 "variability_signals": ("variability_signals", None)}
# Enumerated answers by proof. An answer whose first word is none of these reads as other; an empty one as blank.
CHOICES = {"operating_now": ("yes", "no", "unknown"), "target_task": ("yes", "no", "unknown"),
           "manual_today": ("yes", "no", "unknown"), "existing_automation": ("full", "partial", "no", "unknown"),
           "facility_type": ("operations", "office", "mailing", "unknown"),
           "facility_operator": ("self", "contractor", "tenant", "unknown")}
PERSON_FIELDS = frozenset({"person_name", "person_title", "person_url", "person_quote", "person_date"})
EMAIL_FIELDS = frozenset({"email", "email_url", "email_quote"})
# Exactly one question per site (design v1.1, section 10), the first whose check is open, in this order. A is
# the last: its check, existing automation, is always open. Every other open check is recorded, not asked.
QUESTIONS = (("S", "site_link", "Is {task} done at your {site} site, or somewhere else in the company?"),
             ("M", "manual_workflow",
              "Which parts of {task} at {site} still need people, and what has kept them from being automated?"),
             ("A", "existing_automation", "What has kept the remaining {task} work at {site} from being automated so far?"))
CONTACT_QUESTIONS = {
    "decision_remit": "Does this person decide on a robot pilot for this task at this site? A title does not prove it.",
    "decision_maker": "Who decides on a robot pilot for this task at this site?",
    "person_current": "Does this person still hold this role?",
    "recipient": "Which published business address reaches the decision maker?",
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
    """The exact create request: the site's task input, its stable key as metadata, the stage's form, and a
    source policy that keeps LinkedIn out of the provider's search."""
    form = FORMS[stage]
    return {"processor": processor, "input": site["task_input"],
            "metadata": {"site_key": site["site_key"], "form": form["version"]},
            "source_policy": SOURCE_POLICY,
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
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _public_url(value):
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
    except (TypeError, ValueError, AttributeError):
        return False
    return (isinstance(value, str) and len(value) <= 2000 and parts.scheme in {"http", "https"} and "." in host
            and not parts.username and not parts.password)


def parse_location(text):
    """The street, city and state a free-text US location gives, whichever are present, as in
    '12 Main St, Springfield, IL 62701' or the backlog's 'Springfield, United States; state not individually
    established' (city only). A trailing country is dropped; a note after a semicolon is ignored."""
    parts = [part.strip() for part in (text or "").split(";")[0].split(",") if part.strip()]
    while parts and normalized(parts[-1]) in {"us", "usa", "united states", "united states of america"}:
        parts.pop()
    if not parts:
        return {}
    match = re.fullmatch(r"([A-Za-z .]+?)(?:\s+\d{5}(?:-\d{4})?)?", parts[-1])
    name = match.group(1).strip() if match else ""
    state = name.upper() if name.upper() in STATE_NAMES else STATE_CODES.get(normalized(name))
    if state:
        parts.pop()
    city = parts.pop() if parts else None
    return _compact({"street": ", ".join(parts) or None, "city": city, "state": state})


def from_inventory(record, *, calibration=False):
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
        canonical(["discovery_inventory", normalized(operator or ""), normalized(site or ""),
                   normalized(location)]).encode())
    return {"schema_version": INPUT, "site_key": key, "origin": "discovery_inventory", "calibration": calibration,
            "identity": {}, "address": parse_location(location),
            "task_input": _compact({"site_name": site, "operator": operator, "location": location,
                                    "task_hint": task, "known_source_urls": list(urls)})}


def from_site_universe(row, *, calibration=False):
    """A site from one site universe export row. With an OSHA ITA or EPA FRS source and a street, city and
    state, that government record is the primary source for the exact site address. The operator still needs
    a quote: the export does not say which source gave the name or operator."""
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
    if GOVERNMENT_SOURCES & set(sources) and street and city and state:
        identity["physical_site"] = {"source": "government_record", "source_ids": sorted(set(sources)),
                                     "site_id": row["site_id"], "answer": address}
    return {"schema_version": INPUT, "site_key": row["site_id"], "origin": "site_universe", "calibration": calibration,
            "identity": identity, "address": _compact({"street": street, "city": city, "state": state}),
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


def parse_batch_size(value):
    if type(value) is not int or not 1 <= value <= MAX_RUNS:
        raise ScreenError("site_screen_batch_size_invalid")
    return value


def select_batch(sites, size, seed=DEFAULT_SEED):
    """``size`` sites: the first ones in file (rank) order, with about one in ten taken instead from below that
    cut and flagged ``calibration``, so the ranking's yield can be measured (design v1.1, section 10).

    Calibration sites are those with the lowest sha256(seed:site_key), so a batch is the same for one input,
    size and seed on any host. One follows each nine ranked sites, so a run that stops early keeps the mix.
    When every site fits, the batch is the whole input and has no calibration site."""
    size = parse_batch_size(size)
    if not isinstance(seed, str) or not REFERENCE.fullmatch(seed):
        raise ScreenError("site_screen_seed_invalid")
    if size >= len(sites):
        return [dict(site, calibration=False) for site in sites]
    count = max(1, round(size / CALIBRATION_EVERY)) if size >= CALIBRATION_EVERY else 0
    ranked, rest = sites[:size - count], sites[size - count:]
    picked = sorted(rest, key=lambda site: _sha256(f"{seed}:{site['site_key']}".encode()))[:count]
    calibration = [dict(site, calibration=True) for site in sorted(picked, key=rest.index)]
    batch = []
    for number, site in enumerate(ranked, 1):
        batch.append(dict(site, calibration=False))
        if number % (CALIBRATION_EVERY - 1) == 0 and calibration:
            batch.append(calibration.pop(0))
    return batch + calibration


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


def anchor_counts(sites):
    """How many sites have each kind of input site anchor, and how many have at least one (usable) or none."""
    kinds = [{anchor["kind"] for anchor in site_anchors(site)} for site in sites]
    return {"usable": sum(bool(kind) for kind in kinds), "none": sum(not kind for kind in kinds),
            **{name: sum(name in kind for kind in kinds) for name in ("street", "city_state", "site_name_city")}}


def plan(raw, *, processor=DEFAULT_PROCESSOR, batch_size=None, seed=DEFAULT_SEED):
    """What one input file holds, the batch ``batch_size`` would screen, and its cost. Reads nothing else."""
    sites, refused = load_sites(raw)
    price = price_of(processor)
    batch = select_batch(sites, batch_size, seed) if batch_size is not None else sites
    return {"command": "plan", "state": "planned", "form": SCREEN, "sites": len(sites),
            "by_origin": dict(Counter(site["origin"] for site in sites)),
            "government_record": {"physical_site": sum("physical_site" in site["identity"] for site in sites)},
            "site_anchors": anchor_counts(sites),
            "input_refused": dict(refused), "processor": processor, "price_usd": str(price),
            "batch": {"size": len(batch), "calibration": sum(site["calibration"] for site in batch),
                      "seed": seed if batch_size is not None else None},
            "estimated_cost_usd": str(price * len(batch)), "provider_calls": 0}


# --- out dir, ledger and spend admission ----------------------------------------------------
def guard_out_dir(path, code_root=CODE_ROOT):
    """The resolved out dir. Outputs hold prospect data and the ledgers hold spend, so an out dir inside this
    code tree or any Git work tree, or on storage the system prunes (VOLATILE_ROOTS), is refused."""
    resolved = Path(path).expanduser().resolve()
    root = Path(code_root).resolve()
    if resolved == root or root in resolved.parents or any(
            (folder / ".git").exists() for folder in (resolved, *resolved.parents)):
        raise ScreenError("site_screen_out_dir_inside_repository")
    for volatile in (Path(value).resolve() for value in VOLATILE_ROOTS):
        if resolved == volatile or volatile in resolved.parents:
            raise ScreenError("site_screen_out_dir_volatile")
    return resolved


def code_state(root=CODE_ROOT):
    """The code a run uses, kept in every intent: the Git commit and whether tracked files differ from it, else a
    release's manifest commit (the installer checks its files), else unknown."""
    root = Path(root)
    try:
        if (root / ".git").exists():
            git = ["git", "-C", str(root)]
            commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                                    check=True).stdout.strip()
            dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                   text=True, timeout=10, check=True).stdout.strip()
            if COMMIT.fullmatch(commit):
                return {"commit": commit, "dirty": bool(dirty), "source": "git"}
        manifest = _json((root / "manifest.json").read_bytes()) if (root / "manifest.json").is_file() else None
        if isinstance(manifest, dict) and isinstance(manifest.get("source_commit"), str) and COMMIT.fullmatch(
                manifest["source_commit"]):
            return {"commit": manifest["source_commit"], "dirty": None, "source": "release_manifest"}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "dirty": None, "source": "unknown"}


def refuse_on_worker(environ=None):
    """The worker spends only through paid_resource_admission, which this line does not use yet."""
    if WORKER_FLAG in (os.environ if environ is None else environ):
        raise ScreenError("site_screen_worker_needs_paid_admission")


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
                "sealed": ("line", "sha256"), "pinned": ("pin",)}
PIN_FIELDS = frozenset({"schema_version", "ceiling_usd", "max_runs", "created_at", "owner_reference"})


def parse_reference(value):
    """The owner reference a pin records: printable ASCII, never PENDING."""
    text = value.strip() if isinstance(value, str) else ""
    if not text or not REFERENCE.fullmatch(text) or text.upper().startswith("PENDING"):
        raise ScreenError("site_screen_owner_reference_invalid")
    return text


def _pin_valid(pin):
    try:
        return (isinstance(pin, dict) and set(pin) == PIN_FIELDS and pin["schema_version"] == OWNER_CEILING
                and str(parse_ceiling(pin["ceiling_usd"])) == pin["ceiling_usd"]
                and parse_max_runs(pin["max_runs"]) == pin["max_runs"]
                and parse_reference(pin["owner_reference"]) == pin["owner_reference"]
                and isinstance(pin["created_at"], str))
    except ScreenError:
        return False


def _check_event(event, stage):
    """One line's shape; anything else is damage. ``stage`` None is the spend journal, which holds the pin
    and the events of every stage."""
    kind = event.get("event") if isinstance(event, dict) else None
    if kind not in EVENT_FIELDS or event.get("schema_version") != LEDGER or not all(
            field in event for field in EVENT_FIELDS[kind]):
        raise ScreenError("site_screen_ledger_invalid")
    if kind == "sealed":
        valid = (event.get("stage") == stage and type(event["line"]) is int and isinstance(event["sha256"], str)
                 and SHA.fullmatch(event["sha256"]))
    elif kind == "pinned":
        valid = stage is None and event.get("stage") is None and _pin_valid(event["pin"])
    else:
        valid = (event.get("stage") in STAGES and (stage is None or event["stage"] == stage)
                 and isinstance(event.get("site_key"), str) and SHA.fullmatch(event["site_key"])
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
    """One append-only JSONL file: a stage's run ledger, or (stage None) the out dir's spend journal. Each line
    is written with one write and fsynced."""

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
    return sum((Decimal(site["price_usd"]) for site in sites.values() if _billable(site)), Decimal(0))


def run_count(sites):
    return sum(site["state"] in ("created", "unknown") for site in sites.values())


def parse_ceiling(value):
    """The owner's spend ceiling in USD: finite, above 0 and at most MAX_CEILING_USD."""
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ScreenError("site_screen_ceiling_invalid") from None
    if not amount.is_finite() or not Decimal(0) < amount <= MAX_CEILING_USD:
        raise ScreenError("site_screen_ceiling_invalid")
    return amount


def parse_max_runs(value):
    if type(value) is not int or not 1 <= value <= MAX_RUNS:
        raise ScreenError("site_screen_max_runs_invalid")
    return value


def limits(pin, ceiling_usd, max_runs, owner_reference):
    """This invocation's ceiling, run limit and owner reference. Against a pin they may only be lower."""
    ceiling, limit, reference = parse_ceiling(ceiling_usd), parse_max_runs(max_runs), parse_reference(owner_reference)
    if pin is not None:
        if reference != pin["owner_reference"]:
            raise ScreenError("site_screen_owner_reference_mismatch")
        if ceiling > Decimal(pin["ceiling_usd"]):
            raise ScreenError("site_screen_ceiling_above_pin")
        if limit > pin["max_runs"]:
            raise ScreenError("site_screen_max_runs_above_pin")
    return ceiling, limit, reference


def admit(stages, *, price, ceiling, max_runs):
    """Refuse a create that would pass ``max_runs`` or the spend ceiling, counted over every stage of the out
    dir. Checked before every create."""
    if sum(run_count(sites) for sites in stages) + 1 > max_runs:
        raise ScreenError("site_screen_max_runs_reached")
    if sum((committed_usd(sites) for sites in stages), Decimal(0)) + price > ceiling:
        raise ScreenError("site_screen_spend_ceiling_reached")


def _same(first, second):
    return [canonical(event) for event in first] == [canonical(event) for event in second]


class Workspace:
    """One out dir: the owner's pinned ceiling, the spend journal, and for each stage a ledger, raw provider
    responses and verified records. Outputs hold prospect data and the ledgers hold spend, so the out dir is
    never inside this code tree, any Git work tree or storage the system prunes."""

    def __init__(self, path, *, create=False, code_root=CODE_ROOT):
        self.root = guard_out_dir(path, code_root)
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ScreenError("site_screen_out_dir_missing")
        self.pin_path = self.root / "owner_ceiling.json"

    def ledger(self, stage):
        return Ledger(self.root / stage / "runs.jsonl", stage)

    def journal(self):
        return Ledger(self.root / "spend.jsonl", None)

    def path(self, stage, kind, key):
        return self.root / stage / kind / (key + ".json")

    def record_path(self, stage, key):
        """A derived record's file, named with the rule it was computed under."""
        rule = (SCREEN_RULE if stage == "screen" else CONTACT_RULE).removeprefix("blueprint.")
        return self.root / stage / "records" / f"{key}.{rule}.json"

    def records(self, stage):
        """The derived records written under the stage's current rule."""
        rule = (SCREEN_RULE if stage == "screen" else CONTACT_RULE).removeprefix("blueprint.")
        folder = self.root / stage / "records"
        return [_json(path.read_bytes()) for path in sorted(folder.glob(f"*.{rule}.json"))] if folder.is_dir() else []

    def append(self, stage, event):
        """Fsync one event to the spend journal, then to its stage ledger. A crash between the two leaves the
        journal one event ahead, which ``states`` completes."""
        self.journal().append(event)
        self.ledger(stage).append(event)

    def create_pin(self, ceiling, max_runs, reference):
        """Pin the owner's ceiling, run limit and reference: in the journal first, then owner_ceiling.json."""
        pin = {"schema_version": OWNER_CEILING, "ceiling_usd": str(ceiling), "max_runs": max_runs,
               "created_at": _now(), "owner_reference": reference}
        self.journal().append({"schema_version": LEDGER, "stage": None, "event": "pinned", "pin": pin,
                               "recorded_at": pin["created_at"]})
        _write_once(self.pin_path, (canonical(pin) + "\n").encode())
        return pin

    def _pin(self):
        try:
            pin = _json(self.pin_path.read_bytes())
        except FileNotFoundError:
            return None
        if not _pin_valid(pin):
            raise ScreenError("site_screen_owner_ceiling_invalid")
        return pin

    def _has_outputs(self):
        return any(next((self.root / stage / kind).glob("*.json"), None) for stage in STAGES
                   for kind in ("results", "evidence", "records"))

    def states(self):
        """Each stage's folded ledger and the pin, once the pin, the spend journal and the ledgers agree.

        A missing journal is refused once the out dir has a pin, a ledger or any output; a missing pin once the
        journal holds any run. The one crash window of each two-file write is completed from the journal: the
        pin file after its journal line, and a ledger's last event after its journal copy. Any other
        difference is refused, so a deleted or edited file never resets spend."""
        journal, pin = self.journal().events(), self._pin()
        ledgers = {stage: self.ledger(stage).events() for stage in STAGES}
        if not journal:
            if pin is not None or any(ledgers.values()) or self._has_outputs():
                raise ScreenError("site_screen_spend_journal_missing")
            return {stage: {} for stage in STAGES}, None
        head, body = journal[0], journal[1:]
        if head["event"] != "pinned" or any(event["event"] == "pinned" for event in body):
            raise ScreenError("site_screen_spend_journal_invalid")
        if pin is None:
            if body:
                raise ScreenError("site_screen_owner_ceiling_missing")
            _write_once(self.pin_path, (canonical(head["pin"]) + "\n").encode())
            pin = head["pin"]
        if pin != head["pin"]:
            raise ScreenError("site_screen_owner_ceiling_mismatch")
        for stage in STAGES:
            mine = [event for event in body if event["stage"] == stage]
            if _same(mine, ledgers[stage]):
                continue
            if mine and body[-1] is mine[-1] and _same(mine[:-1], ledgers[stage]):
                self.ledger(stage).append(mine[-1])
                ledgers[stage] = mine
                continue
            raise ScreenError("site_screen_spend_journal_mismatch")
        states = {stage: fold(ledgers[stage]) for stage in STAGES}
        for stage in STAGES:  # A kept result or page read needs its created run, or spend was reset around it.
            for kind in ("results", "evidence"):
                for path in sorted((self.root / stage / kind).glob("*.json")):
                    if states[stage].get(path.stem, {}).get("state") != "created":
                        raise ScreenError("site_screen_spend_journal_mismatch")
        return states, pin

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


def _submit(workspace, stage, sites, states, pin, *, client, owner_reference, ceiling_usd, max_runs, processor,
            apply, input_sha256=None):
    """Create one run per site that has none, in order, each admitted first over every stage. Without
    ``apply`` the same admission runs and nothing is written or sent. The caller holds the out dir lock."""
    price = price_of(processor)
    ceiling, limit, reference = limits(pin, ceiling_usd, max_runs, owner_reference)
    if apply and client is None:
        raise ScreenError("site_screen_client_missing")
    pinned = "pinned" if pin is not None else "created" if apply else "would_create"
    code = code_state() if apply else None
    if apply and pin is None:
        pin = workspace.create_pin(ceiling, limit, reference)
    shown = pin or {"ceiling_usd": str(ceiling), "max_runs": limit, "owner_reference": reference}
    state, others = states[stage], [states[name] for name in STAGES if name != stage]
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
            admit([state, *others], price=price, ceiling=ceiling, max_runs=limit)
        except ScreenError as refusal:
            stop = str(refusal)
            break
        if not apply:
            state[key] = {"state": "unknown", "price_usd": str(price)}  # What this create would commit.
            counts["would_create"] += 1
            continue
        attempt = secrets.token_hex(8)
        workspace.append(stage, _event(stage, "intent", key, attempt=attempt, processor=processor,
                                       price_usd=str(price), ceiling_usd=str(ceiling), max_runs=limit,
                                       form_sha256=FORMS[stage]["sha256"], input=site, input_sha256=input_sha256,
                                       code=code))
        state[key] = {"state": "unknown", "price_usd": str(price)}
        try:
            run = client.create(create_body(stage, site, processor))
        except ProviderRefused as error:
            workspace.append(stage, _event(stage, "refused", key, attempt=attempt, code=str(error),
                                           http_status=error.status))
            state[key]["state"] = "refused"
            refused[str(error)] += 1
            if error.stop:
                stop = str(error)
                break
            continue
        except OutcomeUnknown as error:
            workspace.append(stage, _event(stage, "uncertain", key, attempt=attempt, code=str(error)))
            counts["outcome_unknown"] += 1
            stop = str(error)
            break
        status = run.get("status") if run.get("status") in STATUSES else None
        workspace.append(stage, _event(stage, "created", key, attempt=attempt, run_id=run["run_id"], status=status))
        state[key].update(state="created", status=status)
        counts["created"] += 1
    every = [state, *others]
    return {"command": "run" if stage == "screen" else "contact", "stage": stage, "form": FORMS[stage]["version"],
            "apply": apply, "state": "stopped" if stop else "complete" if apply else "planned", "stop": stop,
            "sites": len(sites), **counts, "refused": dict(refused),
            "runs": sum(run_count(sites) for sites in every),
            "committed_usd": str(sum((committed_usd(sites) for sites in every), Decimal(0))),
            "stage_runs": run_count(state), "stage_committed_usd": str(committed_usd(state)),
            "ceiling_usd": str(ceiling), "max_runs": limit, "processor": processor, "price_usd": str(price), "code": code,
            "pin": {"state": pinned, **{name: shown[name] for name in ("ceiling_usd", "max_runs", "owner_reference")}}}


def run(raw, workspace, *, client, owner_reference, ceiling_usd, max_runs, processor=DEFAULT_PROCESSOR, apply=False,
        batch_size=None, seed=DEFAULT_SEED, environ=None):
    """Screen each site of one input file, or of its ``batch_size`` batch (``select_batch``), that has no run
    yet, within the pinned ceiling and ``max_runs``. A batch never exceeds ``max_runs``."""
    refuse_on_worker(environ)
    sites, refused = load_sites(raw)
    if batch_size is not None:
        if parse_batch_size(batch_size) > parse_max_runs(max_runs):
            raise ScreenError("site_screen_batch_exceeds_max_runs")
        sites = select_batch(sites, batch_size, seed)
    with workspace.lock():
        states, pin = workspace.states()
        result = _submit(workspace, "screen", sites, states, pin, client=client, owner_reference=owner_reference,
                         ceiling_usd=ceiling_usd, max_runs=max_runs, processor=processor, apply=apply,
                         input_sha256=_sha256(bytes(raw)))
    return {**result, "input_refused": dict(refused), "calibration": sum(site["calibration"] for site in sites),
            "batch": {"size": batch_size, "seed": seed} if batch_size is not None else None}


def contact_input(record):
    """What a contact run is told: the proven operator and site, the task, and the website of the domain whose
    page proved the operator (never the bare website answer)."""
    identity, answers, given = record["identity"], record["answers"], record["input"]
    address = (identity.get("physical_site") or {}).get("answer") or answers.get("site_identity")
    domains = record.get("operator_domains") or []
    return _compact({"operator": answers.get("operator_identity") or given.get("operator"),
                     "site_name": given.get("site_name"), "site_address": address, "location": given.get("location"),
                     "target_task": answers.get("target_task"), "website": f"https://{domains[0]}" if domains else None})


def contact_sites(workspace, states):
    """One contact input per outreach-ready screen record under the current rule, recomputed from the stored
    result and page reads, in the order the screen stage submitted them."""
    records = stage_records(workspace, states, "screen")
    return [{"schema_version": INPUT, "site_key": key, "origin": records[key]["origin"],
             "calibration": records[key]["calibration"], "identity": {}, "address": records[key]["address"],
             "operator_domains": records[key]["operator_domains"], "task_input": contact_input(records[key])}
            for key in states["screen"] if key in records and records[key]["tier"] == "outreach_ready"]


def contact(workspace, *, client, owner_reference, ceiling_usd, max_runs, processor=DEFAULT_PROCESSOR, apply=False,
            environ=None):
    """The contact stage: one run per outreach-ready site, under the same pinned ceiling and ``max_runs``."""
    refuse_on_worker(environ)
    with workspace.lock():
        states, pin = workspace.states()
        return _submit(workspace, "contact", contact_sites(workspace, states), states, pin, client=client,
                       owner_reference=owner_reference, ceiling_usd=ceiling_usd, max_runs=max_runs,
                       processor=processor, apply=apply)


def _stored_status(raw):
    value = _json(raw) or {}
    run = value.get("run") if isinstance(value.get("run"), dict) else value
    return run.get("status") if run.get("status") in TERMINAL else None


def collect(workspace, *, client, reader=None, today=None, wait_seconds=COLLECT_WAIT_SECONDS,
            poll_seconds=POLL_SECONDS, monotonic=time.monotonic, sleep=time.sleep):
    """Store each created run's terminal response once, then record the observation. Reads are not billed;
    a failed read is tried again on the next pass until ``wait_seconds`` ends. A completed contact result is
    sealed first (seal_contact): its email is checked on our own read of its page, and every address but a
    verified one is removed before anything is stored."""
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
        raise ScreenError("site_screen_wait_invalid")
    pages, today = Pages(reader), today or datetime.now(timezone.utc).date()
    observed, read_errors, pending = Counter(), Counter(), {}
    with workspace.lock():
        deadline = monotonic() + wait_seconds
        states, _ = workspace.states()
        for stage in STAGES:
            for key, site in states[stage].items():
                if site["state"] != "created" or site["observed"]:
                    continue
                path = workspace.path(stage, "results", key)
                status = _stored_status(path.read_bytes()) if path.exists() else None
                if status:  # Stored before an interruption; only the observation is missing.
                    workspace.append(stage, _event(stage, "observed", key, run_id=site["run_id"], status=status,
                                                   result_sha256=_sha256(path.read_bytes())))
                    observed[f"{stage}_{status}"] += 1
                    continue
                pending[(stage, key)] = site
        while pending:
            for (stage, key), site in list(pending.items()):
                run_id = site["run_id"]
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
                if stage == "contact" and run_value["status"] == "completed":
                    raw = seal_contact(workspace, key, site["input"], raw, pages, today)
                _write_once(workspace.path(stage, "results", key), raw)
                workspace.append(stage, _event(stage, "observed", key, run_id=run_id, status=run_value["status"],
                                               result_sha256=_sha256(raw)))
                observed[f"{stage}_{run_value['status']}"] += 1
                del pending[(stage, key)]
            if not pending or monotonic() + poll_seconds > deadline:
                break
            sleep(poll_seconds)
    return {"command": "collect", "state": "pending" if pending else "complete", "observed": dict(observed),
            "still_running": len(pending), "read_errors": dict(read_errors)}


# --- verification ---------------------------------------------------------------------------
def words(value):
    """Whole-word text, as verification.normalized: NFKC, lower case, letters and digits only, single spaces.
    Case, spacing and punctuation never change a match; any other difference does."""
    return normalized(value) if isinstance(value, str) else ""


def has_phrase(phrase, text):
    """True when the word string ``phrase`` stands in the word string ``text`` as whole words, in order."""
    return bool(phrase) and f" {phrase} " in f" {text} "


def light(text):
    """Text for finding email addresses (``words`` drops the @): NFKC, lower case, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text if isinstance(text, str) else "").lower().split())


def addresses(text):
    """Every email address in the text, as whole tokens: a longer address never matches a shorter one."""
    tokens = EMAIL.findall(light(text))
    return set(tokens) | {token.lstrip("'`") for token in tokens}


def email_address(value):
    """The provider's address, lower case, or None when it is not one plain address."""
    text = light(value)
    text = (text.removeprefix("mailto:")).strip("<> ")
    return text if EMAIL.fullmatch(text) else None


def _host(url):
    """The host of an http(s) URL (a bare host is read as https), or None."""
    try:
        parts = urlsplit(url if "://" in url else "https://" + url)
        host = (parts.hostname or "").rstrip(".")
    except (TypeError, ValueError, AttributeError):
        return None
    return host if parts.scheme in {"http", "https"} and host else None


def never_fetch(url):
    """True for any URL that reaches LinkedIn: a host of linkedin.com, lnkd.in or a subdomain of either, or either
    name anywhere in the URL, as in an archive, translation or redirect wrapper."""
    if not isinstance(url, str):
        return False
    host, text = _host(url) or "", unquote(unquote(url)).lower()
    return any(host == domain or host.endswith("." + domain) or domain in text for domain in NEVER_FETCH)


def url_key(value):
    """Same-URL comparison, as the in-flight verification.url_key: scheme, host case, www., a trailing slash and
    the fragment do not differ."""
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").lower().removeprefix("www.")
        if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
            return None
        return host + parts.path.rstrip("/") + ("?" + parts.query if parts.query else "")
    except (AttributeError, TypeError, ValueError):
        return None


def public_page_reader(seconds=PAGE_READ_SECONDS):
    """The daily agent's own public-page reader, ``search.source``, under its wall-time alarm. It refuses a
    NEVER_FETCH host on the first request and on every redirect, before any connection."""
    if threading.current_thread() is not threading.main_thread():
        raise ScreenError("site_screen_page_reader_needs_main_thread")  # The alarm is a main-thread signal.
    from tools.daily_research import search  # Standard library only, like this module.

    def read(url):
        with search.bounded_request(seconds):
            page = search.source({"url": url}, blocked_domains=NEVER_FETCH)
        return {"text": page["text"]}
    return read


class Pages:
    """The page reads of one command: each URL is read once, a LinkedIn URL never, and the default reader is
    made only when a read is needed."""

    def __init__(self, reader=None):
        self._reader, self.cache, self.reads = reader, {}, 0

    def __call__(self, url):
        if url not in self.cache:
            self.cache[url] = self._read(url)
        return self.cache[url]

    def _read(self, url):
        if _host(url) is None:
            return {"state": "unreachable", "code": "site_screen_page_url_invalid"}
        if never_fetch(url):
            return {"state": "not_allowed", "code": "site_screen_page_not_allowed"}
        if self._reader is None:
            self._reader = public_page_reader()
        self.reads += 1
        try:
            page = self._reader(url)
        except Exception as error:  # noqa: BLE001 - a failed read is an observation, kept as a stable code
            code = str(error)
            return {"state": "unreachable", "code": code if CODE.fullmatch(code) else "site_screen_page_read_failed"}
        if not isinstance(page, dict) or not isinstance(page.get("text"), str):
            return {"state": "unreachable", "code": "site_screen_page_read_failed"}
        return {"state": "ok", "text": page["text"], "sha256": _sha256(page["text"].encode())}


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


def _string(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def cited_urls(stage, content):
    """Every URL a result gives as an answer source."""
    fields = [name + "_url" for name in SCREEN_PROOFS] if stage == "screen" else ["person_url", "email_url"]
    return sorted({_string(content.get(field)) for field in fields} - {""})


def read_evidence(stage, key, result, pages, today):
    """Our reads of every URL the result cites, with their date, kept so the record can be recomputed offline."""
    content, _ = output_of(result)
    return {"schema_version": EVIDENCE, "stage": stage, "site_key": key, "checked_on": today.isoformat(),
            "pages": {url: pages(url) for url in cited_urls(stage, content)}}


def evidence_index(evidence, basis):
    """Text by URL key, as the in-flight verification.evidence_index: our page reads, and the provider's
    citation excerpts, never one cited from LinkedIn. Each entry is (whole-word text, sha256, raw text)."""
    index = {"pages": {}, "excerpts": {}}
    for url, page in (evidence.get("pages") or {}).items():
        key = url_key(url)
        if key and isinstance(page, dict) and page.get("state") == "ok" and isinstance(page.get("text"), str):
            index["pages"].setdefault(key, []).append((words(page["text"]), page.get("sha256"), page["text"]))
    for entry in basis:
        citations = entry.get("citations")
        for citation in citations if isinstance(citations, list) else ():
            url = citation.get("url") if isinstance(citation, dict) else None
            key = url_key(url)
            notes = citation.get("excerpts") if key and not never_fetch(url) else None
            for note in notes if isinstance(notes, list) else ():
                if isinstance(note, str) and note.strip():
                    index["excerpts"].setdefault(key, []).append((words(note), _sha256(note.encode()), note))
    return index


def holding(quote, url, index, kinds=("pages", "excerpts")):
    """(level, sha256, raw text) of the first text kept for the quote's own URL that holds it whole-word."""
    phrase, key = words(quote), url_key(url)
    if key is None or never_fetch(url) or len(phrase.split()) < MIN_QUOTE_WORDS:
        return None, None, None
    for kind in kinds:
        for text, sha, raw in index[kind].get(key, ()):
            if has_phrase(phrase, text):
                return ("verified_on_page" if kind == "pages" else "in_citation_excerpt"), sha, raw
    return None, None, None


def quote_level(quote, url, index):
    """(level, sha256) of a quote proven at its own URL, in the shape of the in-flight verification.quote_level:
    verified_on_page when our read of that URL holds it whole-word, else in_citation_excerpt when a provider
    excerpt cited for that same URL does. (None, None) otherwise, and for a quote of fewer than MIN_QUOTE_WORDS
    words or any LinkedIn URL."""
    level, sha, _ = holding(quote, url, index)
    return level, sha


def proof(quote, url, index, evidence):
    """One quoted answer's verification: its level, the proving text's sha256, or our read's code."""
    if not _string(url) or not _string(quote):
        return {"level": "no_quote"}
    if never_fetch(url):
        return {"level": "source_not_allowed"}
    if len(words(quote).split()) < MIN_QUOTE_WORDS:
        return {"level": "quote_too_short"}
    level, sha = quote_level(quote, url, index)
    if level:
        return {"level": level, "sha256": sha}
    page = (evidence.get("pages") or {}).get(_string(url))
    page = page if isinstance(page, dict) else {}
    if page.get("state") == "ok":
        return {"level": "unverified"}
    return {"level": "unverified_page_unreachable", "read": page.get("code") or "site_screen_page_not_read"}


def _choice(value, allowed):
    """The enumerated answer: its first word when allowed, else other; blank when empty."""
    first = words(value).split()[:1]
    return first[0] if first and first[0] in allowed else "other" if first else "blank"


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


def significant(name):
    """A name's words without legal forms and articles."""
    return [word for word in words(name).split() if word not in LEGAL_WORDS]


def names_operator(quote, operator):
    """True when the quote holds every significant word of the operator's name."""
    tokens, found = significant(operator), set(words(quote).split())
    return bool(tokens) and all(token in found for token in tokens)


def distinctive(name):
    """A company name's distinctive words: its significant words without generic ones (Manufacturing, Holdings),
    or all of its significant words when nothing else is left."""
    tokens = significant(name)
    return [token for token in tokens if token not in GENERIC_COMPANY_WORDS] or tokens


def same_operator(first, second):
    """True when the distinctive words of one name are all in the other."""
    one, other = set(distinctive(first)), set(distinctive(second))
    return bool(one and other) and (one <= other or other <= one)


def names_task(quote, task):
    """True when the quote holds a word of the task phrase other than a stopword."""
    return bool({word for word in words(task).split() if word not in TASK_STOPWORDS} & set(words(quote).split()))


def _canon(text, table):
    return " ".join(table.get(word, word) for word in words(text).split())


def sentences(text):
    """A text's sentences and lines. A period after an abbreviation (St., Inc., Rd.) or an initial ends none."""
    text = unicodedata.normalize("NFKC", text if isinstance(text, str) else "")
    pieces, start = [], 0
    for match in re.finditer(r"\n|[.!?;](?=\s)", text):
        before = re.search(r"([A-Za-z]+)$", text[start:match.start()])
        if match.group(0) == "." and before and (len(before.group(1)) == 1 or before.group(1).lower() in ABBREVIATIONS):
            continue
        pieces.append(text[start:match.end()])
        start = match.end()
    pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def _proper(word):
    """A pattern for one word of a place name: capitalized in the text (case-free after the first letter)."""
    return re.escape(word[:1].upper()) + "(?i:" + re.escape(word[1:]) + ")" if word[:1].isalpha() else re.escape(word)


def city_pattern(city):
    """A place-name pattern for the city: each word capitalized, abbreviations allowed (Ft. for Fort), and not
    glued by a hyphen or letters to its neighbours, so 'e-commerce' and 'our mission' never match."""
    tokens = []
    for word in _canon(city, CITY_WORDS).split():
        forms = [word, *(short for short, full in CITY_WORDS.items() if full == word)]
        tokens.append("(?:" + "|".join(_proper(form) + (r"\.?" if form != word else "") for form in forms) + ")")
    return r"(?<![\w-])" + r"[^\w\n]+".join(tokens) + r"(?![\w-])" if tokens else None


def state_pattern(state):
    """The state's code in capitals (IN, never 'in') or its name capitalized."""
    name = r"[^\w\n]+".join(_proper(word) for word in STATE_NAMES.get(state, "").split())
    return "(?:" + re.escape(state) + (f"|{name}" if name else "") + r")(?![\w-])"


def street_anchor(street):
    """The part of a street that names one building: the comma part that starts with a house number."""
    for part in (street or "").split(","):
        if re.match(r"\s*\d+[A-Za-z]?\s+\S", part):
            return part.strip()
    return None


def site_anchors(site, found=None):
    """What ties a text to this site: a street with a house number, a city followed by its state, and the site
    name with the city in one sentence; from the input address and, when given, from a provider-found address
    that the site quote proved. A city alone never does: it may be a common word."""
    anchors = []
    for address in (site.get("address") or {}, found or {}):
        street, city, state = street_anchor(address.get("street")), address.get("city"), address.get("state")
        if street:
            anchors.append({"kind": "street", "street": street})
        if city and state:
            anchors.append({"kind": "city_state", "city": city, "state": state})
    city = (site.get("address") or {}).get("city")
    place = set(_canon(city, CITY_WORDS).split()) if city else set()
    name = [word for word in significant(site["task_input"].get("site_name") or "")
            if word not in GENERIC_SITE_WORDS and CITY_WORDS.get(word, word) not in place]
    if city and name:
        anchors.append({"kind": "site_name_city", "name": name, "city": city})
    return [anchor for number, anchor in enumerate(anchors) if anchor not in anchors[:number]]


def names_site(text, anchors):
    """The kind of the first anchor the text holds, or None."""
    for anchor in anchors:
        if anchor["kind"] == "street":
            if has_phrase(_canon(anchor["street"], STREET_WORDS), _canon(text, STREET_WORDS)):
                return "street"
        elif anchor["kind"] == "city_state":
            pattern = city_pattern(anchor["city"]) + r"[ \t]*,?[ \t]*" + state_pattern(anchor["state"])
            if any(re.search(pattern, sentence) for sentence in sentences(text)):
                return "city_state"
        elif any(set(anchor["name"]) <= set(words(sentence).split()) and re.search(city_pattern(anchor["city"]), sentence)
                 for sentence in sentences(text)):
            return "site_name_city"
    return None


def found_address(site, answers, verification):
    """The provider-found site address, when the proven site quote itself holds its street and it lies in the
    input's city and state; never when the input has its own street. None otherwise."""
    given = site.get("address") or {}
    if street_anchor(given.get("street")) or verification["site_identity"]["level"] not in PROVEN:
        return None
    found = parse_location(answers["site_identity"])
    street = street_anchor(found.get("street"))
    if not street or not has_phrase(_canon(street, STREET_WORDS), _canon(answers["site_identity_quote"], STREET_WORDS)):
        return None
    if given.get("city") and _canon(given["city"], CITY_WORDS) != _canon(found.get("city") or "", CITY_WORDS):
        return None
    if given.get("state") and found.get("state") and given["state"] != found["state"]:
        return None
    return found


def government_host(host):
    return host.endswith((".gov", ".mil", ".fed.us")) or bool(re.search(r"\.(?:gov|mil)\.[a-z]{2}$|\.state\.[a-z]{2}\.us$",
                                                                       host))


def operator_domains(answers, verification):
    """The domain whose page proves the operator: the operator quote's page, when that quote is proven and names
    the operator. Never the bare website answer, and never a directory, aggregator, job board, social, free-mail
    or government host, or LinkedIn."""
    url = answers["operator_identity_url"]
    if verification["operator_identity"]["level"] not in PROVEN or not names_operator(
            answers["operator_identity_quote"], answers["operator_identity"]):
        return []
    domain, host = site_domain(url), _host(url) or ""
    if (not domain or never_fetch(url) or domain in NOT_OPERATOR_DOMAINS or domain in FREE_MAIL
            or government_host(host)):
        return []
    return [domain]


def site_label(site):
    """The site's name in a question: its city, else the input site name, else the operator."""
    city = (site.get("address") or {}).get("city")
    if city:
        return city.title() if city.isupper() else city
    return site["task_input"].get("site_name") or site["task_input"].get("operator") or ""


def screen_gates(site, answers, choices, verification, index, *, valid):
    """The inputs of the tier, in the shape of the in-flight verification.outreach_gates, from one screen.

    A fact is verified_fact only when its quote is proven at its own URL and names its answer. The operator
    quote holds every significant word of the operator's name, and that name matches the input's operator
    when the input has one. The site quote names a site anchor (site_anchors), or holds the street of a
    provider-found address in the input's city (found_address); an OSHA ITA or EPA FRS record also proves
    the address. The task quote holds a task word, and it, its page or a same-URL excerpt names a site anchor
    or the found address; else the task is company level. A closed site, a proven office or mailing address,
    an operator the input does not name, a proven contractor or tenant site the input attributes to another
    operator, a proven manual 'no', and this task proven fully automated are contradictions."""
    given = site["task_input"]

    def proven(name):
        return verification[name]["level"] in PROVEN

    def proofs(claim, name):
        return [{"claim": claim, "source_id": name, "url": answers[name + "_url"],
                 "quote_sha256": _sha256(answers[name + "_quote"].encode()), "level": verification[name]["level"],
                 "tool_result_sha256": verification[name].get("sha256")}]

    operator = bool(answers["operator_identity"]) and proven("operator_identity") and names_operator(
        answers["operator_identity_quote"], answers["operator_identity"])
    record = site["identity"].get("physical_site")
    found = found_address(site, answers, verification) if answers["site_identity"] else None
    anchors = site_anchors(site, found)
    if record:
        place = [{"claim": "physical_site", "source_id": "government_record", "url": None, "quote_sha256": None,
                  "level": "government_record", "tool_result_sha256": None, "source_ids": record["source_ids"],
                  "site_id": record["site_id"]}]
    else:
        place = proofs("physical_site", "site_identity") if (
            answers["site_identity"] and proven("site_identity")
            and (found or names_site(answers["site_identity_quote"], site_anchors(site)))) else []
    task = (bool(answers["target_task"]) and choices["target_task"] == "yes" and proven("target_task")
            and names_task(answers["target_task_quote"], answers["target_task"]))
    key = url_key(answers["target_task_url"])
    texts = [answers["target_task_quote"]] + [entry[2] for kind in ("pages", "excerpts") for entry in index[kind].get(key, ())]
    scope = ("site" if any(names_site(text, anchors) for text in texts) else "company") if task else None
    attributed = given.get("operator") or (given.get("site_name") if site["origin"] == "site_universe" else None)
    named = bool(given.get("operator") and answers["operator_identity"]) and not same_operator(
        given["operator"], answers["operator_identity"])
    flags = {"closed": choices["operating_now"] in ("no", "other"),
             "office_or_mailing": choices["facility_type"] in ("office", "mailing") and proven("facility_type"),
             "operator_mismatch": named or (choices["facility_operator"] in ("contractor", "tenant")
                                            and proven("facility_operator") and bool(attributed)
                                            and not same_operator(attributed, answers["operator_identity"])),
             "found_address": bool(found)}
    manual = choices["manual_today"] if proven("manual_today") else None
    automation = choices["existing_automation"] if proven("existing_automation") else None
    states = {"operator": "contradicted" if flags["operator_mismatch"] else "verified_fact" if operator else "unresolved",
              "physical_site": ("contradicted" if flags["closed"] or flags["office_or_mailing"]
                                else "verified_fact" if place else "unresolved"),
              "site_task": "verified_fact" if scope == "site" else "unresolved",
              "human_workflow": {"yes": "verified_fact", "no": "contradicted"}.get(manual, "unresolved"),
              "plausible_fit": "unresolved",
              "counterevidence": {"full": "contradicted", "partial": "checked", "no": "checked"}.get(automation,
                                                                                                  "unresolved")}
    facts = {"operator": {"primary_sources_usable": operator,
                          "proofs": proofs("operator", "operator_identity") if operator else []},
             "physical_site": {"primary_sources_usable": bool(place), "proofs": place},
             "site_task": {"primary_sources_usable": scope == "site",
                           "proofs": proofs("site_task", "target_task") if scope == "site" else []}}
    return {"eligible_for_qualified_promotion": False, "assessment_valid": valid, "identity_present": True,
            "duplicate": False, "conflict": False, "valid_until": None, "states": states, "facts": facts,
            "task": answers["target_task"], "site": site_label(site), "site_task_scope": scope, "flags": flags}


def outreach_tier(gates):
    """SCREEN_RULE over screen_gates: the in-flight verification.outreach_tier, with design v1.1's single
    question. Replace this with that function once it lands and bump SCREEN_RULE; records are recomputed.

    outreach_ready: operator, physical_site and site_task are verified_fact with proofs, the result is valid,
    and nothing is contradicted. Anything else, including any defect here, is none. The question is the first
    of S, M and A (QUESTIONS) whose check is open; every other open check is recorded, not asked."""
    block = {"rule_version": SCREEN_RULE, "proving_sources": [], "open_checks": [], "question": None,
             "question_template": None, "blockers": []}
    try:
        facts, states = gates["facts"], gates["states"]
        block["proving_sources"] = [facts[name]["proofs"][0] for name in PROVEN_FACTS if facts[name]["proofs"]]
        blockers = [name + "_contradicted" for name in (*CLAIMS, "counterevidence") if states.get(name) == "contradicted"]
        for failed, code in ((gates["assessment_valid"] is not True, "assessment_invalid"),
                             (gates["identity_present"] is not True, "identity_missing"),
                             (gates["duplicate"] is not False, "duplicate"),
                             (gates["conflict"] is not False, "duplicate_conflict"),
                             (gates["flags"]["operator_mismatch"] is True, "operator_mismatch"),
                             (gates["site_task_scope"] == "company", "company_level_task")):
            if failed:
                blockers.append(code)
        for name in PROVEN_FACTS:
            if states.get(name) != "verified_fact":
                blockers.append(name + "_not_verified_fact")
            elif facts[name]["primary_sources_usable"] is not True:
                blockers.append(name + "_primary_source_unusable")
            elif not facts[name]["proofs"]:
                blockers.append(name + "_quote_unproven")
        task, site = gates["task"], gates["site"]
        if not task or not site:
            blockers.append("question_task_or_site_missing")
        checks = (["site_link"] if gates["site_task_scope"] != "site" else []) + (
            ["manual_workflow"] if states.get("human_workflow") != "verified_fact" else []) + [
            "existing_automation", "freshness", "fit", "interest"]
        block["open_checks"] = checks
        if task and site:
            template, _, text = next(item for item in QUESTIONS if item[1] in checks)
            block.update(question=text.format(task=task, site=site), question_template=template)
        tier = "none" if blockers else "outreach_ready"
        return {"tier": tier, "eligible_for_outreach_ready": tier == "outreach_ready",
                "outreach_ready": {**block, "blockers": list(dict.fromkeys(blockers))}}
    except Exception:  # noqa: BLE001 - any defect in the tier computation yields none
        return {"tier": "none", "eligible_for_outreach_ready": False,
                "outreach_ready": {**block, "proving_sources": [], "blockers": ["tier_computation_unavailable"]}}


def _evidence(evidence_raw):
    evidence = _json(evidence_raw)
    return evidence if isinstance(evidence, dict) else {}


def screen_record(site, run_id, result_raw, evidence_raw):
    """One site's screen under SCREEN_RULE, recomputed from its stored raw result and page reads alone."""
    evidence = _evidence(evidence_raw)
    content, basis = output_of(_json(result_raw))
    answers = {field: _string(content.get(field)) for field in SCREEN_SCHEMA["properties"]}
    index = evidence_index(evidence, basis)
    verification = {name: proof(answers[name + "_quote"], answers[name + "_url"], index, evidence)
                    for name in SCREEN_PROOFS}
    choices = {name: _choice(answers[SCREEN_PROOFS[name][0]], allowed) for name, allowed in CHOICES.items()}
    gates = screen_gates(site, answers, choices, verification, index, valid=bool(content))
    outcome = outreach_tier(gates)
    block = outcome["outreach_ready"]
    return {"schema_version": SCREEN, "rule_version": SCREEN_RULE, "site_key": site["site_key"],
            "origin": site["origin"], "calibration": site.get("calibration") is True, "input": site["task_input"],
            "address": site.get("address") or {}, "identity": site["identity"], "run_id": run_id,
            "result_sha256": _sha256(result_raw), "evidence_sha256": _sha256(evidence_raw),
            "checked_on": evidence.get("checked_on"), "answers": answers, "verification": verification,
            "choices": choices, "task_scope": gates["site_task_scope"],
            "variability": {"answer": answers["variability_signals"] or None,
                            "proven": bool(answers["variability_signals"])
                            and verification["variability_signals"]["level"] in PROVEN},
            "gates": gates, "tier": "outreach_ready" if outcome["tier"] == "outreach_ready" else "screened",
            "blockers": block["blockers"], "proving_sources": block["proving_sources"],
            "open_checks": block["open_checks"], "question": block["question"],
            "question_template": block["question_template"],
            "operator_domains": operator_domains(answers, verification) if gates["states"]["operator"] == "verified_fact"
            else []}


def _person(text, index, evidence, today):
    name, url, quote = text["person_name"], text["person_url"], text["person_quote"]
    if not name:
        return {"verified": False, "level": "no_person"}
    if never_fetch(url):
        return {"verified": False, "level": "person_source_not_allowed"}
    if quote and not has_phrase(words(name), words(quote)):
        return {"verified": False, "level": "unverified", "reason": "site_screen_quote_lacks_name"}
    item = proof(quote, url, index, evidence)
    if item["level"] == "no_quote":
        return {"verified": False, "level": "unverified", "reason": "site_screen_quote_missing"}
    if item["level"] not in PROVEN:
        return {"verified": False, "level": item["level"], **({"reason": item["read"]} if item.get("read") else {})}
    return {"verified": True, "level": item["level"], "name": name, "title": text["person_title"] or None, "url": url,
            "date": text["person_date"] or None, "current": _fresh(text["person_date"], today)}


def site_domain(url):
    """The registrable domain of a URL's host: its last two labels, or three under a two-letter country code's
    second level (co.uk). None without a host."""
    labels = (_host(url) or "").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in {"co", "com", "net", "org", "gov", "ac", "edu"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 and all(labels[-2:]) else None


def email_check(text, site, index, evidence):
    """The contact's published business address, proven only on our own read of its cited page.

    It must parse as one address on the operator's own domain, the one whose page proved the operator in the
    screen (operator_domains), or a subdomain of it, never a free-mail domain; its quote must hold the exact
    address and stand whole-word on that page, where the address must also stand as a whole token. A provider
    excerpt never counts, and LinkedIn never does."""
    raw, url, quote = text["email"], text["email_url"], text["email_quote"]
    if not raw:
        return {"verified": False, "level": "no_email", "discarded": False}
    address, operators = email_address(raw), site.get("operator_domains") or []
    domain = address.rpartition("@")[2] if address else ""
    if address is None:
        level, code = "unverified", "site_screen_email_invalid"
    elif never_fetch(url) or never_fetch(raw):
        level, code = "person_source_not_allowed", None
    elif domain in FREE_MAIL or site_domain("https://" + domain) in FREE_MAIL:
        level, code = "unverified", "site_screen_email_free_mail"
    elif not operators:
        level, code = "unverified", "site_screen_operator_domain_unproven"
    elif not any(domain == operator or domain.endswith("." + operator) for operator in operators):
        level, code = "unverified", "site_screen_email_off_operator_domain"
    elif not _string(quote):
        level, code = "unverified", "site_screen_quote_missing"
    elif address not in addresses(quote):
        level, code = "unverified", "site_screen_quote_lacks_address"
    else:
        level, _, page = holding(quote, url, index, kinds=("pages",))
        if level is None:
            item = proof(quote, url, {"pages": index["pages"], "excerpts": {}}, evidence)
            level, code = item["level"], item.get("read")
        elif address not in addresses(page):
            level, code = "unverified", "site_screen_address_not_on_source"
        else:
            code = None
    if level != "verified_on_page":
        # Discarded: no stored result, page read or record keeps this address (redact).
        return {"verified": False, "level": level, "discarded": True, **({"reason": code} if code else {})}
    return {"verified": True, "level": level, "discarded": False, "address": address, "url": url}


def address_role(address, person):
    """What a verified address reaches, from the address itself; the provider's channel label is ignored.

    person: its local part holds a word of at least three letters from the verified person's name. team or
    general: a role word (TEAM_INBOX, GENERAL_INBOX; a press inbox is general). refused: a careers, legal,
    support or similar inbox. unknown: anything else, such as another person's address."""
    local = address.partition("@")[0]
    parts, letters = set(re.findall(r"[a-z]+", local)), "".join(re.findall(r"[a-z]+", local))
    names = [word for word in words(person.get("name")).split() if len(word) >= 3] if person["verified"] else []
    if any(name in letters for name in names):
        return "person"
    for role, vocabulary in (("refused", REFUSED_INBOX), ("team", TEAM_INBOX), ("general", GENERAL_INBOX)):
        if parts & vocabulary:
            return role
    return "unknown"


def recipient_rank(kind):
    """Sort key in the owner's preference order; none sorts last."""
    return RECIPIENT_PREFERENCE.index(kind) if kind in RECIPIENT_PREFERENCE else len(RECIPIENT_PREFERENCE)


def contact_questions(person, kind):
    """A title alone never proves remit, so a named person's remit is always an open question."""
    names = ["decision_remit" if person["verified"] else "decision_maker"]
    if person["verified"] and not person["current"]:
        names.append("person_current")
    if kind == "none":
        names.append("recipient")
    return [{"check": name, "question": CONTACT_QUESTIONS[name]} for name in names]


def redact(value, keep=None):
    """``value`` with every email address but ``keep`` replaced by REDACTED, in every string it holds."""
    if isinstance(value, str):
        return EMAIL_ANY.sub(lambda match: match.group(0) if match.group(0).lower() == keep else REDACTED, value)
    if isinstance(value, list):
        return [redact(item, keep) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, keep) for key, item in value.items()}
    return value


def seal_contact(workspace, key, site, raw, pages, today):
    """The contact result to store: its email checked on our own read of its page first, then every address but
    a verified one removed from the result and from the page reads, before either is written."""
    result = _json(raw)
    content, basis = output_of(result)
    text = {field: _string(content.get(field)) for field in CONTACT_SCHEMA["properties"]}
    path = workspace.path("contact", "evidence", key)
    if path.exists():  # Kept before an interruption: its decision stands, and nothing is read again.
        evidence = _evidence(path.read_bytes())
    else:
        evidence = read_evidence("contact", key, result, pages, today)
        check = email_check(text, site, evidence_index(evidence, basis), evidence)
        evidence = redact({**evidence, "email": {name: check[name] for name in ("level", "reason") if name in check}},
                          check.get("address"))
        _write_once(path, (json.dumps(evidence, sort_keys=True) + "\n").encode())
    keep = email_address(text["email"]) if (evidence.get("email") or {}).get("level") == "verified_on_page" else None
    return (json.dumps(redact(result, keep), sort_keys=True) + "\n").encode()


def contact_record(site, run_id, result_raw, evidence_raw):
    """One site's contact under CONTACT_RULE, recomputed from its stored result and page reads alone. A discarded
    address was removed before anything was stored, so its decision is the one kept with the page reads."""
    evidence = _evidence(evidence_raw)
    content, basis = output_of(_json(result_raw))
    text = {field: _string(content.get(field)) for field in CONTACT_SCHEMA["properties"]}
    index = evidence_index(evidence, basis)
    checked_on = evidence.get("checked_on")
    today = date.fromisoformat(checked_on) if isinstance(checked_on, str) else date.min
    person = _person(text, index, evidence, today)
    decision = evidence.get("email") if isinstance(evidence.get("email"), dict) else {}
    if decision.get("level") in (None, "verified_on_page", "no_email"):
        email = email_check(text, site, index, evidence)
    else:
        email = {"verified": False, "level": decision["level"], "discarded": True,
                 **({"reason": decision["reason"]} if decision.get("reason") else {})}
    role = address_role(email["address"], person) if email["verified"] else None
    kind = {"person": "person_email", "team": "team_inbox", "general": "general_inbox"}.get(role, "none")
    label = re.sub(r"[\s-]+", "_", text["channel_type"].lower())
    channel_url = text["channel_url"] if _public_url(text["channel_url"]) and not never_fetch(text["channel_url"]) else None
    return {"schema_version": CONTACT, "rule_version": CONTACT_RULE, "site_key": site["site_key"],
            "origin": site["origin"], "input": site["task_input"], "run_id": run_id,
            "result_sha256": _sha256(result_raw), "evidence_sha256": _sha256(evidence_raw), "checked_on": checked_on,
            "decision_role": text["decision_role"] or None, "person": person,
            "email": {**email, "role": role} if role else email,
            "channel": {"label": label if label in CHANNELS else "none", "url": channel_url},
            "recipient": {"kind": kind, "rank": recipient_rank(kind),
                          "address": email.get("address") if kind != "none" else None},
            "open_questions": contact_questions(person, kind)}


RECORDS = {"screen": screen_record, "contact": contact_record}


def stage_records(workspace, states, stage):
    """The stage's records under its current rule, recomputed from the stored results and page reads. A site
    whose pages are not read yet is left out. Reads no page and calls no provider."""
    records = {}
    for key, site in states[stage].items():
        evidence = workspace.path(stage, "evidence", key)
        if site.get("status") == "completed" and site.get("observed") and evidence.exists():
            records[key] = RECORDS[stage](site["input"], site["run_id"],
                                          workspace.path(stage, "results", key).read_bytes(), evidence.read_bytes())
    return records


def _write_derived(path, value):
    """Replace a derived file atomically (mode 0600): it is recomputed from stored inputs whenever needed."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(value, indent=1, sort_keys=True) + "\n").encode())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify(workspace, *, reader=None, today=None):
    """Read and keep the cited pages of each completed result once, then write each record under its stage's
    current rule as records/<site_key>.<rule>.json. A new rule rewrites nothing old and reads no page again.
    Counts only."""
    today = today or datetime.now(timezone.utc).date()
    pages = Pages(reader)
    counts = {stage: Counter({"records": 0, "pages_kept": 0}) for stage in STAGES}
    with workspace.lock():
        states, _ = workspace.states()
        for stage in STAGES:
            for key, site in states[stage].items():
                if site.get("status") != "completed" or not site.get("observed"):
                    continue
                evidence = workspace.path(stage, "evidence", key)
                if not evidence.exists():
                    result = _json(workspace.path(stage, "results", key).read_bytes())
                    kept = read_evidence(stage, key, result, pages, today)
                    _write_once(evidence, (json.dumps(kept, sort_keys=True) + "\n").encode())
                    counts[stage]["pages_kept"] += 1
        for stage in STAGES:
            for key, record in stage_records(workspace, states, stage).items():
                _write_derived(workspace.record_path(stage, key), record)
                counts[stage]["records"] += 1
                counts[stage][record["tier"] if stage == "screen" else "recipient_" + record["recipient"]["kind"]] += 1
    return {"command": "verify", "state": "complete", "rules": {"screen": SCREEN_RULE, "contact": CONTACT_RULE},
            "screen": dict(counts["screen"]), "contact": dict(counts["contact"]), "page_reads": pages.reads}


# --- summary --------------------------------------------------------------------------------
def _runs(sites):
    observed = Counter(site["status"] for site in sites.values() if site.get("observed"))
    return {"sites": len(sites), "created": sum(site["state"] == "created" for site in sites.values()),
            "outcome_unknown": sum(site["state"] == "unknown" for site in sites.values()),
            "refused_not_created": sum(site["state"] == "refused" for site in sites.values()),
            "completed": observed["completed"], "failed": observed["failed"], "cancelled": observed["cancelled"],
            "in_flight": sum(site["state"] == "created" and not site["observed"] for site in sites.values())}


def _screen_counts(records):
    def tiers(group):
        return {tier: sum(record["tier"] == tier for record in group) for tier in TIERS}

    return {"records": len(records), "tiers": tiers(records),
            "tiers_by_origin": {origin: tiers([r for r in records if r["origin"] == origin])
                                for origin in sorted({r["origin"] for r in records})},
            "tiers_by_calibration": {name: tiers([r for r in records if r["calibration"] is flag])
                                     for name, flag in (("ranked", False), ("calibration", True))},
            "fields": {name: {"answers": dict(Counter(r["choices"].get(name) or (
                "present" if r["answers"][name] else "blank") for r in records)),
                "levels": dict(Counter(r["verification"][name]["level"] for r in records))} for name in SCREEN_PROOFS},
            "claims": {name: dict(Counter(r["gates"]["states"][name] for r in records))
                       for name in (*CLAIMS, "counterevidence")},
            "physical_site_basis": dict(Counter((r["gates"]["facts"]["physical_site"]["proofs"] or [{}])[0].get(
                "source_id", "unproven") for r in records)),
            "task_scope": dict(Counter(r["task_scope"] or "unproven" for r in records)),
            "blockers": dict(Counter(code for r in records for code in r["blockers"])),
            "questions": dict(Counter(r["question_template"] or "none" for r in records if r["tier"] == "outreach_ready")),
            "open_checks": dict(Counter(check for r in records for check in r["open_checks"])),
            "variability_proven": sum(r["variability"]["proven"] for r in records)}


def _contact_counts(records):
    return {"records": len(records),
            "recipients": {kind: sum(r["recipient"]["kind"] == kind for r in records)
                           for kind in (*RECIPIENT_PREFERENCE, "none")},
            "person_levels": dict(Counter(r["person"]["level"] for r in records)),
            "email_levels": dict(Counter(r["email"]["level"] for r in records)),
            "emails_discarded": sum(r["email"]["discarded"] for r in records),
            "channels": dict(Counter(r["channel"]["label"] for r in records)),
            "open_questions": dict(Counter(q["check"] for r in records for q in r["open_questions"]))}


def summary(workspace):
    """Counts by field, claim, tier and recipient, and cost, for both stages, recomputed under each stage's
    current rule from stored results and page reads; also written to summary.json. Never names, addresses or
    quotes, and no page read or provider call."""
    report = {"schema_version": SUMMARY, "command": "summary", "state": "complete",
              "rules": {"screen": SCREEN_RULE, "contact": CONTACT_RULE}}
    billed = committed = Decimal(0)
    with workspace.lock():
        states, pin = workspace.states()
        for stage in STAGES:
            sites, records = states[stage], list(stage_records(workspace, states, stage).values())
            stage_billed = sum((Decimal(site["price_usd"]) for site in sites.values()
                                if site.get("observed") and site["status"] == "completed"), Decimal(0))
            stage_committed = committed_usd(sites)
            billed, committed = billed + stage_billed, committed + stage_committed
            unread = sum(site.get("status") == "completed" and site.get("observed") for site in sites.values())
            report[stage] = {"form": FORMS[stage]["version"], "runs": _runs(sites), "pages_not_read": unread - len(records),
                             **(_screen_counts(records) if stage == "screen" else _contact_counts(records)),
                             "estimated_cost_usd": str(stage_billed), "committed_usd": str(stage_committed)}
        report.update(estimated_cost_usd=str(billed), committed_usd=str(committed),
                      ceiling_usd=pin["ceiling_usd"] if pin else None, max_runs=pin["max_runs"] if pin else None)
        path = workspace.root / "summary.json"
        _write_derived(path, report)
    return report
