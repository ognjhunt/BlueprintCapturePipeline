"""Robot-team universe: find robot teams, screen each one with quoted evidence. Standard library only.

Owner decision 2026-10-05 (company GCS
``operations/recovery/2026-10-05/owner-decisions/owner-decision-robot-team-universe-20261005.json``, a $15
ceiling): Blueprint's free invited beta needs robot teams with a policy or a robot it can evaluate on a site's
task. The universe leans on early-stage teams of every kind (software, hardware, full stack), stealth teams and
funding announcements, not only the large platform companies, which are harder to work with. This is ADP-010
partner discovery at the partner-phase day-7 gate: robot teams that seek sites or evaluations, by form and task,
are demand evidence. Every team here is a prospect, never a relationship. Nothing here sends, drafts or writes a
CRM.

Two paid stages, one Parallel Task run (processor ``core``, $0.025 per completed run; failed runs are not billed)
per subject:

- ``discover`` runs each query of the reviewed query set (``queries.v1.json``, pinned by ``QUERIES_SHA256``)
  through the ``blueprint.team-discovery.v1`` form: up to 25 companies, each with its website, robot form, task
  focus and one source URL with an exact quote and a date. Companies become teams by the registrable domain of
  their own website (``team_domain``); every name keeps its source.
- ``screen`` runs each discovered team through the ``blueprint.team-screen.v1`` form: robot forms, task evidence,
  stage and funding, HQ and geography, openness signals, whether it seeks pilot sites, and a published business
  contact, each answer with a URL and an exact quote. Teams whose discovery quote is proven go first; the others
  wait unless ``include_unproven``.

The spend machinery is the site screen's, shared rather than copied: ``TeamWorkspace`` subclasses
``site_screen.Workspace`` and both stages create through ``site_screen._submit``. The first ``--apply`` pins the
owner's ceiling, run limit and reference; an intent is fsynced to the spend journal before every create; a
subject with a stored run id, or with an unknown outcome, is never submitted again; and both stages share one
ceiling and one run limit. ``collect`` is ``site_screen.collect``. The shared parts keep their ``site_screen_``
codes, and their ledger names each subject's key ``site_key`` (a query key in discover, a team key in screen);
this line's own codes start with ``team_universe_``.

Verification reads each cited page once with the daily agent's reader (``site_screen.Pages``) and keeps the text.
A quote is ``verified_on_page``, ``in_citation_excerpt`` or ``unverified`` exactly as in the site screen, and a
proven quote must also name its answer (``screen_record``) on the team's own domain or in a quote that names the
team. LinkedIn is never read and never evidence. The contact is a role inbox (partnerships, sales, press, info and
the like) on the team's own domain, published verbatim on a page our own read holds, or else a contact page on
that domain. No person is asked for, no address is guessed, and every other address is removed from stored
results and page reads before they are written (``TeamWorkspace.seal``).

Records are recomputed from stored raw results and page reads under each stage's rule, so a later rule needs no
paid re-run. The out dir must be owner-only (mode 700), outside every Git work tree and outside /tmp. Command
output and ``summary.json`` hold counts only, never names, domains, URLs or quotes.
"""
import json
import re
import stat
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from tools.daily_research import site_screen as ss
from tools.daily_research.site_screen import ScreenError

DISCOVERY = "blueprint.team-discovery.v1"
SCREEN = "blueprint.team-screen.v1"
QUERY_SET = "blueprint.team-discovery-queries.v1"
INPUT = "blueprint.team-universe.input.v1"
EVIDENCE = "blueprint.team-universe.evidence.v1"
TEAMS = "blueprint.team-universe.teams.v1"
SUMMARY = "blueprint.team-universe.summary.v1"
# The rules each stage's records are recomputed under; a derived file's name carries its rule version.
DISCOVERY_RULE = "blueprint.team-discovery-rule.v1"
SCREEN_RULE = "blueprint.team-screen-rule.v1"
STAGES = ("discover", "screen")
DEFAULT_PROCESSOR = ss.DEFAULT_PROCESSOR
QUERIES_PATH = Path(__file__).resolve().with_name("queries.v1.json")
# The reviewed query set. An edit of queries.v1.json refuses until it is reviewed and this pin is updated.
QUERIES_SHA256 = "80740b6c7ffb2913c2c1cd6df2f3b5e7793947ea20c19c4b15a95b504b451e6b"
RANKED_NAME = "ranked.team-rank.v1.json"  # Written by rank.py; summary counts its tiers.
MAX_COMPANIES = 25  # Per discovery run; companies beyond it are counted and ignored.
MAX_QUERIES = 200
MAX_INPUT_SOURCES = 5  # Discovery source URLs sent with a team's screen.

ROBOT_FORMS = ("fixed_arm", "mobile_manipulator", "humanoid", "wheeled", "bimanual", "amr_with_arm", "software_only")
# The site screen's task families (site_screen.FOCUS_HINTS), so family weights from its report line up.
TASK_FAMILIES = {
    "fixed_arm_machine_tending": "machine tending: loading and unloading parts at CNC machines, lathes or presses",
    "kitting_assembly": "kitting and assembly",
    "palletizing_depalletizing": "palletizing and depalletizing",
    "sorting_pick_and_place": "tote and order picking, and sorter induction",
    "mobile_manipulator_case_picking": "mobile case picking and tote moving",
    "truck_trailer_unloading": "truck and trailer unloading",
    "shelf_restocking": "shelf restocking",
    "hospital_logistics": "hospital logistics",
    "food_prep_manipulation": "food preparation",
    "bimanual_folding": "laundry and folding",
    "recycling_sorting": "recycling sorting",
}
QUERY_FAMILIES = {
    "funding_by_task": "recent funding announcements, pre-seed to Series B, by task family",
    "funding_by_form": "recent funding announcements, pre-seed to Series B, by robot form",
    "stealth": "teams that emerged from stealth, and stealth teams seen in job posts or funding news",
    "accelerator": "accelerator batches",
    "robot_learning": "robot-learning, foundation-model and policy companies",
    "integrator": "integrators, by task family",
    "university_spinout": "university spinouts",
    "trade_show": "trade-show exhibitors",
    "open_source": "open-source policy and robot releases",
}
STAGE_CHOICES = ("pre_seed", "seed", "series_a", "series_b", "series_c_or_later", "public", "acquired", "bootstrapped",
                 "grant_funded", "stealth", "unknown")
EVIDENCE_KINDS = ("deployment", "pilot", "demo", "none", "unknown")
SIGNALS = ("design_partners", "simulation", "learned_policy", "shares_policy", "api_sdk", "seeking_partners")
# Quoted screen answers: each has <name>_url and <name>_quote.
PROOFS = ("company", "robot_forms", "task_evidence", "funding", "hq", "deployment_geography", *SIGNALS, "contact")

# Words a proven quote must hold to prove a claim (whole words, after site_screen.words).
FORM_WORDS = {
    "fixed_arm": ("arm", "arms", "robot arm", "robot arms", "robotic arm", "robotic arms", "cobot", "cobots",
                  "collaborative robot", "collaborative robots", "workcell", "workcells", "work cell", "work cells",
                  "robotic cell", "robot cell", "industrial robot", "industrial robots"),
    "mobile_manipulator": ("mobile manipulator", "mobile manipulators", "mobile manipulation", "mobile robot",
                           "mobile robots", "mobile base"),
    "humanoid": ("humanoid", "humanoids", "bipedal", "biped", "legged"),
    "wheeled": ("wheeled", "wheels", "wheel"),
    "bimanual": ("bimanual", "dual arm", "dual arms", "two arms", "two armed", "two arm", "both arms", "two handed"),
    "amr_with_arm": ("amr", "amrs", "autonomous mobile robot", "autonomous mobile robots"),
    "software_only": ("software", "platform", "foundation model", "foundation models", "model", "models", "policy",
                      "policies", "operating system", "robot brain", "api", "sdk"),
}
FAMILY_WORDS = {
    "fixed_arm_machine_tending": ("machine tending", "tending", "tend", "tends", "cnc", "lathe", "lathes", "milling",
                                  "presses", "molding", "moulding", "injection molding", "machining", "machine shop",
                                  "machine shops"),
    "kitting_assembly": ("kitting", "kit", "kits", "assembly", "assemble", "assembles", "assembling", "subassembly",
                         "subassemblies", "screwdriving", "fastening"),
    "palletizing_depalletizing": ("palletizing", "palletising", "palletize", "palletizes", "palletizer", "palletizers",
                                  "depalletizing", "depalletising", "depalletize", "depalletizer", "pallet", "pallets",
                                  "mixed case", "mixed cases"),
    "sorting_pick_and_place": ("picking", "pick", "picks", "piece picking", "each picking", "order picking",
                               "pick and place", "tote", "totes", "bin", "bins", "induction", "inducting", "sortation",
                               "sorter", "sorters", "fulfillment", "fulfilment"),
    "mobile_manipulator_case_picking": ("case picking", "case pick", "mobile manipulator", "mobile manipulators",
                                        "mobile manipulation", "tote", "totes", "cases", "racks", "carts", "conveyors"),
    "truck_trailer_unloading": ("unloading", "unload", "unloads", "unloader", "unloaders", "trailer", "trailers",
                                "truck", "trucks", "container", "containers", "loading dock", "loading docks",
                                "devanning"),
    "shelf_restocking": ("restocking", "restock", "restocks", "shelf", "shelves", "stocking", "grocery store",
                         "grocery stores", "retail store", "retail stores", "convenience store", "convenience stores",
                         "backroom", "sales floor"),
    "hospital_logistics": ("hospital", "hospitals", "healthcare", "health system", "health systems", "clinic",
                           "clinics", "nurse", "nurses", "patient", "patients", "medication", "medications",
                           "pharmacy", "specimen", "specimens", "lab samples"),
    "food_prep_manipulation": ("food", "foods", "kitchen", "kitchens", "cooking", "cook", "meal", "meals", "restaurant",
                               "restaurants", "bakery", "bakeries", "salad", "salads", "pizza", "burger", "burgers",
                               "bowls", "ingredients"),
    "bimanual_folding": ("folding", "fold", "folds", "folded", "laundry", "laundries", "garment", "garments", "towel",
                         "towels", "linen", "linens", "clothes", "clothing", "apparel", "textile", "textiles"),
    "recycling_sorting": ("recycling", "recyclables", "recyclable", "recycle", "waste", "material recovery",
                          "materials recovery", "mrf", "mrfs", "scrap", "e waste", "sorting line"),
}
STAGE_WORDS = {
    "pre_seed": ("pre seed", "preseed"), "seed": ("seed",), "series_a": ("series a",), "series_b": ("series b",),
    "series_c_or_later": ("series c", "series d", "series e", "series f", "series g"),
    "public": ("ipo", "initial public offering", "publicly traded", "nasdaq", "nyse", "stock exchange"),
    "acquired": ("acquired", "acquisition", "acquires", "merger"),
    "bootstrapped": ("bootstrapped", "self funded", "profitable"),
    "grant_funded": ("grant", "grants", "sbir", "sttr", "awarded"),
    "stealth": ("stealth",),
}
SIGNAL_WORDS = {
    "design_partners": ("design partner", "design partners", "design partnership", "design partnerships", "pilot",
                        "pilots", "pilot program", "pilot programs", "early access", "early adopter", "early adopters",
                        "beta program", "partner program", "co development", "codevelopment", "co develop", "trial",
                        "trials"),
    "simulation": ("simulation", "simulations", "simulator", "simulators", "simulated", "sim", "isaac", "mujoco",
                   "gazebo", "ros", "ros 2", "ros2", "digital twin", "digital twins"),
    "learned_policy": ("policy", "policies", "foundation model", "foundation models", "vla", "vlas",
                       "vision language action", "robot learning", "learned", "imitation learning",
                       "reinforcement learning", "end to end", "neural network", "neural networks", "ai model",
                       "ai models", "world model", "world models", "robot brain"),
    "shares_policy": ("open source", "open sourced", "open sourcing", "open weights", "open weight", "weights",
                      "checkpoint", "checkpoints", "hugging face", "huggingface", "github", "model card", "released",
                      "release", "download", "downloadable"),
    "api_sdk": ("api", "apis", "sdk", "sdks", "developer", "developers", "developer kit", "dev kit", "integration",
                "integrations", "ros driver", "python"),
    "seeking_partners": ("pilot", "pilots", "pilot program", "pilot partner", "pilot partners", "pilot sites",
                         "design partner", "design partners", "early access", "early adopter", "early adopters",
                         "looking for", "seeking", "partner with us", "work with us", "contact us", "get in touch",
                         "book a demo", "request a demo", "schedule a demo", "waitlist", "apply", "sites"),
}
# Local-part words of a business role inbox (the site screen's team and general inboxes, and partnership words).
ROLE_INBOX = (ss.TEAM_INBOX | ss.GENERAL_INBOX | frozenset({
    "bd", "bizdev", "partner", "partners", "partnering", "alliance", "alliances", "pilot", "pilots", "hi", "team",
    "customer", "customers", "demo", "demos", "growth"})) - ss.REFUSED_INBOX
# Hosts that are never a team's own website: the site screen's directories, brokers, job boards, social networks
# and newswires, and the code hosts, website builders, link pages and program pages a company may be listed on.
PLATFORM_DOMAINS = frozenset({
    "github.com", "github.io", "gitlab.com", "gitlab.io", "bitbucket.org", "huggingface.co", "hf.co", "arxiv.org",
    "notion.site", "notion.so", "substack.com", "wordpress.com", "blogspot.com", "wixsite.com", "squarespace.com",
    "webflow.io", "framer.website", "framer.ai", "carrd.co", "linktr.ee", "vercel.app", "netlify.app", "pages.dev",
    "herokuapp.com", "web.app", "firebaseapp.com", "ycombinator.com", "techstars.com", "wellfound.com", "angel.co",
    "f6s.com", "producthunt.com", "eventbrite.com", "lu.ma", "youtu.be"})
NOT_TEAM_DOMAINS = ss.NOT_OPERATOR_DOMAINS | PLATFORM_DOMAINS


class TeamError(ScreenError):
    """A stable team_universe_* code; never upstream text or a credential."""


def _json(raw):
    return ss._json(raw)


def _dump(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _evidence(raw):
    value = _json(raw)
    return value if isinstance(value, dict) else {}


def _day(value):
    """A form date (YYYY-MM-DD, YYYY-MM or YYYY, read as its first day), or None."""
    match = ss.DAY.fullmatch(value.strip()) if isinstance(value, str) else None
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1))
    except ValueError:
        return None


def _iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def names_any(text, vocabulary):
    """True when the text holds one of the vocabulary's words or phrases as whole words."""
    found = ss.words(text)
    return any(ss.has_phrase(ss.words(phrase), found) for phrase in vocabulary)


def pick(value, allowed):
    """The enumerated answer: the longest allowed id the answer starts with (case, spaces and hyphens read as
    underscores), else other; blank when empty."""
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") if isinstance(value, str) else ""
    if not text:
        return "blank"
    for choice in sorted(allowed, key=len, reverse=True):
        if text == choice or text.startswith(choice + "_"):
            return choice
    return "other"


def picks(value, allowed):
    """Each allowed id a comma-separated answer names, in order and once."""
    parts = re.split(r"[,;/\n]", value) if isinstance(value, str) else []
    return list(dict.fromkeys(choice for choice in (pick(part, allowed) for part in parts) if choice in allowed))


def families_of(text):
    """The task families a short task phrase names."""
    return [family for family in TASK_FAMILIES if names_any(text, FAMILY_WORDS[family])]


# --- forms ------------------------------------------------------------------------------------
FORM_LIST = ", ".join(ROBOT_FORMS) + ", other"
FAMILY_LIST = ", ".join(TASK_FAMILIES) + ", other"
COMPANY_FIELDS = ("name", "website", "robot_form", "task_focus", "source_url", "source_quote", "source_date")
DISCOVERY_SCHEMA = ss._form(
    "Find up to 25 distinct companies or teams that match the query in the input: companies that build, deploy or "
    "integrate robots, or software that controls robots. Every robot form counts: fixed arm, mobile manipulator, "
    "humanoid, wheeled, bimanual, autonomous mobile robot with an arm, and software only. Prefer early-stage, "
    "stealth and lesser-known teams over large public companies. Use only public sources, prefer sources published "
    "on or after the input's published_since date, and quote each source exactly. " + ss.NO_LINKEDIN,
    {"companies": {"type": "array", "description": "Up to 25 companies that match the query, each listed once.",
                   "items": ss._form("One company that matches the query.", {
                       "name": ss._s("The company's name as its own website or the source states it."),
                       "website": ss._s("The company's own official website URL (its home page), or empty when none "
                                        "is found. Never a directory, social network, accelerator, event or news page."),
                       "robot_form": ss._s("The robot forms it builds or serves, comma-separated, from: " + FORM_LIST
                                           + "."),
                       "task_focus": ss._s("The physical tasks its robots or software do, in a short phrase, for "
                                           "example machine tending, palletizing or tote picking."),
                       "source_url": ss._s("The single best public source URL that shows why this company matches the "
                                           "query: a funding announcement, news story, accelerator, event or job page, "
                                           "or the company's own page."),
                       "source_quote": ss._s("An exact sentence of at least five words copied verbatim from that "
                                             "source that names the company and supports the match."),
                       "source_date": ss._s(ss.DATE)})},
     "notes": ss._s("One or two sentences on what could not be established.")})


def _quoted(name, question):
    """A yes/no/unknown answer with its URL and quote."""
    return {name: ss._s(question + " " + ss.ANSWER), name + "_url": ss._s(ss.URL), name + "_quote": ss._s(ss.QUOTE)}


SCREEN_SCHEMA = ss._form(
    "Research ONE robot company or team, named in the input, as a possible partner for evaluating its robots or "
    "robot policies on real customer sites' tasks. Answer only from public sources about this company, prefer its "
    "own website and sources from the last 18 months, and quote every source exactly. Never name or contact a "
    "person: give only a published business inbox or contact page. " + ss.NO_LINKEDIN,
    {"company": ss._s("The company's name exactly as its own website states it, or empty."),
     "company_url": ss._s("The page on the company's own website that states this name, or empty."),
     "company_quote": ss._s("An exact sentence of at least five words copied verbatim from that page that names the "
                            "company, or empty."),
     "robot_forms": ss._s("The robot forms it builds or serves, comma-separated, from: " + FORM_LIST
                          + ". Empty when unknown."),
     "robot_forms_url": ss._s(ss.URL), "robot_forms_quote": ss._s(ss.QUOTE),
     "task_families": ss._s("The task families in which it has real deployments, pilots or demonstrations, "
                            "comma-separated, from: " + FAMILY_LIST + ". Empty when none is found."),
     "task_evidence": ss._s("The strongest evidence for those tasks. One of: deployment (in use at a customer site), "
                            "pilot (a pilot or trial at a customer site), demo (a public demonstration), none, "
                            "unknown."),
     "task_evidence_url": ss._s(ss.URL), "task_evidence_quote": ss._s(ss.QUOTE), "task_evidence_date": ss._s(ss.DATE),
     "stage": ss._s("The company's funding stage. One of: " + ", ".join(STAGE_CHOICES) + "."),
     "funding_amount": ss._s("The latest funding round's amount with its currency, for example USD 12 million, or "
                             "empty."),
     "funding_date": ss._s("The latest funding round's announcement date as YYYY-MM-DD, or empty."),
     "funding_url": ss._s("The best public source URL for that round or stage, or empty."),
     "funding_quote": ss._s("An exact sentence of at least five words copied verbatim from that source that names "
                            "the company and its round or stage, or empty."),
     "hq": ss._s("The city and country of the company's headquarters, or empty."),
     "hq_url": ss._s(ss.URL), "hq_quote": ss._s(ss.QUOTE),
     "deployment_geography": ss._s("The countries or regions where its robots are deployed or piloted, or empty."),
     "deployment_geography_url": ss._s(ss.URL), "deployment_geography_quote": ss._s(ss.QUOTE),
     **_quoted("design_partners", "Does it run or offer design partnerships, pilots or early-access programs with "
                                  "customers?"),
     **_quoted("simulation", "Does it train or evaluate its robots or policies in simulation, for example Isaac Sim, "
                             "MuJoCo, or ROS with Gazebo?"),
     **_quoted("learned_policy", "Does it build or run its own learned robot policy or model, such as a robot "
                                 "foundation model or a vision-language-action model?"),
     **_quoted("shares_policy", "Does it share a robot policy, model checkpoint or weights publicly or with "
                                "partners?"),
     **_quoted("api_sdk", "Does it offer an API or SDK for its robots or models?"),
     **_quoted("seeking_partners", "Is it seeking customers, sites or pilot partners now?"),
     "seeking_partners_date": ss._s(ss.DATE),
     "contact_email": ss._s("A business email address for partnerships, business development, sales, press or "
                            "general inquiries, published verbatim on a public page and on the company's own domain. "
                            "Never a person's address, never guessed and never derived from a name pattern; empty "
                            "when none is published."),
     "contact_url": ss._s("The page that publishes that address, or else the company's own contact or partnership "
                          "page; empty when none."),
     "contact_quote": ss._s("An exact sentence of at least five words copied verbatim from that page that contains "
                            "the address, or else the page's invitation to get in touch; empty when none."),
     "notes": ss._s("One or two sentences on what could not be established.")})
FORMS = {stage: {"version": version, "json_schema": schema,
                 "sha256": ss._sha256(ss.canonical(schema).encode())}
         for stage, version, schema in (("discover", DISCOVERY, DISCOVERY_SCHEMA), ("screen", SCREEN, SCREEN_SCHEMA))}


# --- the reviewed query set -------------------------------------------------------------------
QUERY_SET_KEYS = frozenset({"schema_version", "query_set", "reviewed_on", "form", "since", "max_companies", "queries"})
QUERY_KEYS = frozenset({"id", "family", "task_focus", "robot_form", "query"})
QUERY_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,79}")


def load_queries(raw):
    """A query set, strictly checked: its version and form, its review date, the date sources should be published
    since, the company cap, and each query once (by id and by text) with a known family, task family and robot form.
    A malformed set is ``team_universe_query_set_invalid``; a malformed or repeated query
    ``team_universe_query_invalid``."""
    document = _json(raw)
    if (not isinstance(document, dict) or set(document) != QUERY_SET_KEYS or document["schema_version"] != QUERY_SET
            or document["form"] != DISCOVERY or not isinstance(document["query_set"], str)
            or not QUERY_ID.fullmatch(document["query_set"]) or _iso_day(document["reviewed_on"]) is None
            or _iso_day(document["since"]) is None or type(document["max_companies"]) is not int
            or not 1 <= document["max_companies"] <= MAX_COMPANIES or not isinstance(document["queries"], list)
            or not 1 <= len(document["queries"]) <= MAX_QUERIES):
        raise TeamError("team_universe_query_set_invalid")
    ids, texts = set(), set()
    for item in document["queries"]:
        if (not isinstance(item, dict) or set(item) != QUERY_KEYS or not isinstance(item["id"], str)
                or not QUERY_ID.fullmatch(item["id"]) or item["id"] in ids or not isinstance(item["family"], str)
                or item["family"] not in QUERY_FAMILIES or item["task_focus"] not in (None, *TASK_FAMILIES)
                or item["robot_form"] not in (None, *ROBOT_FORMS) or not isinstance(item["query"], str)
                or not 20 <= len(item["query"]) <= 600 or item["query"] != " ".join(item["query"].split())
                or ss.words(item["query"]) in texts):
            raise TeamError("team_universe_query_invalid")
        ids.add(item["id"])
        texts.add(ss.words(item["query"]))
    return {"version": QUERY_SET, "sha256": ss._sha256(bytes(raw)),
            **{name: document[name] for name in ("query_set", "reviewed_on", "since", "max_companies")},
            "queries": [dict(item) for item in document["queries"]]}


def reviewed_query_set():
    """The reviewed query set's bytes and contents. An edited file refuses until it is reviewed and
    ``QUERIES_SHA256`` is pinned to it."""
    try:
        raw = Path(QUERIES_PATH).read_bytes()
    except OSError:
        raise TeamError("team_universe_query_set_unreadable") from None
    if ss._sha256(raw) != QUERIES_SHA256:
        raise TeamError("team_universe_query_set_unreviewed")
    return raw, load_queries(raw)


def query_subject(query_set, item):
    """One discovery run's subject. Its key is the question itself (the form, the query text, the date sources should
    be published since and the company cap), so a renamed query is never run twice."""
    task_input = {"query": item["query"], "published_since": query_set["since"],
                  "max_companies": query_set["max_companies"]}
    return {"schema_version": INPUT, "site_key": ss._sha256(ss.canonical(["team_discovery_query", DISCOVERY,
                                                                           task_input]).encode()),
            "origin": "team_discovery_query", "query_id": item["id"], "family": item["family"],
            "task_focus": item["task_focus"], "robot_form": item["robot_form"], "task_input": task_input}


def plan(query_set, *, processor=DEFAULT_PROCESSOR, ceiling_usd=None, max_runs=None, screen_batch=None):
    """The queries, the forms and the worst-case spend: every query run, and every company a query may list screened
    as a separate team (``screen_batch`` bounds that), within the ceiling and run limit when they are given. Reads
    nothing else and calls no provider."""
    price = ss.price_of(processor)
    queries = query_set["queries"]
    upper = len(queries) * query_set["max_companies"]
    screen_runs = upper if screen_batch is None else min(ss.parse_batch_size(screen_batch), upper)
    total = len(queries) + screen_runs
    limits, admitted = None, total
    if ceiling_usd is not None or max_runs is not None:
        ceiling = ss.parse_ceiling(ceiling_usd) if ceiling_usd is not None else None
        limit = ss.parse_max_runs(max_runs) if max_runs is not None else None
        caps = [total] + ([limit] if limit is not None else []) + ([int(ceiling // price)] if ceiling is not None else [])
        admitted = min(caps)
        limits = {"ceiling_usd": str(ceiling) if ceiling is not None else None, "max_runs": limit,
                  "admitted_runs": admitted, "screen_runs_left": max(0, admitted - len(queries))}
    return {"command": "plan", "state": "planned", "forms": {"discover": DISCOVERY, "screen": SCREEN},
            "query_set": {"id": query_set["query_set"], "sha256": query_set["sha256"], "queries": len(queries),
                          "by_family": dict(Counter(item["family"] for item in queries))},
            "processor": processor, "price_usd": str(price),
            "discover": {"runs": len(queries), "worst_case_usd": str(price * len(queries))},
            "screen": {"teams_upper_bound": upper, "batch": screen_batch, "runs": screen_runs,
                       "worst_case_usd": str(price * screen_runs)},
            "uncapped_worst_case_usd": str(price * total), "limits": limits,
            "worst_case_usd": str(price * admitted), "provider_calls": 0}


# --- teams: one per registrable domain --------------------------------------------------------
def on_domain(host, domain):
    host = (host or "").lower().rstrip(".")
    return bool(domain) and (host == domain or host.endswith("." + domain))


def own_page(url, domain):
    """True when the URL is on the team's own domain or a subdomain of it, and is not LinkedIn."""
    return bool(url) and not ss.never_fetch(url) and on_domain(ss._host(url), domain)


def university_host(host):
    return host.endswith(".edu") or bool(re.search(r"\.(?:edu|ac)\.[a-z]{2}$", host))


def team_domain(url):
    """The registrable domain of a company's own website, or None and the refusal: a missing or invalid URL,
    LinkedIn, a university host, or a host that is never a company's own site (a directory, data broker, code host,
    website builder, program page, free mail or government host)."""
    url = ss._string(url)
    if not url:
        return None, "team_universe_website_missing"
    if ss.never_fetch(url):
        return None, "team_universe_website_not_allowed"
    domain = ss.site_domain(url) if ss._public_url(url) else None
    if not domain:
        return None, "team_universe_website_invalid"
    host = ss._host(url)
    if university_host(host):
        return None, "team_universe_website_university"
    if domain in NOT_TEAM_DOMAINS or domain in ss.FREE_MAIL or ss.government_host(host):
        return None, "team_universe_website_not_own"
    return domain, None


def team_key(domain):
    return ss._sha256(ss.canonical(["team", domain]).encode())


def team_subject(domain, group):
    """One team's screen subject: its most used name, its own website, and every discovery source of every name."""
    counts = Counter(ss.words(item["name"]) for _, item in group)
    best = max(counts.values())
    name = next(item["name"] for _, item in group if counts[ss.words(item["name"])] == best)
    proven = [item for _, item in group if item["proven"]]
    days = [day for day in (_day(item["date"]) for item in proven) if day]
    forms = sorted({form for _, item in group for form in item["robot_forms"]})
    urls = [item["source_url"] for _, item in sorted(group, key=lambda pair: not pair[1]["proven"]) if item["source_url"]]
    return {"schema_version": INPUT, "site_key": team_key(domain), "origin": "team_discovery", "domain": domain,
            "discovery": {"proven": bool(proven), "mentions": len(group),
                          "query_families": sorted({record["family"] for record, _ in group}),
                          "families": sorted({family for _, item in group for family in item["families"]}),
                          "robot_forms": forms, "latest": max(days).isoformat() if days else None,
                          "sources": [{"query_id": record["query_id"], "query_family": record["family"],
                                       "name": item["name"], "source_url": item["source_url"],
                                       "level": item["verification"]["level"], "named": item["named"],
                                       "proven": item["proven"], "date": item["date"]} for record, item in group]},
            "task_input": ss._compact({"company": name, "website": group[0][1]["website"],
                                       "robot_form_hint": ", ".join(forms) or None,
                                       "task_focus_hint": next((item["task_focus"] for _, item in group
                                                                if item["task_focus"]), None),
                                       "known_source_urls": list(dict.fromkeys(urls))[:MAX_INPUT_SOURCES]})}


def team_list(records):
    """The discovered teams, one per registrable domain in the order discovery first named them, and refusal counts.
    Two different names on one domain (a directory, a news site or a mistake) make no team."""
    groups, refused = {}, Counter()
    for record in records:
        for item in record["companies"]:
            if not item.get("valid"):
                refused["team_universe_company_invalid"] += 1
            elif not item["name"]:
                refused["team_universe_company_name_missing"] += 1
            elif item["domain"] is None:
                refused[item["domain_refusal"]] += 1
            else:
                groups.setdefault(item["domain"], []).append((record, item))
    teams = []
    for domain, group in groups.items():
        names = list(dict.fromkeys(item["name"] for _, item in group))
        if any(not ss.same_operator(first, second) for number, first in enumerate(names) for second in names[number + 1:]):
            refused["team_universe_domain_names_conflict"] += 1
            continue
        teams.append(team_subject(domain, group))
    return teams, refused


def order_teams(teams, weights=None):
    """Proven discovery first, then the highest family weight among the team's discovered task families (when
    weights are given), then the most recent proven source, then the team key."""
    normalized = (weights or {}).get("normalized") or {}

    def priority(team):
        found = team["discovery"]
        weight = max((normalized.get(family, 0) for family in found["families"]), default=0)
        latest = _day(found["latest"])
        return (not found["proven"], -weight, -(latest.toordinal() if latest else 0), team["site_key"])
    return sorted(teams, key=priority)


# --- the out dir ------------------------------------------------------------------------------
def rule_of(stage):
    return DISCOVERY_RULE if stage == "discover" else SCREEN_RULE


class TeamWorkspace(ss.Workspace):
    """One team universe out dir: the site screen's pin, spend journal, ledgers and admission, with the discover
    and screen stages and their forms. It must be owner-only: a new out dir is made with mode 700, and an existing
    one that group or others can open is refused, never changed."""
    stages, forms = STAGES, FORMS

    def __init__(self, path, *, create=False, code_root=ss.CODE_ROOT):
        super().__init__(path, create=create, code_root=code_root)
        if stat.S_IMODE(self.root.stat().st_mode) & 0o077:
            raise TeamError("team_universe_out_dir_not_private")

    def record_path(self, stage, key):
        return self.root / stage / "records" / f"{key}.{rule_of(stage).removeprefix('blueprint.')}.json"

    def records(self, stage):
        """The derived records written under the stage's current rule."""
        rule = rule_of(stage).removeprefix("blueprint.")
        folder = self.root / stage / "records"
        return [_json(path.read_bytes()) for path in sorted(folder.glob(f"*.{rule}.json"))] if folder.is_dir() else []

    def teams_path(self):
        return self.root / "discover" / f"teams.{DISCOVERY_RULE.removeprefix('blueprint.')}.json"

    def seal(self, stage, key, site, raw, pages, today):
        """A completed result as it is stored: a discovery result without any email address; a team screen with
        only a verified role inbox, after its pages are read and its contact decided (screen_evidence)."""
        result = _json(raw)
        if stage == "discover":
            return _dump(ss.redact(result, None))
        evidence = screen_evidence(self, key, site, result, pages, today)
        return _dump(ss.redact(result, kept_address(result, evidence)))


# --- paid stages ------------------------------------------------------------------------------
def parse_limit(value):
    if type(value) is not int or not 1 <= value <= MAX_QUERIES:
        raise TeamError("team_universe_limit_invalid")
    return value


def _shaped(result, command, **extra):
    shaped = {**result, "command": command, **extra}
    shaped["subjects"] = shaped.pop("sites")
    return shaped


def discover(workspace, query_set, *, client, owner_reference, ceiling_usd, max_runs, processor=DEFAULT_PROCESSOR,
             apply=False, limit=None, environ=None):
    """Run each query that has no run yet, in file order (``limit``: only the first ones; a canary is 1), within the
    pinned ceiling and run limit shared with the screen. A dry run unless ``apply``."""
    ss.refuse_on_worker(environ)
    subjects = [query_subject(query_set, item) for item in query_set["queries"]]
    if limit is not None:
        subjects = subjects[:parse_limit(limit)]
    with workspace.lock():
        states, pin = workspace.states()
        result = ss._submit(workspace, "discover", subjects, states, pin, client=client,
                            owner_reference=owner_reference, ceiling_usd=ceiling_usd, max_runs=max_runs,
                            processor=processor, apply=apply, input_sha256=query_set["sha256"])
    return _shaped(result, "discover", query_set=query_set["query_set"], queries=len(query_set["queries"]),
                   limit=limit)


def screen_queue(workspace, states, *, weights=None, include_unproven=False):
    """The teams the screen would take, in priority order (order_teams), and their counts. A team whose discovery
    quote is not proven waits unless ``include_unproven``."""
    teams, refused = team_list(stage_records(workspace, states, "discover").values())
    proven = [team for team in teams if team["discovery"]["proven"]]
    queue = order_teams(teams if include_unproven else proven, weights)
    return queue, {"discovered": len(teams), "proven": len(proven),
                   "unproven_held": 0 if include_unproven else len(teams) - len(proven), "refused": dict(refused)}


def screen(workspace, *, client, owner_reference, ceiling_usd, max_runs, processor=DEFAULT_PROCESSOR, apply=False,
           batch_size=None, include_unproven=False, weights=None, environ=None):
    """Screen each discovered team that has no run yet, in priority order (``batch_size``: only the first ones, at
    most ``max_runs``), within the pinned ceiling and run limit shared with discovery. A dry run unless ``apply``."""
    ss.refuse_on_worker(environ)
    if batch_size is not None and ss.parse_batch_size(batch_size) > ss.parse_max_runs(max_runs):
        raise TeamError("team_universe_batch_exceeds_max_runs")
    with workspace.lock():
        states, pin = workspace.states()
        queue, teams = screen_queue(workspace, states, weights=weights, include_unproven=include_unproven)
        if batch_size is not None:
            queue = queue[:batch_size]
        result = ss._submit(workspace, "screen", queue, states, pin, client=client, owner_reference=owner_reference,
                            ceiling_usd=ceiling_usd, max_runs=max_runs, processor=processor, apply=apply,
                            input_sha256=ss._sha256(ss.canonical([team["site_key"] for team in queue]).encode()))
    return _shaped(result, "screen", teams=teams, batch=batch_size, include_unproven=include_unproven)


# --- verification -----------------------------------------------------------------------------
def companies_of(content):
    items = content.get("companies") if isinstance(content, dict) else None
    return items if isinstance(items, list) else []


def answers_of(content):
    return {field: ss._string(content.get(field)) for field in SCREEN_SCHEMA["properties"]}


def cited_urls(stage, content):
    """Every URL a result gives as a source: each kept company's source, or each screen answer's."""
    if stage == "discover":
        urls = {ss._string(item.get("source_url")) for item in companies_of(content)[:MAX_COMPANIES]
                if isinstance(item, dict)}
    else:
        urls = {ss._string(content.get(name + "_url")) for name in PROOFS}
    return sorted(urls - {""})


def read_evidence(stage, key, content, pages, today):
    """Our reads of every URL the result cites, with their date, kept so its record can be recomputed offline."""
    return {"schema_version": EVIDENCE, "stage": stage, "site_key": key, "checked_on": today.isoformat(),
            "pages": {url: pages(url) for url in cited_urls(stage, content)}}


def inbox_kind(address):
    """What an address's local part names: role (every word a role word, or a two-letter region or language tag),
    refused (careers, legal, support and the like), else personal_or_unknown, which never counts."""
    parts = re.findall(r"[a-z]+", address.partition("@")[0])
    if not parts or set(parts) & ss.REFUSED_INBOX:
        return "refused"
    if set(parts) & ROLE_INBOX and all(part in ROLE_INBOX or len(part) <= 2 for part in parts):
        return "role"
    return "personal_or_unknown"


def email_check(answers, subject, index, evidence):
    """The published role inbox, proven only on our own read of its cited page: one address on the team's own domain
    (or a subdomain), never free mail, a role inbox (inbox_kind), whose quote holds the exact address and stands
    whole-word on that page, where the address also stands as a whole token. A provider excerpt never counts."""
    raw, url, quote = answers["contact_email"], answers["contact_url"], answers["contact_quote"]
    if not raw:
        return {"verified": False, "level": "no_email", "discarded": False}
    address = ss.email_address(raw)
    mail = address.rpartition("@")[2] if address else ""
    code = None
    if address is None:
        level, code = "unverified", "team_universe_email_invalid"
    elif ss.never_fetch(url) or ss.never_fetch(raw):
        level = "source_not_allowed"
    elif mail in ss.FREE_MAIL or ss.site_domain("https://" + mail) in ss.FREE_MAIL:
        level, code = "unverified", "team_universe_email_free_mail"
    elif not on_domain(mail, subject["domain"]):
        level, code = "unverified", "team_universe_email_off_team_domain"
    elif inbox_kind(address) != "role":
        level, code = "unverified", "team_universe_email_not_role_inbox"
    elif not quote:
        level, code = "unverified", "team_universe_quote_missing"
    elif address not in ss.addresses(quote):
        level, code = "unverified", "team_universe_quote_lacks_address"
    else:
        level, _, page = ss.holding(quote, url, index, kinds=("pages",))
        if level is None:
            item = ss.proof(quote, url, {"pages": index["pages"], "excerpts": {}}, evidence)
            level, code = item["level"], item.get("read")
        elif address not in ss.addresses(page):
            level, code = "unverified", "team_universe_address_not_on_source"
    if level != "verified_on_page":
        return {"verified": False, "level": level, "discarded": True, **({"reason": code} if code else {})}
    return {"verified": True, "level": level, "discarded": False, "address": address, "url": url}


def page_check(answers, subject, index, evidence):
    """A contact page: on the team's own domain, never LinkedIn, and its quote stands whole-word on our own read."""
    url, quote = answers["contact_url"], answers["contact_quote"]
    if not url:
        return {"verified": False, "level": "no_contact_page"}
    if ss.never_fetch(url):
        return {"verified": False, "level": "source_not_allowed"}
    if not own_page(url, subject["domain"]):
        return {"verified": False, "level": "off_team_domain"}
    level, _, _ = ss.holding(quote, url, index, kinds=("pages",))
    if level is None:
        return {"verified": False, "level": ss.proof(quote, url, {"pages": index["pages"], "excerpts": {}},
                                                     evidence)["level"]}
    return {"verified": True, "level": level, "url": url}


def screen_evidence(workspace, key, subject, result, pages, today):
    """The page reads kept for one team screen. They are read once, the email decided on them, and then every email
    address but a verified role inbox is removed before they are written. Kept reads are never read again."""
    path = workspace.path("screen", "evidence", key)
    if path.exists():
        return _evidence(path.read_bytes())
    content, basis = ss.output_of(result)
    evidence = read_evidence("screen", key, content, pages, today)
    email = email_check(answers_of(content), subject, ss.evidence_index(evidence, basis), evidence)
    decision = {"email_level": email["level"], **({"email_reason": email["reason"]} if email.get("reason") else {})}
    evidence = ss.redact({**evidence, "contact": decision}, email.get("address"))
    ss._write_once(path, _dump(evidence))
    return evidence


def kept_address(result, evidence):
    """The one address a stored team screen keeps: its contact email, when the kept decision verified it."""
    content, _ = ss.output_of(result)
    decision = evidence.get("contact") if isinstance(evidence.get("contact"), dict) else {}
    return ss.email_address(ss._string(content.get("contact_email"))) if decision.get(
        "email_level") == "verified_on_page" else None


def contact_of(answers, subject, index, evidence):
    """The contact route: a verified role inbox, else a verified contact page on the team's domain, else none. An
    address that was discarded is gone, so its kept decision stands."""
    decision = evidence.get("contact") if isinstance(evidence.get("contact"), dict) else {}
    if decision.get("email_level") in (None, "verified_on_page", "no_email"):
        email = email_check(answers, subject, index, evidence)
    else:
        email = {"verified": False, "level": decision["email_level"], "discarded": True,
                 **({"reason": decision["email_reason"]} if decision.get("email_reason") else {})}
    page = page_check(answers, subject, index, evidence)
    route = "role_inbox" if email["verified"] else "contact_page" if page["verified"] else "none"
    contact = {"route": route, "email_level": email["level"], "page_level": page["level"],
               "discarded": email["discarded"]}
    if email.get("reason"):
        contact["email_reason"] = email["reason"]
    if route == "role_inbox":
        contact.update(address=email["address"], url=email["url"])
    elif route == "contact_page":
        contact["url"] = page["url"]
    return contact


def discovery_record(subject, run_id, result_raw, evidence_raw):
    """One query's companies under DISCOVERY_RULE, recomputed from its stored result and page reads alone. A company
    is proven when its quote stands at its own source URL (verified_on_page or in_citation_excerpt) and holds every
    significant word of its name; it is fresh when its source date is between published_since and the read."""
    evidence = _evidence(evidence_raw)
    content, basis = ss.output_of(_json(result_raw))
    items, index = companies_of(content), ss.evidence_index(evidence, basis)
    checked, since = _day(evidence.get("checked_on")), _day(subject["task_input"].get("published_since"))
    companies = []
    for number, item in enumerate(items[:MAX_COMPANIES]):
        if not isinstance(item, dict):
            companies.append({"index": number, "valid": False})
            continue
        text = {field: ss._string(item.get(field)) for field in COMPANY_FIELDS}
        domain, refusal = team_domain(text["website"])
        verification = ss.proof(text["source_quote"], text["source_url"], index, evidence)
        named = bool(text["name"]) and ss.names_operator(text["source_quote"], text["name"])
        day = _day(text["source_date"])
        companies.append({
            "index": number, "valid": True, "name": text["name"] or None, "website": text["website"] or None,
            "domain": domain, "domain_refusal": refusal, "robot_forms": picks(text["robot_form"], ROBOT_FORMS),
            "task_focus": text["task_focus"] or None, "families": families_of(text["task_focus"]),
            "source_url": text["source_url"] or None, "source_quote": text["source_quote"] or None,
            "verification": verification, "named": named, "proven": verification["level"] in ss.PROVEN and named,
            "date": text["source_date"] or None, "fresh": bool(day and since and checked and since <= day <= checked)})
    return {"schema_version": DISCOVERY, "rule_version": DISCOVERY_RULE, "site_key": subject["site_key"],
            "query_id": subject["query_id"], "family": subject["family"], "task_focus": subject["task_focus"],
            "robot_form": subject["robot_form"], "run_id": run_id, "result_sha256": ss._sha256(result_raw),
            "evidence_sha256": ss._sha256(evidence_raw), "checked_on": evidence.get("checked_on"),
            "listed": len(items), "beyond_cap": max(0, len(items) - MAX_COMPANIES), "companies": companies}


def screen_record(subject, run_id, result_raw, evidence_raw):
    """One team's screen under SCREEN_RULE, recomputed from its stored result and page reads alone.

    A claim counts only when its quote is proven at its own URL (verified_on_page or in_citation_excerpt) and names
    its answer, on the team's own domain or in a quote that names the team: the company name on its own site (a
    different proven name contradicts the discovered one); a robot form or a task family by one of its words, the
    task as a deployment, pilot or demonstration; a funding stage by its words; a signal answered yes by one of its
    words; the HQ and geography by a word of the answer. Activity is the latest such date, including discovery's."""
    evidence = _evidence(evidence_raw)
    content, basis = ss.output_of(_json(result_raw))
    answers, index = answers_of(content), ss.evidence_index(evidence, basis)
    checked = _day(evidence.get("checked_on"))
    domain, given = subject["domain"], subject["task_input"].get("company") or ""
    verification = {name: ss.proof(answers[name + "_quote"], answers[name + "_url"], index, evidence)
                    for name in PROOFS}

    def proven(name):
        return verification[name]["level"] in ss.PROVEN

    company = answers["company"]
    identity = (bool(company) and proven("company") and own_page(answers["company_url"], domain)
                and ss.names_operator(answers["company_quote"], company))
    contradicted = identity and bool(given) and not ss.same_operator(given, company)
    names = [name for name in (company if identity else "", given) if name]

    def about(name):
        return own_page(answers[name + "_url"], domain) or any(ss.names_operator(answers[name + "_quote"], team)
                                                               for team in names)

    def shown(name, vocabulary):
        return proven(name) and about(name) and names_any(answers[name + "_quote"], vocabulary)

    def described(name):
        wanted = {word for word in ss.words(answers[name]).split() if word not in ss.TASK_STOPWORDS}
        found = set(ss.words(answers[name + "_quote"]).split())
        return {"answer": answers[name] or None, "proven": bool(wanted & found) and proven(name) and about(name)}

    forms = picks(answers["robot_forms"], ROBOT_FORMS)
    kind = pick(answers["task_evidence"], EVIDENCE_KINDS)
    families = picks(answers["task_families"], TASK_FAMILIES)
    task = [family for family in families if kind in ("deployment", "pilot", "demo")
            and shown("task_evidence", FAMILY_WORDS[family])]
    stage = pick(answers["stage"], STAGE_CHOICES)
    stage_ok = stage in STAGE_WORDS and shown("funding", STAGE_WORDS[stage])
    signals = {name: {"answer": pick(answers[name], ("yes", "no", "unknown"))} for name in SIGNALS}
    for name, signal in signals.items():
        signal["proven"] = signal["answer"] == "yes" and shown(name, SIGNAL_WORDS[name])
    task_day = _day(answers["task_evidence_date"])
    dated = ((answers["task_evidence_date"], bool(task)), (answers["funding_date"], stage_ok),
             (answers["seeking_partners_date"], signals["seeking_partners"]["proven"]),
             ((subject.get("discovery") or {}).get("latest"), True))
    days = [day for value, counted in dated if counted for day in (_day(value),) if day and checked and day <= checked]
    claims = {"company": identity, "robot_forms": any(shown("robot_forms", FORM_WORDS[form]) for form in forms),
              "task_evidence": bool(task), "funding": stage_ok,
              **{name: signals[name]["proven"] for name in SIGNALS}}
    return {"schema_version": SCREEN, "rule_version": SCREEN_RULE, "site_key": subject["site_key"], "domain": domain,
            "origin": subject["origin"], "input": subject["task_input"], "discovery": subject.get("discovery"),
            "run_id": run_id, "result_sha256": ss._sha256(result_raw), "evidence_sha256": ss._sha256(evidence_raw),
            "checked_on": evidence.get("checked_on"), "answers": answers, "verification": verification,
            "identity": {"state": "contradicted" if contradicted else "verified_fact" if identity else "unresolved",
                         "company": company if identity else None},
            "robot_forms": {"claimed": forms, "proven": [form for form in forms if shown("robot_forms",
                                                                                          FORM_WORDS[form])]},
            "task": {"evidence": kind, "claimed": families, "proven": task, "date": answers["task_evidence_date"] or None,
                     "recent": bool(task and task_day and checked and 0 <= (checked - task_day).days <= ss.FRESH_DAYS)},
            "stage": {"answer": stage, "verified": stage_ok,
                      "amount": (answers["funding_amount"] or None) if stage_ok else None,
                      "date": (answers["funding_date"] or None) if stage_ok else None},
            "hq": described("hq"), "geography": described("deployment_geography"), "signals": signals,
            "contact": contact_of(answers, subject, index, evidence),
            "latest_activity": max(days).isoformat() if days else None,
            "proving_sources": [{"claim": name, "url": answers[name + "_url"],
                                 "quote_sha256": ss._sha256(answers[name + "_quote"].encode()),
                                 "level": verification[name]["level"],
                                 "tool_result_sha256": verification[name].get("sha256")}
                                for name, counted in claims.items() if counted]}


def stage_records(workspace, states, stage):
    """The stage's records under its current rule, recomputed from stored results and page reads, in the order the
    stage submitted them. A subject whose pages are not read yet is left out. Reads no page and calls no provider."""
    build = discovery_record if stage == "discover" else screen_record
    records = {}
    for key, entry in states[stage].items():
        evidence = workspace.path(stage, "evidence", key)
        if entry.get("status") == "completed" and entry.get("observed") and evidence.exists():
            records[key] = build(entry["input"], entry["run_id"], workspace.path(stage, "results", key).read_bytes(),
                                 evidence.read_bytes())
    return records


def verify(workspace, *, reader=None, today=None):
    """Read and keep the cited pages of each completed result once (a team screen's are kept when it is collected),
    then write every record under its stage's current rule and the team list. A new rule rewrites nothing old and
    reads no page again. Counts only."""
    today = today or datetime.now(timezone.utc).date()
    pages, kept = ss.Pages(reader), Counter()
    with workspace.lock():
        states, _ = workspace.states()
        for stage in STAGES:
            for key, entry in states[stage].items():
                path = workspace.path(stage, "evidence", key)
                if entry.get("status") != "completed" or not entry.get("observed") or path.exists():
                    continue
                result = _json(workspace.path(stage, "results", key).read_bytes())
                if stage == "discover":
                    content, _ = ss.output_of(result)
                    ss._write_once(path, _dump(ss.redact(read_evidence(stage, key, content, pages, today), None)))
                else:
                    screen_evidence(workspace, key, entry["input"], result, pages, today)
                kept[stage] += 1
        records = {stage: stage_records(workspace, states, stage) for stage in STAGES}
        for stage in STAGES:
            for key, record in records[stage].items():
                ss._write_derived(workspace.record_path(stage, key), record)
        teams, refused = team_list(records["discover"].values())
        ss._write_derived(workspace.teams_path(), {"schema_version": TEAMS, "rule_version": DISCOVERY_RULE,
                                                   "teams": teams, "refused": dict(refused)})
    screens = list(records["screen"].values())
    return {"command": "verify", "state": "complete", "rules": {"discover": DISCOVERY_RULE, "screen": SCREEN_RULE},
            "discover": {"records": len(records["discover"]), "pages_kept": kept["discover"],
                         "companies": sum(len(record["companies"]) for record in records["discover"].values()),
                         "teams": len(teams), "teams_proven": sum(team["discovery"]["proven"] for team in teams)},
            "screen": {"records": len(screens), "pages_kept": kept["screen"],
                       "identity_verified": sum(record["identity"]["state"] == "verified_fact" for record in screens),
                       "contact_routes": dict(Counter(record["contact"]["route"] for record in screens))},
            "page_reads": pages.reads}


# --- summary ----------------------------------------------------------------------------------
def discover_counts(records):
    items = [item for record in records for item in record["companies"]]
    valid = [item for item in items if item.get("valid")]
    families = {}
    for record in records:
        group = families.setdefault(record["family"], Counter({"queries": 0, "companies": 0, "proven": 0}))
        group["queries"] += 1
        group["companies"] += sum(bool(item.get("valid")) for item in record["companies"])
        group["proven"] += sum(bool(item.get("proven")) for item in record["companies"])
    return {"records": len(records),
            "companies": {"listed": sum(record["listed"] for record in records), "kept": len(items),
                          "beyond_cap": sum(record["beyond_cap"] for record in records),
                          "invalid": len(items) - len(valid),
                          "levels": dict(Counter(item["verification"]["level"] for item in valid)),
                          "named": sum(item["named"] for item in valid), "proven": sum(item["proven"] for item in valid),
                          "fresh": sum(item["fresh"] for item in valid),
                          "domain_refusals": dict(Counter(item["domain_refusal"] for item in valid
                                                          if item["domain_refusal"]))},
            "by_query_family": {family: dict(counts) for family, counts in sorted(families.items())}}


def screen_counts(records):
    return {"records": len(records),
            "identity": dict(Counter(record["identity"]["state"] for record in records)),
            "robot_forms": dict(Counter(form for record in records for form in record["robot_forms"]["proven"])),
            "task_families": dict(Counter(family for record in records for family in record["task"]["proven"])),
            "task_evidence": dict(Counter(record["task"]["evidence"] for record in records if record["task"]["proven"])),
            "stages": dict(Counter(record["stage"]["answer"] for record in records if record["stage"]["verified"])),
            "signals": {name: sum(record["signals"][name]["proven"] for record in records) for name in SIGNALS},
            "contact_routes": dict(Counter(record["contact"]["route"] for record in records)),
            "emails_discarded": sum(record["contact"]["discarded"] for record in records),
            "fields": {name: dict(Counter(record["verification"][name]["level"] for record in records))
                       for name in PROOFS}}


def summary(workspace):
    """Counts by stage, field, claim and route, and cost, recomputed under each stage's current rule from stored
    results and page reads, plus the tiers of the last rank; also written to summary.json. Never names, domains,
    URLs or quotes, and no page read or provider call."""
    report = {"schema_version": SUMMARY, "command": "summary", "state": "complete",
              "rules": {"discover": DISCOVERY_RULE, "screen": SCREEN_RULE}}
    billed = committed = Decimal(0)
    with workspace.lock():
        states, pin = workspace.states()
        records = {stage: list(stage_records(workspace, states, stage).values()) for stage in STAGES}
        for stage in STAGES:
            entries = states[stage]
            stage_billed = sum((Decimal(entry["price_usd"]) for entry in entries.values()
                                if entry.get("observed") and entry["status"] == "completed"), Decimal(0))
            stage_committed = ss.committed_usd(entries)
            billed, committed = billed + stage_billed, committed + stage_committed
            unread = sum(entry.get("status") == "completed" and bool(entry.get("observed")) for entry in entries.values())
            counts = discover_counts(records[stage]) if stage == "discover" else screen_counts(records[stage])
            report[stage] = {"form": FORMS[stage]["version"], "runs": ss._runs(entries),
                             "pages_not_read": unread - len(records[stage]), **counts,
                             "estimated_cost_usd": str(stage_billed), "committed_usd": str(stage_committed)}
        teams, refused = team_list(records["discover"])
        proven = sum(team["discovery"]["proven"] for team in teams)
        report["teams"] = {"discovered": len(teams), "proven": proven, "unproven": len(teams) - proven,
                           "refused": dict(refused)}
        path = workspace.root / RANKED_NAME
        ranked = _json(path.read_bytes()) if path.exists() else None
        rows = ranked.get("teams") if isinstance(ranked, dict) and isinstance(ranked.get("teams"), list) else None
        report["rank"] = None if rows is None else {
            "teams": len(rows), "tiers": {tier: sum(isinstance(row, dict) and row.get("tier") == tier for row in rows)
                                          for tier in ("beta_candidate", "prospect", "insufficient")}}
        report.update(estimated_cost_usd=str(billed), committed_usd=str(committed),
                      ceiling_usd=pin["ceiling_usd"] if pin else None, max_runs=pin["max_runs"] if pin else None)
        ss._write_derived(workspace.root / "summary.json", report)
    return report
