# Site universe: design v1

Status: proposed. The owner approved the direction on 2026-10-04.
Supports ADP-010 partner discovery (partner-phase day-7 gate).

## Problem

The daily research agent enumerates sites through web search: about 20 searches and 20 minutes per run. It writes sourced evidence for each site. The October 4 run kept 4 formal candidates. Bulk public sources list thousands of operating sites in one state alone. Enumeration is a dataset problem, so an agent should not repeat it every morning.

## Goals

- A large, deduplicated and versioned list of real US operating sites. It is not limited to warehouses: the site types follow what robot teams can do now.
- Every row records its source, license and retrieval time. A source without recorded rights is refused.
- Each daily run researches the next best unresearched sites and keeps every outcome, so no run starts again from zero.
- Funnel counts and costs at every stage, so we invest in the narrowest stage.

## Non-goals (v1)

- No paid data products, no scraping against site terms and no personal data.
- No change to qualification rules: CRM admission still requires the existing evidence and QA.
- No automation of the weekly refresh until the first snapshots have been reviewed.

## Three layers

| Layer | Producer | Cost per site | Size | Cadence |
| --- | --- | --- | --- | --- |
| Site universe | `tools/site_universe` builder from bulk sources | about $0 | thousands per state | weekly snapshot |
| Ranked backlog | deterministic selection from the snapshot plus history | about $0 | top N per run | every run |
| Qualified CRM | existing research agent, QA and publication | dollars | a few to tens per run | every run |

FindAll and Exa sit between the first two layers. They are paid screening for criteria that a bulk source cannot answer (for example "does manual case picking"), at cents per site. They are not the enumerator.

## Capability-driven taxonomy

`tools/site_universe/taxonomy.json` is data, not code. Each row maps:

1. a robot **capability** that teams demonstrate now (for example fixed-arm machine tending, mobile case picking, bimanual folding, sorting, food preparation, shelf restocking, hospital logistics, lab sample handling or palletizing);
2. to **task families**;
3. to **site types** (plants by subsector, distribution centers, industrial laundries, recycling facilities, food processing, commercial kitchens, grocery and retail, hospitals, labs, hotels, greenhouses);
4. to **source selectors**: NAICS prefixes for OSHA and EPA, OSM tag filters and directory sources.

Each row carries public evidence (demos, launches, deployments), a status (`active` or `candidate`) and a date. Robot-team intelligence (the Team Directory and the weekly robot-team discovery) proposes new rows. An `active` row enters the next refresh, so a new robot capability becomes new site types through a data change, with no code change.

## Sources and rights (v1)

| Source | Provides | License |
| --- | --- | --- |
| OSHA Injury Tracking Application establishment data | name, address, NAICS, annual average employees | US government work |
| EPA Facility Registry Service | name, address, coordinates, NAICS | US government work |
| OpenStreetMap (Overpass, bounded and cached) | name, coordinates, building footprint, tags | ODbL-1.0, attribution "© OpenStreetMap contributors" |
| USDA FSIS inspection directory | meat, poultry and egg plants | US government work |

The source registry records each source's URL, license, attribution, share-alike requirement, allowed uses and verification date. The builder refuses a source without them.

Status on 2026-10-04: EPA FRS and OpenStreetMap download automatically. OSHA ITA and USDA FSIS return HTTP 403 to every automated client, and the builder never bypasses bot protection. An operator downloads those files in a browser and imports them with `python -m tools.site_universe import-raw --source <id> --file <path>`. OpenStreetMap data is ODbL: keep snapshots internal, because public use would trigger share-alike. v2 candidates are county parcel data (licenses vary by county), operator location pages (per-site terms) and job-posting signals.

## Data model

- `SourceRecord`: one row from one source. It has normalized address fields, coordinates, NAICS, category, employees, building area, `raw_sha256` and `retrieved_at`. Personal fields are dropped at parse time.
- `Site`: one physical site. It keeps all its source records, the merge reason for each, `taxonomy_matches`, the union of licenses and `attribution_required`. Its `site_id` is the SHA-256 of its `id_key`, which uses only the site's own records:
  - the address key (normalized street, city, state and ZIP) of its address record, plus the normalized name of its earliest record by source id and record id;
  - or, without an address key, a geohash-7 cell plus that name;
  - or, without coordinates, that name plus the city, state and ZIP.
  - When several sites share a key, the site whose earliest record sorts first keeps it, and the others add the reference of their earliest record.

### Stability of `site_id`

`site_id` is stable within a snapshot. Across refreshes it stays the same only while the site's own records stay the same. A site that gains or loses a record can get another address record, earliest record or coordinate source, and then a new id. Other sites do not change it: a second business at the same address, for example, leaves the first site's id unchanged.

Measured on the Texas inputs of 2026-10-05, adding OSHA ITA to EPA FRS and OpenStreetMap changed 241 of 35,523 ids: 240 sites that gained records and 1 site that was split. None of the 33,170 sites with unchanged records changed id. The first v1 scheme switched from the address key alone to the address key plus name when a second site appeared at the address. With that scheme the same refresh changed 2,092 of 35,494 ids, 1,752 of them for sites whose records did not change. The scheme change re-keys ids once: 32,963 of the 44,918 ids of snapshot `52d60507` are not in snapshot `2dc4b1dd`.

Follow-up, before daily integration uses ids from more than one snapshot: link ids across refreshes. Each refresh writes `previous_ids` for every site whose id changed, found through shared source record references and, failing that, through the same address key and name, so research history keyed by an old id still finds the site. Until then, the runner must treat ids from different snapshots as different keys.

## Dedupe

Records with the same normalized address key merge. Records within 75 m with similar names also merge. Two different names at one address stay separate: a multi-tenant building is not one site. A merge is also refused when it would put two such names into one site through a third record, for example a name that resembles both tenants (`joins_refused_<reason>` in the merge statistics). Two records at one address count as one business when their names match or when the operator of one matches the name or operator of the other, so a store that OSHA lists under a store number with the chain as operator still joins the chain's other records. Within each merge rule the best name match joins first, so the result does not depend on the input order. For each field, the most precise source wins: OSHA for employees, OSHA or EPA for the address, OSM for coordinates and footprint.

## Snapshot contract

- `sites.jsonl.gz` holds canonical JSON sorted by `site_id`, written by gzip with mtime 0. The snapshot ID is its SHA-256.
- `manifest.json` records the schema version, raw input SHA-256s, the source registry entries, counts per source, category and taxonomy row, merge statistics, the license union and `"distribution": "internal_only"`.
- The same raw inputs give byte-identical output. A snapshot is never edited; a refresh makes a new one.
- Snapshots, rankings and the raw cache stay outside the repository, which is public. The builder, the ranker and `import-raw` refuse an output or raw-cache directory inside any checkout of this repository, and `.gitignore` lists `sites.jsonl.gz`, `ranked.jsonl.gz` and `review-top.md`.

## Ranking v1

Status: built on 2026-10-04 for ADP-010 partner discovery. The goal is one design partner: a site plus a robot team, with a rigid-object task for a fixed arm first. Mobile manipulation and the other active capabilities are also ranked.

```bash
python -m tools.site_universe rank --snapshot <dir> --out <dir> \
  [--capabilities a,b] [--top N] [--exclusions-input <file.jsonl>] [--config <file>]
```

The ranking is not a model. `tools/site_universe/rank.py` adds weighted components that measure fit evidence. No component measures or claims buying interest. A naive sort by size put defense primes, Amazon fulfillment centers, Samsung, large hospital systems and misclassified offices at the top. Ranking v1 prefers reachable, decisive, mid-size operators with a real physical task. Size is a curve with a peak, not a contest.

### Weights file

`tools/site_universe/rank_config.json` is versioned data. The owner tunes it with no code change. The id of a config is the SHA-256 of the file bytes, and the rank manifest records it, so every output is bound to one config. The ranker fails closed when:
- the config is malformed;
- a section, rule, matcher, phrase entry or evidence item has a missing or unknown key, so a misspelt key (for example `unless_phrase`) cannot switch a rule off;
- a site type, capability or category id in the config is not in the taxonomy (for example `plant_genral`);
- `values_by_source_count` does not define `"1"`;
- the weights do not sum to 100;
- a capability, site type or category in the snapshot has no weight;
- the snapshot has a site type that the taxonomy does not have.

### Components

Each component has a value from -1 to 1 and a weight in points. The weights sum to 100, so a score reads as points out of 100. Each row keeps all its components: value, weight, points and the basis for the value.

| Component | Points | Evidence |
| --- | --- | --- |
| `capability_fit` | 25 | The best capability weight in scope. Fixed-arm machine tending is 1.0. Kitting and palletizing are 0.85, and hospital logistics is 0.25. Bimanual folding is 0.1, because deformable work is frozen. A generic OpenStreetMap type (`industrial_general`, `plant_general`) adds capabilities only when the site has no specific type. A capability that only a secondary site type supports (for example a warehouse code on a plant) is multiplied by 0.7. |
| `task_evidence` | 10 | NAICS codes that name the task: machine shops (332710) 1.0, injection molding (326199) 0.9, structural steel (3323) 0.2. The default is 0.5. |
| `size_fit` | 22 | OSHA ITA annual average employees on a log curve. The value is 1.0 from 100 to 1,000 employees and 0 at 50 and at 2,500. It is negative below 50 and above 2,500, down to -1.0 at 10,000. When employees are unknown, the OpenStreetMap footprint counts at half confidence. Unknown size is 0. |
| `category_fit` | 10 | The weight of the site type or category, for example manufacturing 1.0, warehousing 0.8, hospitals 0.3 and storefront dry cleaners 0.1. |
| `operator_scale` | 13 | The number of sites in the snapshot that share the operator's company stem or the site name ("Pecan Healthcare PH WEXMOOR COUNTY" gives "Pecan"). A unit number in the name ("GROCER #962", "0418 HARDWARE MART OF LARKSTONE") means a chain of at least 25 sites. A leading number followed by a thoroughfare word anywhere in the name ("4400 Commerce St Plant"), or equal to the site's own house number, is an address, not a unit number. 1 site is 1.0, 25 sites are 0 and 150 sites are -1.0. |
| `activity_evidence` | 8 | The strongest proof of current operation: an OSHA filing for 2024 or later is 1.0, an FSIS listing 0.9, an active FRS record 0.8, an FRS record without a status 0.4 and a map listing only 0.3. |
| `ownership` | 6 | Private is 1.0, public is -1.0 and unknown is 0.5. Public means an OSHA government type, NAICS 491110 or 92, or a public name such as "Postal Service". |
| `source_corroboration` | 4 | 1 source is 0, 2 sources are 0.7 and 3 or more are 1.0. |
| `location_quality` | 2 | A street address adds 0.5 and precise coordinates add 0.5. |

Ranked rows are in order of score, highest first, and ties break by `site_id`. The same snapshot, config and inputs give byte-identical outputs. A per-capability list (for example in the review) scores capability fit, task evidence and category fit for that capability only.

### Exclusions

Each rule has an id, a reason and evidence URLs. An excluded site keeps its score and components. It is written after the ranked sites with the ids of the rules that matched. `--top` limits only the ranked rows, so no exclusion is dropped silently.

| Rule | Matches | TX sites, snapshot `2dc4b1dd` |
| --- | --- | --- |
| `in_house_robotics` | Amazon (also Whole Foods) and Tesla, by name or operator | 175 |
| `known_robot_deployment` | UPS, FedEx, DHL, GXO, Walmart, Sam's Club and Mercado Libre, with the fleet evidence from the taxonomy | 1,563 |
| `defense_itar` | NAICS 336411, 336414, 336415, 336419, 336992 and 332992-332994 in any code of the site, and named primes (Lockheed Martin, RTX, Boeing, L3Harris, General Dynamics, Northrop Grumman, BAE Systems, Bell Textron, Elbit, SpaceX and others) | 268 |
| `office_headquarters` | Primary NAICS 551114, or office words in the name (office, headquarters, HQ, corporate, support center, support office). A name that also names a physical operation is kept: "Office/Warehouse", or "Retail Support Center", which is a distribution center. | 183 |
| `frs_inactive` | Every FRS program record of the site is closed. An OSHA filing for 2024 or later overrides this, because the establishment reported workers that year: 94 such sites are kept. A listing in the USDA FSIS inspection directory also overrides it. | 5,346 |
| `no_location` | No street address and no coordinates | 1,121 |

Phrases match whole words in the name, the other names or the operator. A company acronym (UPS, DHL and GXO; RTX for defense) is listed under `acronyms` and matches only in a company-name position: as the first word of a name, or as the whole operator once its legal form (Inc, LLC, Corp and the like) is removed. So "UPS" and "RTX Corporation" match, but "UPS Holdings LLC" (another company) and "Roll Ups Packaging" do not. Longer company names that start with an acronym ("DHL Supply Chain", "GXO Logistics") are listed as phrases.

`--exclusions-input` takes a JSONL file for existing CRM rows and recent rejections. Each line has exactly one of `site_id`, `name` (optionally narrowed by `city`, `state` or `postal_code`) or `operator`, plus a `reason` and a `source`. A match is recorded as rule `input:<source>`. The manifest lists the rows that match no site, so stale rows are visible. A `site_id` row can go stale after a refresh (see Stability of `site_id`), so long-lived rows should use a name. The CRM is not available offline, so v1 ships the hook and its tests only.

### Outputs

- `ranked.jsonl.gz`: the ranked rows in rank order, then every excluded row by `site_id`. It is canonical JSON, written by gzip with mtime 0.
- `rank-manifest.json`: the config SHA-256 and weights, the snapshot id, the taxonomy used, counts per capability, exclusion counts by rule, score percentiles, file SHA-256s, the evidence URLs not yet verified, the snapshot's license union and `"distribution": "internal_only"`.
- `review-top.md`: the top 25 per capability (name, city, primary site type, NAICS, employees, score and the top 3 components), plus the highest-scoring excluded sites per rule, for owner review. It holds ODbL data, so it is never committed.

### Limits

- Operator scale sees a chain only when its operator or names repeat inside one state's snapshot. Many hospital systems file each hospital under its own name, and franchise stores file as separate companies, so those sites look independent.
- OSHA ITA gives employees for 26% of Texas sites. Unknown size scores 0, so OSHA sites fill the top of each list.
- Capability weights and NAICS task evidence are judgement, not measured conversion. Many top sites tie, and the tie-break by `site_id` is arbitrary.
- The exclusion phrase lists are hand-made. 20 evidence URLs are not yet verified on the web; the manifest lists them.
- NAICS 336411 also excludes civil aircraft makers.
- There is no region preference yet.

Next signals: job posts that name the task, FindAll or Exa screening for the task, and CRM joins for existing rows and rejections.

## Daily integration (after the 7 a.m. release is observed)

1. Control pins one reviewed snapshot (`site_universe: {snapshot_id, uri, generation}`), like the other reviewed research inputs.
2. At run start, the runner selects the next N sites deterministically. It takes the pinned snapshot, removes sites already researched, qualified or rejected in history, and ranks the rest. N is sized to the research and QA envelope.
3. The agent receives that slice as its starting backlog. It researches the top of it, may add sites that it discovers, and records an outcome for each site it touched.
4. Outcomes are stored as events: screened, researched with the evidence gap, qualified, rejected with the reason, duplicate. The next run's selection reads them, so work carries forward.
5. Integration is behind a config flag. It is switched on after the owner reviews the first snapshot.

## Funnel metrics

Report these separately for each run and each week:
- universe size and new sites;
- sites screened;
- sites researched;
- formal candidates;
- verified or accepted;
- contacted;
- replied;
- conversations.

Each count is reported with its cost and time. Counts at different stages are not interchangeable. Measured conversion decides which stage to improve next.

## Rollout

1. Build the producer and the first Texas snapshot, and review its counts and samples. (Tonight.)
2. Add the ranking v1 and the owner review of its top sites per capability. (Built 2026-10-04; see Ranking v1.)
3. Add runner integration behind a flag, plus funnel metrics.
4. Turn it on, then expand state by state.
5. Automate the weekly refresh last.

## Risks

- Source gaps or stale data. The manifest names retrieval dates, and refresh is weekly.
- False merges. Merge reasons are recorded, and tests pin multi-tenant buildings.
- False splits. The tenant guard cannot tell one company whose records at one address carry unrelated names from two tenants, so it keeps them apart. On the Texas inputs it split 62 sites of snapshot `52d60507`; a few of those are true separations, such as a contractor that files at its customer's address.
- License drift. The registry records the verification date, and the builder fails closed.
- Overpass load. One bounded query per state and selector, cached, with timeouts.
