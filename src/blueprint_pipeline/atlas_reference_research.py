"""Bounded public reference readback using the existing SSRF-safe page fetcher."""

from pathlib import Path
import math
import time


def retain_references(proposal, root: Path, *, fetcher=None):
    from .website_object_spec_research import PageFetcher, _normalized_text, quoted_values

    refs = proposal.get("scale_references", (proposal.get("automation") or {}).get("scale_references", []))
    urls = list(dict.fromkeys(r.get("citation_url") for r in refs if r.get("citation_url")))
    reader = fetcher or PageFetcher(
        pages_root=root / "reference_pages", max_fetches=2, deadline=time.monotonic() + 45
    )
    fetched = {url: reader.fetch(url) for url in urls[:2]}
    records = []
    for ref in refs:
        url = ref.get("citation_url")
        if not url:
            continue
        page = fetched.get(
            url, {"status": "not_fetched", "reason": "bounded_reference_lookup_limit"}
        )
        quote = ref.get("citation_quote") or ""
        log = next((r for r in reversed(reader.log) if r["url"] == url), {})
        text = Path(log["text_path"]).read_text() if log.get("status") == "ok" else ""
        quoted = bool(quote.strip()) and _normalized_text(quote) in _normalized_text(text)
        values = quoted_values(quote, default_unit="m") if quoted else []
        supported = any(math.isclose(v, ref["assumed_distance_m"], rel_tol=0.05) for v in values)
        records.append(
            {
                "url": url,
                "frame_id": ref["frame_id"],
                "label": ref["label"],
                "quote": quote,
                "status": "retained_public_reference"
                if quoted and supported
                else "unconfirmed_reference",
                "fetch_status": page["status"],
                "quote_found": quoted,
                "quoted_length_matches_assumption": supported,
                "source": log,
                "object_model_match": "candidate_only_not_identified_from_generic_size",
                "physical_measurement": False,
            }
        )
    return records
