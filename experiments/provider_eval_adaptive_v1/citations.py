"""Public citation validation, with the confirmed Chef numeric pagination key."""

import re
from urllib.parse import urlsplit, urlunsplit

from experiments.provider_eval_recovery.adapters import Limits, normalize


class ProviderInputWarning(ValueError):
    pass


class CitationNormalizationError(ValueError):
    pass


def validated_url(mode, url):
    if not isinstance(url, str):
        raise CitationNormalizationError("invalid_citation_url_type")
    parsed = urlsplit(url)
    pagination = (parsed.scheme == "https" and parsed.netloc.lower() == "www.chefrobotics.ai"
                  and parsed.path in ("", "/")
                  and re.fullmatch(r"3f5a3c8b_page=[0-9]{1,6}", parsed.query))
    checked = urlunsplit(parsed._replace(query="")) if pagination else url
    # Existing validation still rejects userinfo, fragments, unsupported query
    # keys, signed/credential links and redirects. No citation is dereferenced.
    try:
        normalize(mode, {"results": [{"url": checked, "title": "", "excerpts": [], "snippet": ""}]})
    except (ValueError, TypeError) as exc:
        raise CitationNormalizationError("local_citation_validation_failed") from exc
    return checked, "chef_numeric_pagination" if pagination else "existing_public_citation"


def audit_urls(mode, raw):
    if not isinstance(raw.get("results"), list):
        raise CitationNormalizationError("invalid_source_results_shape")
    report = []
    for result in raw["results"][:10]:
        if not isinstance(result, dict):
            raise CitationNormalizationError("invalid_source_result_shape")
        _, rule = validated_url(mode, result.get("url"))
        report.append({"url": result["url"], "rule": rule})
    return report


def normalize_public(mode, raw):
    # Audit every bounded citation before clipping context, including URLs after
    # the character budget would otherwise stop parsing the response.
    audit_urls(mode, raw)
    result, remaining = [], 6000
    for source in raw["results"][:10]:
        url = source["url"]
        checked, _ = validated_url(mode, url)
        try:
            single = normalize(mode, {"results": [{**source, "url": checked}]}, Limits(evidence_chars=6000))
        except (ValueError, TypeError) as exc:
            raise CitationNormalizationError("local_citation_source_parsing_failed") from exc
        if not single:
            break
        item = single[0]
        overhead = len(url) + len(item["title"])
        if overhead >= remaining:
            break
        text = item["text"][:remaining - overhead]
        result.append({**item, "url": url, "text": text})
        remaining -= overhead + len(text)
    return result
