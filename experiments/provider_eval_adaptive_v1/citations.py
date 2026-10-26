"""Bounded offline display citations; unsupported sources are quarantined."""

import re
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from experiments.provider_eval_recovery.adapters import Limits, normalize
from experiments.provider_eval_recovery.harness import digest

TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id"}
PAGINATION = re.compile(r"(?:page|p|page_number|offset|start|[a-f0-9]{8}_page)")
SAFE_TRACKING = re.compile(r"[A-Za-z0-9_.-]{1,80}")
SAFE_ANCHOR = re.compile(r"[A-Za-z][A-Za-z0-9_:.-]{0,79}")
PRIVATE_MARKERS = re.compile(r"token|secret|password|credential|api[-_]?key|auth|signature|redirect|session", re.I)


class ProviderInputWarning(ValueError):
    pass


class CitationNormalizationError(ValueError):
    pass


def validated_url(mode, url):
    if (not isinstance(url, str) or len(url) > 2048 or re.search(r"[\s\\\x00-\x1f\x7f]", url)):
        raise CitationNormalizationError("invalid_citation_url_syntax")
    try:
        parsed = urlsplit(url)
        # Reject before stripping anything. No userinfo, private marker, encoded
        # delimiter, signed query or redirect target can be repaired into safety.
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or "@" in parsed.netloc
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("unsafe_authority")
        parsed.port  # rejects malformed ports
        base = urlunsplit(parsed._replace(query="", fragment=""))
        normalize(mode, {"results": [{"url": base, "title": "", "excerpts": [], "snippet": ""}]})
        # The existing BMW locale selector remains exact and unchanged.
        if not parsed.fragment:
            try:
                normalize(mode, {"results": [{"url": url, "title": "", "excerpts": [], "snippet": ""}]})
                return url, "existing_public_citation"
            except ValueError:
                pass
        # Literal &amp; is handled only in the query separator, never decoded in
        # authority/path. Values and unknown keys remain strictly bounded.
        query = parsed.query.replace("&amp;", "&")
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True, max_num_fields=8) if query else []
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate_query_key")
        kept, dropped = [], False
        for key, value in pairs:
            if PRIVATE_MARKERS.search(key + "=" + value):
                raise ValueError("private_query_marker")
            if key in TRACKING and SAFE_TRACKING.fullmatch(value):
                dropped = True
            elif PAGINATION.fullmatch(key) and re.fullmatch(r"[0-9]{1,6}", value):
                kept.append(key + "=" + value)
            else:
                raise ValueError("unsupported_query")
        if parsed.fragment and (not SAFE_ANCHOR.fullmatch(parsed.fragment) or PRIVATE_MARKERS.search(parsed.fragment)):
            raise ValueError("unsupported_fragment")
        display = urlunsplit(parsed._replace(query="&".join(kept), fragment=""))
        old_chef = (parsed.scheme == "https" and parsed.netloc.lower() == "www.chefrobotics.ai"
                    and parsed.path in ("", "/") and re.fullmatch(r"3f5a3c8b_page=[0-9]{1,6}", parsed.query)
                    and not parsed.fragment)
        rule = ("chef_numeric_pagination" if old_chef else
                "bounded_tracking_anchor_removed" if dropped or parsed.fragment else "numeric_pagination")
        return display, rule
    except (ValueError, TypeError) as exc:
        raise CitationNormalizationError("local_citation_validation_failed") from exc


def audit_urls(mode, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise CitationNormalizationError("invalid_source_results_shape")
    report = []
    for rank, result in enumerate(raw["results"][:10], 1):
        if not isinstance(result, dict):
            raise CitationNormalizationError("invalid_source_result_shape")
        url = result.get("url")
        try:
            display, rule = validated_url(mode, url)
            row = {"url": display, "rule": rule}
            if display != url:
                row.update(raw_url_sha256=digest(url), rank=rank)
        except CitationNormalizationError:
            # Neither unsafe URL nor its accompanying title/text reaches Sol.
            # Original source bytes remain in the digest-bound raw envelope.
            row = {"rule": "quarantined_unsupported_or_unsafe_citation", "rank": rank,
                   "raw_url_sha256": digest(url)}
        report.append(row)
    return report


def normalize_public(mode, raw):
    audit = audit_urls(mode, raw)  # audit all ten before clipping evidence
    result, remaining = [], 6000
    for source, row in zip(raw["results"][:10], audit):
        if "url" not in row:
            continue
        url = row["url"]
        parsed = urlsplit(url)
        # Generic numeric pagination is validated above. Legacy parser receives
        # its query-free URL; restore the validated display URL afterwards.
        checked = urlunsplit(parsed._replace(query="")) if row["rule"] != "existing_public_citation" else url
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
