#!/usr/bin/env python3
"""Reject unsupported public SC3 accuracy and rank-fidelity claims."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_PUBLIC_SURFACES = (
    "README.md",
    "VISION.md",
)
NON_PUBLIC_DOC_NAME_MARKERS = ("AUDIT", "BACKLOG", "CHANGELOG")

PERCENT_ACCURACY = re.compile(
    r"\b(?:92\.9|93(?:\.0)?)\s*%\s*(?:accuracy|accurate|rank[- ]fidelity)\b",
    re.IGNORECASE,
)
BLUEPRINT_NEAR_METRIC = re.compile(
    r"(?:\bBlueprint(?:'s)?\b[\s\S]{0,120}(?:\b0\.929\b|\b92\.9\s*%)|"
    r"(?:\b0\.929\b|\b92\.9\s*%)[\s\S]{0,120}\bBlueprint(?:'s)?\b)",
    re.IGNORECASE,
)
RANK_OR_CORRELATION = re.compile(r"\b(?:rank[- ]fidelity|correlation)\b", re.IGNORECASE)
SC3_METRIC = re.compile(r"(?:\b0\.929\b|\b92\.9\s*%)", re.IGNORECASE)
EXPLICIT_DISCLAIMER = re.compile(
    r"(?:not (?:a )?Blueprint measurements?|not Blueprint numbers?|"
    r"Blueprint has not measured)",
    re.IGNORECASE,
)
SC3_SOURCE = re.compile(r"\bSC3-Eval\b", re.IGNORECASE)
SC3_FAMILY = re.compile(r"\bSC3(?:-Eval)?\b", re.IGNORECASE)
OSCAR_FAMILY = re.compile(r"\bOSCAR\b", re.IGNORECASE)
PEARSON_METRIC = re.compile(r"\bPearson\b", re.IGNORECASE)
POLICY_SAMPLE = re.compile(r"\b(?:seven|7)\b[\s\S]{0,40}\bpolic(?:y|ies|ies')\b", re.IGNORECASE)
EVALUATION_SCOPE = re.compile(
    r"\b(?:closed-loop|overall|headline|in-distribution|out-of-distribution|OOD)\b",
    re.IGNORECASE,
)
LEGACY_SISR_DELTA = re.compile(
    r"(?:\bSISR(?:[ _-]+)delta\b|\bsisr_delta\b)",
    re.IGNORECASE,
)
# These numerical values are benchmark-card scoped.  They are deliberately
# kept separate from the general 0.929 attribution rule so an OSCAR number
# cannot silently become an SC3 result (or vice versa) in public copy.
OSCAR_BENCHMARK_METRIC = re.compile(
    r"(?:\b(?:0\.571|0\.750|0\.852)\b|"
    r"\bsuccess[_ -]rate[_ -]difference(?:[_ -]pp)?\b[\s:=`]*1\.73\b)",
    re.IGNORECASE,
)
SC3_BENCHMARK_METRIC = re.compile(
    r"(?:\b0\.929\b|\b92\.9\s*%|\bMMRV\b[\s:=`]*0\.119\b)",
    re.IGNORECASE,
)


def _metric_attributed_to_family(
    paragraph: str,
    *,
    family: re.Pattern[str],
    metric: re.Pattern[str],
) -> bool:
    """Return true when a known metric is nearest to the wrong family.

    Nearest-family binding handles both ``SC3 reports 0.852`` and ``0.852 is
    SC3's result`` while permitting honest side-by-side OSCAR/SC3 comparisons.
    """

    for metric_match in metric.finditer(paragraph):
        before = paragraph[: metric_match.start()]
        previous_boundaries = list(re.finditer(r"[;.!?](?:\s|$)", before))
        start = previous_boundaries[-1].end() if previous_boundaries else 0
        after = paragraph[metric_match.end() :]
        next_boundary = re.search(r"[;.!?](?:\s|$)", after)
        end = (
            metric_match.end() + next_boundary.start()
            if next_boundary is not None
            else len(paragraph)
        )
        sentence = paragraph[start:end]
        metric_start = metric_match.start() - start
        metric_end = metric_match.end() - start
        candidates: list[tuple[int, bool]] = []
        for candidate_family in (SC3_FAMILY, OSCAR_FAMILY):
            for match in candidate_family.finditer(sentence):
                if match.end() <= metric_start:
                    distance = metric_start - match.end()
                elif metric_end <= match.start():
                    distance = match.start() - metric_end
                else:
                    distance = 0
                candidates.append((distance, candidate_family is family))
        if candidates and min(candidates, key=lambda item: item[0])[1]:
            return True
    return False


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    code: str
    message: str


def _paragraphs(text: str) -> list[tuple[int, str]]:
    paragraphs: list[tuple[int, str]] = []
    lines = text.splitlines()
    start = 0
    current: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not current:
                start = line_number
            current.append(line)
            continue
        if current:
            paragraphs.append((start, "\n".join(current)))
            current = []
    if current:
        paragraphs.append((start, "\n".join(current)))
    return paragraphs


def lint_path(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    for line, paragraph in _paragraphs(text):
        if LEGACY_SISR_DELTA.search(paragraph):
            findings.append(
                Finding(
                    path,
                    line,
                    "legacy_sisr_delta_metric",
                    "Use OSCAR's benchmark-scoped success_rate_difference_pp metric name.",
                )
            )

        if _metric_attributed_to_family(
            paragraph,
            family=SC3_FAMILY,
            metric=OSCAR_BENCHMARK_METRIC,
        ):
            findings.append(
                Finding(
                    path,
                    line,
                    "oscar_metric_transferred_to_sc3",
                    "Do not attribute OSCAR benchmark-card metrics to SC3-Eval.",
                )
            )

        if _metric_attributed_to_family(
            paragraph,
            family=OSCAR_FAMILY,
            metric=SC3_BENCHMARK_METRIC,
        ):
            findings.append(
                Finding(
                    path,
                    line,
                    "sc3_metric_transferred_to_oscar",
                    "Do not attribute SC3-Eval benchmark-card metrics to OSCAR.",
                )
            )

        if PERCENT_ACCURACY.search(paragraph):
            findings.append(
                Finding(
                    path,
                    line,
                    "unsupported_percent_accuracy",
                    "Do not describe SC3/Blueprint as 93% accurate or equivalent.",
                )
            )

        metric_claim = SC3_METRIC.search(paragraph) and RANK_OR_CORRELATION.search(paragraph)
        if BLUEPRINT_NEAR_METRIC.search(paragraph) and not EXPLICIT_DISCLAIMER.search(paragraph):
            findings.append(
                Finding(
                    path,
                    line,
                    "blueprint_0929_attribution",
                    "0.929 must be attributed to SC3-Eval and disclaimed as a Blueprint measurement.",
                )
            )

        if metric_claim:
            required = (
                (SC3_SOURCE, "sc3_source_missing", "Name SC3-Eval as the source."),
                (PEARSON_METRIC, "pearson_metric_missing", "Name Pearson as the metric."),
                (POLICY_SAMPLE, "policy_sample_missing", "State the seven-policy sample/unit."),
                (EVALUATION_SCOPE, "evaluation_scope_missing", "State the evaluated split or scope."),
                (
                    EXPLICIT_DISCLAIMER,
                    "blueprint_disclaimer_missing",
                    "State that this is not a Blueprint measurement.",
                ),
            )
            for pattern, code, message in required:
                if not pattern.search(paragraph):
                    findings.append(Finding(path, line, code, message))
    return findings


def lint(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.is_file():
            findings.append(Finding(path, 1, "claim_surface_missing", "Claim surface is missing."))
            continue
        findings.extend(lint_path(path))
    return findings


def default_surfaces(*, root: Path = ROOT) -> list[Path]:
    surfaces = [root / item for item in ROOT_PUBLIC_SURFACES]
    for path in sorted((root / "docs").glob("*.md")):
        if any(marker in path.name.upper() for marker in NON_PUBLIC_DOC_NAME_MARKERS):
            continue
        surfaces.append(path)
    return surfaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.paths] or default_surfaces()

    findings = lint(paths)
    if findings:
        for finding in findings:
            try:
                display_path = finding.path.relative_to(ROOT)
            except ValueError:
                display_path = finding.path
            print(
                f"{display_path}:{finding.line}: {finding.code}: {finding.message}",
                file=sys.stderr,
            )
        return 1
    print(f"[public-scientific-claims] ok ({len(paths)} surfaces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
