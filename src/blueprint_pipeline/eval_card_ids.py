"""Identifier extraction for eval-card artifacts (task/scenario cards,
scoring methodology, task thresholds, failure-mode taxonomy).

Extracted from capture_orchestrator so the orchestrator stays inside its
source-governance line budget; behavior is unchanged and the orchestrator
re-exports these names for compatibility with existing callers and tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _read_json_mapping(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_cards(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json_mapping(path)
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return []
    return [dict(card) for card in cards if isinstance(card, Mapping)]


def _metric_ids_from_methodology(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_metrics = (
        payload.get("metrics")
        or payload.get("scorecard_metrics")
        or payload.get("standard_scorecard_metrics")
    )
    metric_ids: set[str] = set()
    if isinstance(raw_metrics, Mapping):
        for key, value in raw_metrics.items():
            if isinstance(value, Mapping):
                metric_id = value.get("metric_id") or value.get("metricId") or value.get("id")
            else:
                metric_id = key
            metric_id = str(metric_id or "").strip()
            if metric_id:
                metric_ids.add(metric_id)
        return metric_ids
    if not isinstance(raw_metrics, list):
        return metric_ids
    for metric in raw_metrics:
        if isinstance(metric, Mapping):
            metric_id = metric.get("metric_id") or metric.get("metricId") or metric.get("id")
        else:
            metric_id = metric
        metric_id = str(metric_id or "").strip()
        if metric_id:
            metric_ids.add(metric_id)
    return metric_ids


def _task_ids_from_thresholds(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_thresholds = (
        payload.get("task_thresholds")
        or payload.get("taskThresholds")
        or payload.get("thresholds")
    )
    task_ids: set[str] = set()
    if isinstance(raw_thresholds, Mapping):
        for key, value in raw_thresholds.items():
            if isinstance(value, Mapping):
                task_id = value.get("task_id") or value.get("taskId") or value.get("id")
            else:
                task_id = key
            task_id = str(task_id or "").strip()
            if task_id:
                task_ids.add(task_id)
        return task_ids
    if not isinstance(raw_thresholds, list):
        return task_ids
    for threshold in raw_thresholds:
        if not isinstance(threshold, Mapping):
            continue
        task_id = threshold.get("task_id") or threshold.get("taskId") or threshold.get("id")
        task_id = str(task_id or "").strip()
        if task_id:
            task_ids.add(task_id)
    return task_ids


def _failure_mode_ids_from_taxonomy(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_modes = payload.get("failure_modes") or payload.get("failureModes")
    mode_ids: set[str] = set()
    if not isinstance(raw_modes, list):
        return mode_ids
    for mode in raw_modes:
        if isinstance(mode, Mapping):
            mode_id = mode.get("failure_mode_id") or mode.get("failureModeId") or mode.get("id")
        else:
            mode_id = mode
        mode_id = str(mode_id or "").strip()
        if mode_id:
            mode_ids.add(mode_id)
    return mode_ids
