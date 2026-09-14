"""Bound unused LPIPS metric history without changing per-batch loss tensors."""
from __future__ import annotations

from contextlib import contextmanager
import json


@contextmanager
def bounded_perceptual_metric_state(trainer_class, *, torchmetrics_version):
    """Scope the adapter to the caller's already verified released trainer.

    These versions return a current-batch value from forward, then retain a
    separate cumulative score list. The trainer uses those batch values for
    both loss and validation; it never consumes the cumulative compute state.
    """
    if torchmetrics_version not in {"1.8.2", "1.9.0"}:
        raise ValueError("artifixer_metric_state_version_unverified")
    original = trainer_class.get_losses
    evidence = {"policy": "per_batch_lpips_state_reset.v1",
                "torchmetrics_version": torchmetrics_version, "loss_calls": 0,
                "max_history_before_reset": 0, "max_batch_history": 0,
                "class_method_restored": False}

    def get_losses(trainer, *args, **kwargs):
        metric = trainer.criterions["lpips"]
        if (getattr(metric, "full_state_update", None) is not False
                or not isinstance(getattr(metric, "all_scores", None), list)
                or not callable(getattr(metric, "reset", None))):
            raise ValueError("artifixer_metric_state_contract_unverified")
        evidence["max_history_before_reset"] = max(
            evidence["max_history_before_reset"], len(metric.all_scores))
        metric.reset()
        try:
            result = original(trainer, *args, **kwargs)
            evidence["loss_calls"] += 1
            evidence["max_batch_history"] = max(evidence["max_batch_history"], len(metric.all_scores))
            if evidence["loss_calls"] == 1 or evidence["loss_calls"] % 1000 == 0:
                print("BLUEPRINT_ARTIFIXER_METRIC_STATE:" + json.dumps(evidence, sort_keys=True), flush=True)
            return result
        finally:
            metric.reset()

    trainer_class.get_losses = get_losses
    try:
        yield evidence
    finally:
        trainer_class.get_losses = original
        evidence["class_method_restored"] = True
