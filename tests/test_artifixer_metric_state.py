from __future__ import annotations

import pytest

from blueprint_pipeline.artifixer_metric_state import bounded_perceptual_metric_state


class Metric:
    full_state_update = False

    def __init__(self):
        self.all_scores = []

    def reset(self):
        self.all_scores = []

    def __call__(self, value):
        self.all_scores.append(value)
        return value


class Trainer:
    def __init__(self, metric):
        self.criterions = {"lpips": metric}

    def get_losses(self, value):
        result = self.criterions["lpips"](value)
        if value == "fail":
            raise RuntimeError("training failed")
        return {"total_loss": result}


def test_scoped_reset_bounds_history_and_preserves_returned_values():
    metric = Metric()
    metric.all_scores = [0] * 20_000
    trainer = Trainer(metric)
    original = Trainer.get_losses
    with bounded_perceptual_metric_state(Trainer, torchmetrics_version="1.9.0") as evidence:
        for value in range(100):
            assert trainer.get_losses(value) == {"total_loss": value}
            assert metric.all_scores == []
            # Validation consumes its returned batch score and may leave one
            # entry; the next loss call must clear that history too.
            assert metric(value) == value
    assert Trainer.get_losses is original
    assert evidence == {"policy": "per_batch_lpips_state_reset.v1",
        "torchmetrics_version": "1.9.0", "loss_calls": 100,
        "max_history_before_reset": 20_000, "max_batch_history": 1,
        "class_method_restored": True}


def test_failure_clears_graph_references_and_restores_class_method():
    metric = Metric()
    original = Trainer.get_losses
    with pytest.raises(RuntimeError, match="training failed"):
        with bounded_perceptual_metric_state(Trainer, torchmetrics_version="1.8.2"):
            Trainer(metric).get_losses("fail")
    assert metric.all_scores == []
    assert Trainer.get_losses is original


def test_unknown_metric_version_does_not_patch_trainer():
    original = Trainer.get_losses
    with pytest.raises(ValueError, match="version_unverified"):
        with bounded_perceptual_metric_state(Trainer, torchmetrics_version="future"):
            pytest.fail("must refuse first")
    assert Trainer.get_losses is original


def test_unknown_metric_state_contract_refuses_before_loss():
    metric = Metric()
    metric.full_state_update = True
    with pytest.raises(ValueError, match="contract_unverified"):
        with bounded_perceptual_metric_state(Trainer, torchmetrics_version="1.9.0"):
            Trainer(metric).get_losses(1)
    assert metric.all_scores == []


def test_real_metric_keeps_identical_batch_losses_and_input_gradients(monkeypatch):
    torchmetrics = pytest.importorskip("torchmetrics")
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    # Exercise the real Metric/LPIPS state machinery without downloading VGG.
    class Net(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x, y, normalize=False):
            return (x-y).square().mean(dim=(1, 2, 3), keepdim=True)

    monkeypatch.setattr("torchmetrics.image.lpip._NoTrainLpips", Net)

    class RealTrainer:
        def __init__(self):
            self.criterions = {"lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)}

        def get_losses(self, x, y):
            return self.criterions["lpips"](x, y)

    original = RealTrainer.get_losses
    baseline = RealTrainer()
    observed = []
    for i in range(25):
        x = torch.full((1, 3, 32, 32), .1+i/100, requires_grad=True)
        y = torch.full_like(x, .8)
        loss = baseline.get_losses(x, y)
        loss.backward()
        observed.append((loss.detach(), x.grad.clone()))
    assert len(baseline.criterions["lpips"].all_scores) == 25

    fixed = RealTrainer()
    with bounded_perceptual_metric_state(RealTrainer, torchmetrics_version=torchmetrics.__version__):
        for i, (expected_loss, expected_gradient) in enumerate(observed):
            x = torch.full((1, 3, 32, 32), .1+i/100, requires_grad=True)
            y = torch.full_like(x, .8)
            loss = fixed.get_losses(x, y)
            assert fixed.criterions["lpips"].all_scores == []
            loss.backward()
            assert torch.equal(loss.detach(), expected_loss)
            assert torch.equal(x.grad, expected_gradient)
    assert RealTrainer.get_losses is original
