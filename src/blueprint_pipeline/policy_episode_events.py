"""Native observation/actuator seams shared by legacy and RoboLab drivers.

The generator retains the scientific reset, exact request/media retention,
actuator guards and independent simulator scorer. A driver schedules inference
and each real native step; it never receives privileged scoring observations.
"""
from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyQuery:
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class NativeStep:
    action: Sequence[float]


def drive_episode_events(events: Generator, *, policy: Any, backend: str | None,
                         prompt: str, maximum_steps: int, control_hz: float,
                         action_dimension: int) -> dict[str, Any]:
    if backend == "robolab.v1":
        from .beta_robolab_runtime import run_robolab_episode

        return run_robolab_episode(events, policy=policy, prompt=prompt,
                                   maximum_steps=maximum_steps, control_hz=control_hz,
                                   action_dimension=action_dimension)
    if backend is not None:
        raise ValueError("policy_episode_backend_not_admitted")
    try:
        event = next(events)
        while True:
            if isinstance(event, PolicyQuery):
                try:
                    response = policy.infer(event.observation)
                except Exception as exc:
                    event = events.throw(exc)
                else:
                    event = events.send(response)
            elif isinstance(event, NativeStep):
                event = events.send(None)
            else:
                raise ValueError("policy_episode_unknown_event")
    except StopIteration as completed:
        return completed.value
    finally:
        events.close()


def run_droid_policy_episode(*, evaluation_backend: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Public DROID entrypoint, retaining the native task and evidence engine."""
    try:
        import adp009d_policy_episode as native
    except ModuleNotFoundError:
        from . import adp009d_policy_episode as native

    maximum_steps = int(kwargs.get("max_policy_queries", native.DEFAULT_MAX_POLICY_QUERIES)) * int(
        kwargs.get("open_loop_horizon", native.DROID_OPEN_LOOP_HORIZON)
    ) + int(kwargs.get("settle_window_samples", native.SETTLE_WINDOW_SAMPLES))
    return drive_episode_events(
        native._policy_episode_events(**kwargs), policy=kwargs["policy"],
        backend=evaluation_backend, prompt=str(kwargs["prompt"]),
        maximum_steps=maximum_steps, control_hz=native.DROID_CONTROL_HZ, action_dimension=8,
    )
