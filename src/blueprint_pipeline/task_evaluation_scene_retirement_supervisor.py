"""Fence a fixed installed worker before importing or running its original module.

This conservative launch boundary holds SH for the entire process entrypoint.
A daemon therefore keeps retirement while alive. It does not attest older loaded
code, clear unknown/manual/external readers, or substitute for reference checks.
Absent/disabled policy preserves the original module arguments and exit status.
"""
from __future__ import annotations

import argparse
import runpy
import sys

from .task_evaluation_scene_retirement_access import scene_access


_WORKERS = frozenset('blueprint_pipeline.' + name for name in (
    'pubsub_handoff_listener',
    'task_evaluation_scene_progression',
    'task_evaluation_launch_preparation_worker',
    'task_evaluation_launch_activation_worker',
    'task_evaluation_episode_compilation_worker',
    'task_evaluation_sam31_preparation_execution',
    'task_evaluation_launch_dispatcher',
    'task_evaluation_launch_reconciler',
    'task_evaluation_launch_supervisor',
    'task_evaluation_policy_canary_dispatcher',
    'task_evaluation_terminal_resource_release',
    'operator_policy_canary_continuation',
))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, choices=sorted(_WORKERS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    arguments = selected.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    original_argv = sys.argv
    # Acquire before target import, not after it has opened local artifacts or
    # loaded an unfenced worker. Existing SH reader/publisher locks nest safely.
    with scene_access():
        try:
            sys.argv = [selected.worker, *arguments]
            runpy.run_module(selected.worker, run_name='__main__', alter_sys=True)
        finally:
            sys.argv = original_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
