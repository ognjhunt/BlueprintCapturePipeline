"""Fixed report-only CLI: one context acquisition and one invocation budget."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import PurePosixPath

from .control_plane_reference_budget import ReferenceCollectionBudget
from .task_evaluation_scene_lifecycle_acquisition import Acquisition, ContextAnchor, path, require
from .task_evaluation_scene_lineage_budget import _work_items, _work_parse
from .task_evaluation_scene_preparation_lineage import _pairs, _numeric


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('scene_lifecycle_cli_arguments_invalid')
    def exit(self, status=0, message=None):
        raise ValueError('scene_lifecycle_cli_arguments_invalid')


def main(argv=None, *, monotonic=time.monotonic):
    from .task_evaluation_scene_lifecycle_plan import _build_scene_lifecycle_plan, fallback
    budget, reader, anchor, result = None, None, None, None
    try:
        budget = ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=monotonic)
        budget.tick()
        if argv is None:
            require(isinstance(sys.argv, list) and len(sys.argv) <= 17, 'cli_arguments_invalid')
            argv = sys.argv[1:]
        require(isinstance(argv, list) and len(argv) <= 16, 'cli_arguments_invalid')
        for value in _work_items(argv, budget):
            require(type(value) is str and len(value) <= 4096, 'cli_arguments_invalid')
        budget.measure(argv)
        parser = _Parser(add_help=False, allow_abbrev=False)
        parser.add_argument('--intent-id', required=True)
        parser.add_argument('--context-file', required=True)
        parser.add_argument('--now', type=float, required=True)
        args = parser.parse_args(argv)
        context_path = path(args.context_file, budget)
        reader = Acquisition(budget, [str(PurePosixPath(context_path).parent)])
        raw = reader.read_json(context_path)
        budget.preflight(raw.decode('utf-8'))
        context = _work_parse(budget, json.loads, raw.decode('utf-8'), object_pairs_hook=_pairs,
                              parse_int=_numeric, parse_float=_numeric,
                              parse_constant=lambda _: require(False, 'context_json_invalid'))
        anchor = ContextAnchor(reader, context_path)
        result = _build_scene_lifecycle_plan(intent_id=args.intent_id, context=context,
            observed_at_epoch=args.now, budget=budget, context_anchor=anchor)
    except (ValueError, OSError, TypeError, KeyError, AttributeError, UnicodeError, OverflowError, RecursionError):
        result = fallback(budget.failure if budget is not None and budget.failure else 'scene_lifecycle_cli_input_unproven')
    finally:
        if reader is not None:
            try:
                reader.close()
            except ValueError:
                result = fallback('scene_lifecycle_descriptor_cleanup_unproven')
                if anchor is not None:
                    anchor.serialized = None
        if budget is not None:
            budget.close()
    serialized = anchor.serialized if anchor is not None else None
    # Only fixed bounded refusals reach this serialization after budget closure.
    if serialized is None:
        serialized = json.dumps(result, sort_keys=True, separators=(',', ':'))
    sys.stdout.write(serialized+'\n')
    return 0 if 'historical_lineage' in result else 2
