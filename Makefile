# Canonical CPU / no-GPU developer workflow.
# CPU-only and no-spend: no target here launches a GPU or paid cloud pod.
# See docs/DEV_SETUP.md for the full guide.

PYTHON ?= .venv/bin/python

.PHONY: help setup verify-env test test-collect test-cpu-subset

help:
	@echo "Canonical CPU / no-GPU targets (see docs/DEV_SETUP.md):"
	@echo "  make setup           - uv sync --extra dev (full no-GPU stack: pxr+mujoco+trimesh+boto3)"
	@echo "  make verify-env      - import-probe the full CPU stack"
	@echo "  make test            - run the full CPU test suite ($(PYTHON) -m pytest tests/ -q)"
	@echo "  make test-collect    - collect-only; must report 0 collection errors"
	@echo "  make test-cpu-subset - fast representative no-GPU green subset"

setup:
	uv sync --extra dev

verify-env:
	$(PYTHON) -c 'import pxr, mujoco, trimesh, PIL, numpy, boto3; print("full CPU env ok")'
	$(PYTHON) -m pytest tests/test_cpu_env_contract.py -q

test:
	$(PYTHON) -m pytest tests/ -q

test-collect:
	$(PYTHON) -m pytest --collect-only -q tests/

test-cpu-subset:
	$(PYTHON) -m pytest \
	  tests/test_local_render_preview.py \
	  tests/test_isaac_g1_kitchen_parity_runner.py \
	  tests/test_scene_placement.py \
	  tests/test_placement_validation.py \
	  tests/test_render_visual_qc.py \
	  tests/test_isaac_g1_kitchen_parity_job.py \
	  tests/test_warm_render_server.py \
	  -q
