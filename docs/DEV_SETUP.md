# CPU Development Setup

Use Python 3.12 for the canonical local environment.

```bash
uv sync --extra dev
```

Equivalent pip fallback:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Verify the no-GPU stack before running dry-render, USD placement, or MuJoCo
tests:

```bash
python -m blueprint_pipeline.cpu_env_doctor
python -c 'import PIL, pxr, mujoco, trimesh, collada, boto3, blueprint_pipeline, blueprint_contracts; print("full CPU env ok")'
.venv/bin/python -m pytest tests/test_cpu_env_contract.py -q
```

`google-genai`, OpenAI, and other LLM/provider extras remain optional for CPU
validation. Tests that cover those paths use mocks unless a command explicitly
requests live provider access.
