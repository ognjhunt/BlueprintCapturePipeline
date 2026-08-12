# syntax=docker/dockerfile:1.7
# One-shot OpenPI DROID policy-ranking worker. Checkpoints and InteriorGS-derived
# observations remain external; this image contains code and pinned robot assets.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1@sha256:190cbcca15602bad4531b4310f928fd34a036d50ec3edb6389edc167e21c35b6 /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENPI_SOURCE_URL=https://github.com/Physical-Intelligence/openpi.git
ARG OPENPI_SOURCE_REVISION=15a9616a00943ada6c20a0f158e3adb39df2ccac
ARG MENAGERIE_SOURCE_URL=https://github.com/google-deepmind/mujoco_menagerie.git
ARG MENAGERIE_SOURCE_REVISION=71f066ad0be9cd271f7ed58c030243ef157af9f4
ARG GSUTIL_VERSION=5.37
ARG GSUTIL_ARCHIVE_URL=https://files.pythonhosted.org/packages/9a/6f/5da5e0a8dd3b101809e9f6ba7875facdcd26de42d2ca8fede2ff9769bc39/gsutil-5.37.tar.gz
ARG GSUTIL_ARCHIVE_SHA256=c217df9fba2fb16464ee2e4968ff7d669c9699a69af7003c5deb2da4db8e4298
ARG BLUEPRINT_SOURCE_COMMIT

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    OPENPI_DATA_HOME=/workspace/openpi-data \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    BLUEPRINT_OPENPI_SOURCE_REVISION=${OPENPI_SOURCE_REVISION} \
    BLUEPRINT_MENAGERIE_SOURCE_REVISION=${MENAGERIE_SOURCE_REVISION} \
    BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates clang curl git git-lfs \
      libegl1 libgl1 libglib2.0-0 libglvnd0 libosmesa6 linux-headers-generic \
    && ldconfig \
    && ldconfig -p | grep -F "libgthread-2.0.so.0" \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --recurse-submodules "${OPENPI_SOURCE_URL}" /opt/openpi \
    && git -C /opt/openpi fetch --depth 1 origin "${OPENPI_SOURCE_REVISION}" \
    && git -C /opt/openpi checkout --detach FETCH_HEAD \
    && git -C /opt/openpi submodule update --init --recursive \
    && test "$(git -C /opt/openpi rev-parse HEAD)" = "${OPENPI_SOURCE_REVISION}" \
    && printf '%s\n' "${OPENPI_SOURCE_REVISION}" > /opt/openpi/.blueprint-source-revision \
    && cd /opt/openpi \
    && uv venv --python 3.11.9 "${UV_PROJECT_ENVIRONMENT}" \
    && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev

# OpenPI otherwise falls back to its single-worker fsspec downloader for GCS.
# Keep gsutil isolated from OpenPI's locked environment and bind the exact PyPI
# artifact so the parallel transfer path is reproducible without dependency drift.
RUN curl -fsSL "${GSUTIL_ARCHIVE_URL}" -o /tmp/gsutil.tar.gz \
    && echo "${GSUTIL_ARCHIVE_SHA256}  /tmp/gsutil.tar.gz" | sha256sum -c - \
    && uv venv --python 3.11.9 /opt/gsutil-venv \
    && VIRTUAL_ENV=/opt/gsutil-venv uv pip install /tmp/gsutil.tar.gz \
    && ln -s /opt/gsutil-venv/bin/gsutil /usr/local/bin/gsutil \
    && gsutil version -l | grep -F "gsutil version: ${GSUTIL_VERSION}" \
    && rm -f /tmp/gsutil.tar.gz

RUN git clone --filter=blob:none --no-checkout "${MENAGERIE_SOURCE_URL}" /opt/mujoco-menagerie \
    && git -C /opt/mujoco-menagerie fetch --depth 1 origin "${MENAGERIE_SOURCE_REVISION}" \
    && git -C /opt/mujoco-menagerie sparse-checkout init --cone \
    && git -C /opt/mujoco-menagerie sparse-checkout set franka_emika_panda \
    && git -C /opt/mujoco-menagerie checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/mujoco-menagerie rev-parse HEAD)" = "${MENAGERIE_SOURCE_REVISION}" \
    && printf '%s\n' "${MENAGERIE_SOURCE_REVISION}" > /opt/mujoco-menagerie/.blueprint-source-revision

COPY pyproject.toml README.md LICENSE /opt/blueprint/
COPY src /opt/blueprint/src
COPY docs/experiments/policy_ranking_thesis_20260726/warehouse_policy_cohort_v2_joint_position.json /opt/blueprint/frozen/warehouse_policy_cohort_v2_joint_position.json
COPY docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json /opt/blueprint/frozen/openpi_polaris_checkpoint_inventory.json
COPY docs/experiments/policy_ranking_thesis_20260726/captured_site_ranking_aggregator_v1.json /opt/blueprint/frozen/captured_site_ranking_aggregator_v1.json

# Do not import native x86 JAX/MuJoCo extensions in an ARM-hosted cross-build:
# QEMU does not expose the target CPU/GPU feature set. The worker performs the
# authoritative native imports plus JAX GPU-device gate before checkpoint I/O.
RUN VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}" uv pip install --no-deps -e /opt/blueprint \
    && VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}" uv pip install mujoco==3.9.0 \
    && "${UV_PROJECT_ENVIRONMENT}/bin/python" -m compileall -q /opt/blueprint/src/blueprint_pipeline \
    && "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import importlib.metadata as m, importlib.util as u; assert all(u.find_spec(n) for n in ('jax', 'mujoco', 'openpi')); print(m.version('jax'), m.version('mujoco'), m.version('openpi'))" \
    && test -f /opt/mujoco-menagerie/franka_emika_panda/mjx_scene.xml

WORKDIR /workspace
ENTRYPOINT ["/.venv/bin/python", "-m", "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap", "run"]
