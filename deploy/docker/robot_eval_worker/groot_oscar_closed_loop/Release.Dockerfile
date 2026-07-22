# syntax=docker/dockerfile:1.6
# Frequently pulled Blueprint release.  The digest-pinned foundation is cached
# on GPU hosts and the external model volume is mounted at /models.
ARG FOUNDATION_IMAGE
FROM ${FOUNDATION_IMAGE}
USER root
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256
ARG FOUNDATION_MODEL_ASSETS=external
ARG RUNPOD_SERVERLESS_SDK_VERSION=1.10.1
COPY pyproject.toml README.md LICENSE /tmp/blueprint-release/
COPY src /tmp/blueprint-release/src
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_runpod_serverless.lock /tmp/requirements_runpod_serverless.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_runpod_serverless_sdk.lock /tmp/requirements_runpod_serverless_sdk.lock
COPY --chmod=0755 deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh /opt/blueprint/thin_release_entrypoint.sh
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/groot_oscar_closed_loop_image_healthcheck.py /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py
RUN mkdir -p /opt/blueprint/release-src \
  && cp -a /tmp/blueprint-release/src/blueprint_pipeline /opt/blueprint/release-src/ \
  && for python in /opt/oscar-venv/bin/python /opt/gr00t-venv/bin/python /isaac-sim/python.sh; do \
       site_packages="$(${python} -c 'import site; print(site.getsitepackages()[0])')" \
       && printf '%s\n' "import sys; sys.path.insert(0, '/opt/blueprint/release-src')" > "${site_packages}/blueprint_release_override.pth" \
       && "${python}" -c "import blueprint_pipeline; assert blueprint_pipeline.__file__.startswith('/opt/blueprint/release-src/')" \
       || exit 1; \
     done \
  && /opt/oscar-venv/bin/python -m venv /opt/runpod-serverless-venv \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements_runpod_serverless.lock \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements_runpod_serverless_sdk.lock \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-deps /tmp/blueprint-release \
  && rm -rf /opt/runpod-serverless-venv/lib/python*/site-packages/pip* \
              /opt/runpod-serverless-venv/lib/python*/site-packages/setuptools* \
              /opt/runpod-serverless-venv/lib/python*/site-packages/pkg_resources \
  && find /opt/runpod-serverless-venv -type d -name __pycache__ -prune -exec rm -rf '{}' + \
  && rm -rf /tmp/blueprint-release /tmp/requirements_runpod_serverless.lock /tmp/requirements_runpod_serverless_sdk.lock /root/.cache \
  && test -x /opt/oscar-venv/bin/blueprint-run-robot-eval-worker \
  && /opt/runpod-serverless-venv/bin/python -m blueprint_pipeline.groot_oscar_runpod_serverless_worker --verify-serverless-runtime \
  && case "${FOUNDATION_MODEL_ASSETS}" in \
       external) test ! -e /opt/blueprint/ckpts ;; \
       embedded) \
         test -d /opt/blueprint/ckpts \
         && BLUEPRINT_SOURCE_COMMIT="${BLUEPRINT_SOURCE_COMMIT}" \
            BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256="${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256}" \
            /opt/oscar-venv/bin/python /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time ;; \
       *) echo "invalid FOUNDATION_MODEL_ASSETS=${FOUNDATION_MODEL_ASSETS}" >&2; exit 2 ;; \
     esac
ENV BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256} \
    BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS=${FOUNDATION_MODEL_ASSETS} \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_SIM_MAJOR_VERSION=6 \
    BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true
USER blueprint
WORKDIR /workspace
ENTRYPOINT ["/opt/blueprint/thin_release_entrypoint.sh"]
CMD ["bash"]
