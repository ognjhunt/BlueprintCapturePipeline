# syntax=docker/dockerfile:1.6
# Frequently pulled Blueprint release.  The digest-pinned foundation is cached
# on GPU hosts and the external model volume is mounted at /models.
ARG FOUNDATION_IMAGE
FROM ${FOUNDATION_IMAGE}
USER root
ENV PYTHONDONTWRITEBYTECODE=1
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256
ARG FOUNDATION_MODEL_ASSETS=external
ARG RUNPOD_SERVERLESS_SDK_VERSION=1.10.1
ARG EMBEDDED_FOUNDATION_WBC_SOURCE_REF=6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b
ARG EMBEDDED_FOUNDATION_GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392
ARG EMBEDDED_FOUNDATION_OSCAR_SOURCE_REF=4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb
COPY pyproject.toml README.md LICENSE /tmp/blueprint-release/
COPY src /tmp/blueprint-release/src
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_runpod_serverless.lock /tmp/requirements_runpod_serverless.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_runpod_serverless_sdk.lock /tmp/requirements_runpod_serverless_sdk.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_embedded_carrier_opencv.lock /tmp/requirements_embedded_carrier_opencv.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_oscar_foundation.lock /tmp/requirements_oscar_foundation.lock
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/repair_embedded_carrier.py /tmp/repair_embedded_carrier.py
COPY --chmod=0755 deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh /opt/blueprint/thin_release_entrypoint.sh
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/groot_oscar_closed_loop_image_healthcheck.py /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/oscar_cpu_import_probe.py /opt/blueprint/oscar_cpu_import_probe.py
RUN mkdir -p /opt/blueprint/release-src \
  && cp -a /tmp/blueprint-release/src/blueprint_pipeline /opt/blueprint/release-src/ \
  && for python in /opt/oscar-venv/bin/python /opt/gr00t-venv/bin/python /isaac-sim/python.sh; do \
       site_packages="$(${python} -c 'import site; print(site.getsitepackages()[0])')" \
       && printf '%s\n' "import sys; sys.path.insert(0, '/opt/blueprint/release-src')" > "${site_packages}/blueprint_release_override.pth" \
       && "${python}" -c "import blueprint_pipeline; assert blueprint_pipeline.__file__.startswith('/opt/blueprint/release-src/')" \
       || exit 1; \
     done \
  && printf '%s\n' '#!/opt/oscar-venv/bin/python' \
       'from blueprint_pipeline.robot_eval_worker import main' \
       'raise SystemExit(main())' \
       > /opt/oscar-venv/bin/blueprint-run-robot-eval-worker \
  && chmod 0755 /opt/oscar-venv/bin/blueprint-run-robot-eval-worker \
  && rm -rf /opt/wbc/gear_sonic_deploy/build \
  && /opt/oscar-venv/bin/python -m venv /opt/runpod-serverless-venv \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements_runpod_serverless.lock \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements_runpod_serverless_sdk.lock \
  && /opt/runpod-serverless-venv/bin/python -m pip install --no-deps /tmp/blueprint-release \
  && if [ "${FOUNDATION_MODEL_ASSETS}" = embedded ]; then \
       /opt/runpod-serverless-venv/bin/python -m pip --python /opt/oscar-venv/bin/python install \
         --no-cache-dir --require-hashes \
       --index-url https://download.pytorch.org/whl/cu128 \
         --extra-index-url https://pypi.org/simple \
         -r /tmp/requirements_oscar_foundation.lock \
       && oscar_site_packages="$(/opt/oscar-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
       && /opt/oscar-venv/bin/python -c "from pathlib import Path; from blueprint_pipeline.oscar_wam_gpu_image import transformer_engine_shim_script_text; Path('/tmp/install_transformer_engine_shim.py').write_text(transformer_engine_shim_script_text(), encoding='utf-8')" \
       && /opt/oscar-venv/bin/python /tmp/install_transformer_engine_shim.py "${oscar_site_packages}" \
       && rm -rf "${oscar_site_packages}/transformer_engine" \
       && test -d "${oscar_site_packages}/transformer_engine-2.0.0.dist-info" \
       && test ! -e "${oscar_site_packages}/transformer_engine" \
       && /opt/oscar-venv/bin/python /tmp/install_transformer_engine_shim.py /opt/oscar-public \
       && find /opt/oscar-public -type d -name __pycache__ -prune -exec rm -rf '{}' + \
       && find /opt/oscar-public -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
       && PYTHONPATH=/opt/blueprint/release-src \
          /opt/oscar-venv/bin/python -m blueprint_pipeline.oscar_runtime_source_provenance normalize \
            --source-root /opt/oscar-public \
            --existing-seal /opt/blueprint/oscar_source_provenance.json \
            --output /opt/blueprint/oscar_source_provenance.json \
            --runtime-source-root /opt/oscar-public \
       && chmod 0444 /opt/blueprint/oscar_source_provenance.json \
       && /opt/oscar-venv/bin/python /tmp/repair_embedded_carrier.py \
         --wbc-revision "${EMBEDDED_FOUNDATION_WBC_SOURCE_REF}" \
         --groot-revision "${EMBEDDED_FOUNDATION_GROOT_SOURCE_REF}" \
         --oscar-revision "${EMBEDDED_FOUNDATION_OSCAR_SOURCE_REF}" \
         --output /opt/blueprint/embedded_carrier_repair.json \
       && test -L /opt/OSCAR \
       && test "$(readlink -f /opt/OSCAR)" = /opt/oscar-public \
       && rm /opt/OSCAR \
       && mv /opt/oscar-public /opt/OSCAR \
       && test -d /opt/OSCAR \
       && test ! -L /opt/OSCAR \
       && find /opt/OSCAR -type d -name __pycache__ -prune -exec rm -rf '{}' + \
       && find /opt/OSCAR -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
       && PYTHONPATH=/opt/blueprint/release-src \
          /opt/oscar-venv/bin/python -m blueprint_pipeline.oscar_runtime_source_provenance normalize \
            --source-root /opt/OSCAR \
            --existing-seal /opt/blueprint/oscar_source_provenance.json \
            --output /opt/blueprint/oscar_source_provenance.json \
            --runtime-source-root /opt/OSCAR \
       && chmod 0444 /opt/blueprint/oscar_source_provenance.json \
       && printf '%s\n' '/opt/wbc' > "${oscar_site_packages}/blueprint_gear_sonic_runtime.pth" \
       && rm -f /usr/local/cuda*/bin/nvcc \
       && rm -rf /usr/local/cuda*/include /usr/local/cuda*/lib64/stubs /usr/local/cuda*/targets/*/include /usr/local/cuda*/targets/*/lib/stubs \
       && find /usr/local/cuda* -type f -name '*.a' -delete \
       && rm -rf /opt/wbc/.git /opt/gr00t/.git /opt/OSCAR/.git; \
     fi \
  && oscar_site_packages="$(/opt/oscar-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
  && if ! /opt/oscar-venv/bin/python -c 'import cv2'; then \
       /opt/runpod-serverless-venv/bin/python -m pip install --no-cache-dir --no-deps --require-hashes \
         --target "${oscar_site_packages}" -r /tmp/requirements_embedded_carrier_opencv.lock; \
     fi \
  && /opt/oscar-venv/bin/python -c 'import cv2; assert cv2.__version__' \
  && rm -rf /opt/runpod-serverless-venv/lib/python*/site-packages/pip* \
              /opt/runpod-serverless-venv/lib/python*/site-packages/setuptools* \
              /opt/runpod-serverless-venv/lib/python*/site-packages/pkg_resources \
  && find /opt/runpod-serverless-venv -type d -name __pycache__ -prune -exec rm -rf '{}' + \
  && rm -rf /tmp/blueprint-release /tmp/requirements_runpod_serverless.lock /tmp/requirements_runpod_serverless_sdk.lock /tmp/requirements_embedded_carrier_opencv.lock /tmp/requirements_oscar_foundation.lock /tmp/install_transformer_engine_shim.py /tmp/repair_embedded_carrier.py /root/.cache \
  && test -x /opt/oscar-venv/bin/blueprint-run-robot-eval-worker \
  && /opt/runpod-serverless-venv/bin/python -m blueprint_pipeline.groot_oscar_runpod_serverless_worker --verify-serverless-runtime \
  && case "${FOUNDATION_MODEL_ASSETS}" in \
       external) test ! -e /opt/blueprint/ckpts ;; \
       embedded) \
         test -d /opt/blueprint/ckpts \
         && runuser -u blueprint -- env \
              BLUEPRINT_SOURCE_COMMIT="${BLUEPRINT_SOURCE_COMMIT}" \
              BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256="${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256}" \
              BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release \
              BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS="${FOUNDATION_MODEL_ASSETS}" \
              BLUEPRINT_GEAR_SONIC_SOURCE_REVISION="${EMBEDDED_FOUNDATION_WBC_SOURCE_REF}" \
              PYTHONPATH=/opt/blueprint/release-src:/opt/wbc:/opt/OSCAR \
              /opt/oscar-venv/bin/python /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time ;; \
       *) echo "invalid FOUNDATION_MODEL_ASSETS=${FOUNDATION_MODEL_ASSETS}" >&2; exit 2 ;; \
     esac
ENV BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256} \
    BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8 \
    BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS=${FOUNDATION_MODEL_ASSETS} \
    BLUEPRINT_GEAR_SONIC_SOURCE_REVISION=${EMBEDDED_FOUNDATION_WBC_SOURCE_REF} \
    BLUEPRINT_FOUNDATION_GROOT_SOURCE_REF=${EMBEDDED_FOUNDATION_GROOT_SOURCE_REF} \
    BLUEPRINT_FOUNDATION_OSCAR_SOURCE_REF=${EMBEDDED_FOUNDATION_OSCAR_SOURCE_REF} \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_SIM_MAJOR_VERSION=6 \
    PYTHONPATH=/opt/wbc:/opt/OSCAR \
    BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true
USER blueprint
WORKDIR /workspace
ENTRYPOINT ["/opt/blueprint/thin_release_entrypoint.sh"]
CMD ["bash"]
