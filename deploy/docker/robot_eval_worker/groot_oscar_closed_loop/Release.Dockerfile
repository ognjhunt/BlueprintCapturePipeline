# syntax=docker/dockerfile:1.6
# Frequently pulled Blueprint release.  The digest-pinned foundation is cached
# on GPU hosts and the external model volume is mounted at /models.
ARG FOUNDATION_IMAGE
FROM ${FOUNDATION_IMAGE}
USER root
ARG BLUEPRINT_SOURCE_COMMIT
ARG BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256
COPY pyproject.toml README.md LICENSE /tmp/blueprint-release/
COPY src /tmp/blueprint-release/src
COPY --chmod=0755 deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh /opt/blueprint/thin_release_entrypoint.sh
COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/groot_oscar_closed_loop_image_healthcheck.py /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py
RUN /opt/oscar-venv/bin/python -m pip install --no-deps /tmp/blueprint-release \
  && /opt/gr00t-venv/bin/python -m pip install --no-deps /tmp/blueprint-release \
  && /isaac-sim/python.sh -m pip install --no-deps /tmp/blueprint-release \
  && rm -rf /tmp/blueprint-release /root/.cache \
  && test -x /opt/oscar-venv/bin/blueprint-run-robot-eval-worker \
  && test ! -e /opt/blueprint/ckpts
ENV BLUEPRINT_SOURCE_COMMIT=${BLUEPRINT_SOURCE_COMMIT} \
    BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=${BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256} \
    BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker \
    BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release \
    BLUEPRINT_SIMULATOR_FRAMEWORK=isaac_sim \
    BLUEPRINT_ISAAC_SIM_MAJOR_VERSION=6 \
    BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true
USER blueprint
WORKDIR /workspace
ENTRYPOINT ["/opt/blueprint/thin_release_entrypoint.sh"]
CMD ["bash"]
