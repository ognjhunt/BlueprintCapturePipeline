FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /opt/content-agents
COPY . /opt/content-agents
RUN uv venv /opt/content-agents/.runtime-venv \
    && uv pip install --python /opt/content-agents/.runtime-venv/bin/python \
       -e . -e apps/material_agent -e apps/physics_agent \
       -e apps/texture_agent -e apps/validation_agent

ENV PATH="/opt/content-agents/.runtime-venv/bin:${PATH}"
