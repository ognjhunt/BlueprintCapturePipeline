FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.3
COPY src/blueprint_pipeline/controlled_policy_actions.py /opt/blueprint_pipeline/controlled_policy_actions.py
COPY src/blueprint_pipeline/company_policy_proxy.py /opt/blueprint_pipeline/company_policy_proxy.py
COPY deploy/docker/company_policy_synthetic/policy.py /opt/policy.py
COPY deploy/docker/company_policy_synthetic/entrypoint.py /opt/entrypoint.py
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "/opt/entrypoint.py"]
