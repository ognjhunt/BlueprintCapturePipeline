FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.3
COPY src/blueprint_pipeline/controlled_policy_actions.py /opt/blueprint_pipeline/controlled_policy_actions.py
COPY src/blueprint_pipeline/company_policy_proxy.py /opt/blueprint_pipeline/company_policy_proxy.py
COPY deploy/docker/company_policy_synthetic/policy.py /opt/policy.py
COPY deploy/docker/company_policy_synthetic/entrypoint.py /opt/entrypoint.py
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "/opt/entrypoint.py"]
