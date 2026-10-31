FROM python:3.15-slim@sha256:94d3426f1def7db4a106a232ac121bfd975393e5dfd40a619b955fae45fe7913
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.3
COPY src/blueprint_pipeline/controlled_policy_actions.py /opt/blueprint_pipeline/controlled_policy_actions.py
COPY src/blueprint_pipeline/company_policy_proxy.py /opt/blueprint_pipeline/company_policy_proxy.py
COPY deploy/docker/company_policy_synthetic/entrypoint.py /opt/entrypoint.py
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "/opt/entrypoint.py"]
