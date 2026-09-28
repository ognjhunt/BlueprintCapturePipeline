FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir onnx==1.23.0 onnxruntime==1.30.0 fastapi==0.115.12 uvicorn==0.34.3 rfc8785==0.1.4
COPY src/blueprint_pipeline/policy_model_onnx.py /opt/blueprint_pipeline/policy_model_onnx.py
COPY src/blueprint_pipeline/policy_model_server.py /opt/blueprint_pipeline/policy_model_server.py
COPY src/blueprint_pipeline/controlled_policy_actions.py /opt/blueprint_pipeline/controlled_policy_actions.py
COPY src/blueprint_pipeline/company_policy_proxy.py /opt/blueprint_pipeline/company_policy_proxy.py
COPY src/blueprint_pipeline/company_policy_container_contract_v2.py /opt/blueprint_pipeline/company_policy_container_contract_v2.py
COPY src/blueprint_pipeline/decision_evidence_contracts.py /opt/blueprint_pipeline/decision_evidence_contracts.py
COPY --chown=65532:65532 policy-model-build-input/policy.onnx /opt/policy-model/policy.onnx
COPY --chown=65532:65532 policy-model-build-input/artifact.json /opt/policy-model/artifact.json
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
CMD ["python", "-m", "blueprint_pipeline.policy_model_server"]
