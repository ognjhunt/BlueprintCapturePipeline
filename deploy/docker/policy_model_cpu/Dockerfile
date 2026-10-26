FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
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
