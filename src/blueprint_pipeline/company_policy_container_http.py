"""Authenticated, no-spend HTTP admission for company policy containers."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .company_policy_container_admission import (
    CompanyPolicyContainerAdmissionError,
    stage_company_policy_container_admission,
)


def register_company_policy_container_routes(
    app: FastAPI, *, require_admission: Callable,
    admission_root: Callable[[], Path],
) -> None:
    @app.post(
        "/api/live-pipeline/company-policy-containers",
        dependencies=[Depends(require_admission)],
    )
    async def intake_company_policy_container(request: Request) -> JSONResponse:
        """Retain a secret-free company contract without granting launch authority."""

        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        root = admission_root()
        try:
            receipt = await run_in_threadpool(
                stage_company_policy_container_admission,
                value=payload,
                root=root,
            )
        except CompanyPolicyContainerAdmissionError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "schema_version": "company_policy_container_admission_rejection.v1",
                    "status": "rejected",
                    "accepted": False,
                    "blockers": list(exc.blockers),
                    "launch_authority_granted": False,
                    "provider_mutation_authorized": False,
                    "provider_mutation_performed": False,
                },
            )
        return JSONResponse(status_code=200 if receipt["already_exists"] else 201, content=receipt)

