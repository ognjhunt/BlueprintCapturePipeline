"""Serve verified local or remotely retained result bytes with bounded lifetime."""

from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from .control_plane_disk_budget import ControlPlaneDiskBudgetError
from .task_evaluation_configured_scene_object_store import (
    TaskEvaluationConfiguredSceneObjectStoreError,
)
from .live_pipeline_result_artifact_resolution import (
    TaskEvaluationResultDeliveryError,
    resolve_live_pipeline_result_artifact,
)


class ResultArtifactFileResponse(FileResponse):
    """Release a verified temporary artifact on success, error, or disconnect."""

    def __init__(self, *args, artifact_cleanup=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_cleanup = artifact_cleanup

    async def __call__(self, scope, receive, send):
        try:
            return await super().__call__(scope, receive, send)
        finally:
            if self.artifact_cleanup is not None:
                self.artifact_cleanup()


async def result_artifact_response(
    *,
    legacy_state_root: str | Path,
    policy_canary_result_root: str | Path | None,
    run_id: str,
    artifact_id: str,
) -> FileResponse:
    try:
        path, record = await run_in_threadpool(
            resolve_live_pipeline_result_artifact,
            legacy_state_root=legacy_state_root,
            policy_canary_result_root=policy_canary_result_root,
            run_id=run_id,
            artifact_id=artifact_id,
            retain_read_lease=True,
        )
    except (TaskEvaluationResultDeliveryError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        TaskEvaluationConfiguredSceneObjectStoreError,
        ControlPlaneDiskBudgetError,
        OSError,
    ) as exc:
        raise HTTPException(status_code=503, detail="result_artifact_remote_unavailable") from exc
    disposition = "inline" if record.get("content_type") == "video/mp4" else "attachment"
    return ResultArtifactFileResponse(
        path,
        artifact_cleanup=record.get("_artifact_cleanup"),
        media_type=str(record.get("content_type") or "application/octet-stream"),
        filename=path.name,
        content_disposition_type=disposition,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Blueprint-Artifact-SHA256": str(record["sha256"]),
        },
    )
