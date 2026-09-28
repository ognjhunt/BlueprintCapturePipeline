"""Check one operator-scoped credential without inference or spending authority."""
import json
from pathlib import Path
from urllib.error import HTTPError

from . import safe_outbound_http

URL = "https://api.openai.com/v1/models"
MAX_RESPONSE_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 10


class OpenAICredentialPreflightError(RuntimeError):
    """Only fixed codes are exposed; provider bodies and key fragments are not."""


def check_openai_credential(*, api_key_file, project_id):
    path = Path(api_key_file)
    try:
        if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file() or path.stat().st_mode & 0o027 or path.stat().st_size > 16384):
            raise OpenAICredentialPreflightError("openai_credential_preflight_key_file_invalid")
        key = path.read_text().strip()
    except (OSError, UnicodeError):
        raise OpenAICredentialPreflightError("openai_credential_preflight_key_file_unreadable") from None
    if not key or any(c.isspace() for c in key):
        raise OpenAICredentialPreflightError("openai_credential_preflight_key_invalid")
    if not isinstance(project_id, str) or not project_id or any(c.isspace() for c in project_id):
        raise OpenAICredentialPreflightError("openai_credential_preflight_project_invalid")
    try:
        response = safe_outbound_http.request(URL, method="GET",
            headers={"Authorization": "Bearer " + key, "OpenAI-Project": project_id},
            timeout_seconds=TIMEOUT_SECONDS, max_response_bytes=MAX_RESPONSE_BYTES,
            policy=safe_outbound_http.pinned_api_policy(URL, max_response_bytes=MAX_RESPONSE_BYTES))
    except HTTPError as exc:
        status = exc.code if type(exc.code) is int and 100 <= exc.code <= 599 else 0
        raise OpenAICredentialPreflightError(f"openai_credential_preflight_http_{status}") from None
    except (OSError, ValueError, RuntimeError):
        raise OpenAICredentialPreflightError("openai_credential_preflight_transport_failed") from None
    if response.status != 200:
        raise OpenAICredentialPreflightError("openai_credential_preflight_not_authenticated")
    try:
        payload = json.loads(response.body)
        if not isinstance(payload, dict) or payload.get("object") != "list" or not isinstance(payload.get("data"), list):
            raise ValueError
    except (TypeError, ValueError):
        raise OpenAICredentialPreflightError("openai_credential_preflight_response_invalid") from None
    return {"schema_version": "openai_credential_preflight.v1", "status": "authenticated",
        "provider": "openai", "operation": "models_list", "http_status": 200,
        "attempt_count": 1, "request_timeout_seconds": TIMEOUT_SECONDS,
        "model_inference_performed": False, "provider_mutation_performed": False,
        "spending_authority_granted": False, "raw_secret_values_recorded": False}
