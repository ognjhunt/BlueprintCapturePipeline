"""Pinned official OSCAR release metadata used by learned-WAM runtime paths."""

from __future__ import annotations

from typing import Any, Mapping


OFFICIAL_OSCAR_PROJECT_PAGE_URL = "https://wuzy2115.github.io/oscar-project-page/"
OFFICIAL_OSCAR_SOURCE_URL = "https://github.com/wuzy2115/oscar-public.git"
OFFICIAL_OSCAR_SOURCE_WEB_URL = "https://github.com/wuzy2115/oscar-public"
OFFICIAL_OSCAR_SOURCE_COMMIT = "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
OFFICIAL_OSCAR_HF_REPO = "zywu2115/OSCAR-2B"
OFFICIAL_OSCAR_HF_REVISION = "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6"
OFFICIAL_OSCAR_MODEL_URL = "https://huggingface.co/zywu2115/OSCAR-2B"
OFFICIAL_OSCAR_POLICY_ROLLOUT_DATASET_URL = (
    "https://huggingface.co/datasets/zywu2115/OSCAR_policy_rollout"
)
OFFICIAL_OSCAR_MODEL_NAME = "OSCAR-2B"

OFFICIAL_OSCAR_WAM_IMAGE_TAG_REF = (
    "docker.io/nijelhunt/blueprint-oscar-wam:20260701-cu128-ropefix"
)
OFFICIAL_OSCAR_WAM_IMAGE_DIGEST = (
    "sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3"
)
OFFICIAL_OSCAR_WAM_IMAGE_AMD64_DIGEST = (
    "sha256:dc23334693d2983122f628ffaec9ea481bfdb8f0bfcec9d22efd83baba827b60"
)
OFFICIAL_OSCAR_WAM_IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-oscar-wam@"
    + OFFICIAL_OSCAR_WAM_IMAGE_DIGEST
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalized_source_url(value: str) -> str:
    text = _string(value).rstrip("/")
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text[len("git@github.com:") :]
    elif text.startswith("ssh://git@github.com/"):
        text = "https://github.com/" + text[len("ssh://git@github.com/") :]
    if text.endswith(".git"):
        text = text[:-4]
    return text.lower()


def source_url_is_official(value: str) -> bool:
    return _normalized_source_url(value) == _normalized_source_url(OFFICIAL_OSCAR_SOURCE_URL)


def source_ref_is_official(value: str) -> bool:
    return _string(value) == OFFICIAL_OSCAR_SOURCE_COMMIT


def hf_repo_is_official(value: str) -> bool:
    return _string(value) == OFFICIAL_OSCAR_HF_REPO


def hf_revision_is_official(value: str) -> bool:
    return _string(value) == OFFICIAL_OSCAR_HF_REVISION


def image_ref_digest(value: str) -> str | None:
    text = _string(value)
    marker = "@sha256:"
    if marker not in text:
        return None
    digest = "sha256:" + text.split(marker, maxsplit=1)[1].split("/", maxsplit=1)[0]
    return digest if len(digest) == len("sha256:") + 64 else None


def image_ref_is_digest_pinned(value: str) -> bool:
    return image_ref_digest(value) is not None


def image_ref_is_official(value: str) -> bool:
    digest = image_ref_digest(value)
    if digest:
        return digest in {
            OFFICIAL_OSCAR_WAM_IMAGE_DIGEST,
            OFFICIAL_OSCAR_WAM_IMAGE_AMD64_DIGEST,
        }
    return _string(value) == OFFICIAL_OSCAR_WAM_IMAGE_TAG_REF


def official_release_contract(
    *,
    source_url: str = OFFICIAL_OSCAR_SOURCE_URL,
    source_ref: str = OFFICIAL_OSCAR_SOURCE_COMMIT,
    hf_repo: str = OFFICIAL_OSCAR_HF_REPO,
    hf_revision: str = OFFICIAL_OSCAR_HF_REVISION,
    image_ref: str | None = None,
) -> dict[str, Any]:
    image_digest = image_ref_digest(image_ref or "")
    source_match = source_url_is_official(source_url) and source_ref_is_official(source_ref)
    checkpoint_match = hf_repo_is_official(hf_repo) and hf_revision_is_official(hf_revision)
    image_match = True if image_ref is None else image_ref_is_official(image_ref)
    return {
        "schema_version": "official_oscar_release_contract.v1",
        "model_candidate": "oscar_wam",
        "model_name": OFFICIAL_OSCAR_MODEL_NAME,
        "project_page_url": OFFICIAL_OSCAR_PROJECT_PAGE_URL,
        "source_url": source_url,
        "source_web_url": OFFICIAL_OSCAR_SOURCE_WEB_URL,
        "source_ref": source_ref,
        "expected_source_url": OFFICIAL_OSCAR_SOURCE_URL,
        "expected_source_commit": OFFICIAL_OSCAR_SOURCE_COMMIT,
        "source_url_official": source_url_is_official(source_url),
        "source_ref_pinned_to_reviewed_commit": source_ref_is_official(source_ref),
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "expected_hf_repo": OFFICIAL_OSCAR_HF_REPO,
        "expected_hf_revision": OFFICIAL_OSCAR_HF_REVISION,
        "hf_repo_official": hf_repo_is_official(hf_repo),
        "hf_revision_pinned": hf_revision_is_official(hf_revision),
        "image_ref": image_ref,
        "image_ref_digest": image_digest,
        "expected_image_tag_ref": OFFICIAL_OSCAR_WAM_IMAGE_TAG_REF,
        "expected_image_digest": OFFICIAL_OSCAR_WAM_IMAGE_DIGEST,
        "expected_image_amd64_digest": OFFICIAL_OSCAR_WAM_IMAGE_AMD64_DIGEST,
        "image_ref_digest_pinned": bool(image_digest),
        "image_ref_official": image_match,
        "official_release_match": bool(source_match and checkpoint_match and image_match),
        "policy_rollout_dataset_url": OFFICIAL_OSCAR_POLICY_ROLLOUT_DATASET_URL,
        "policy_rollout_dataset_is_reference_data_not_runtime": True,
    }


def official_release_blockers(
    contract: Mapping[str, Any],
    *,
    require_image_digest: bool = False,
) -> list[str]:
    blockers: list[str] = []
    if contract.get("source_url_official") is not True:
        blockers.append("official_oscar_source_url_mismatch")
    if contract.get("source_ref_pinned_to_reviewed_commit") is not True:
        blockers.append("official_oscar_source_commit_not_pinned")
    if contract.get("hf_repo_official") is not True:
        blockers.append("official_oscar_hf_repo_mismatch")
    if contract.get("hf_revision_pinned") is not True:
        blockers.append("official_oscar_hf_revision_not_pinned")
    if contract.get("image_ref") and contract.get("image_ref_official") is not True:
        blockers.append("official_oscar_provider_image_mismatch")
    if require_image_digest and contract.get("image_ref_digest_pinned") is not True:
        blockers.append("official_oscar_provider_image_digest_missing")
    return blockers
