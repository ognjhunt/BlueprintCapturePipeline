"""Pin and stage the PaliGemma tokenizer required by both G1 pi0.5 policies.

The HumanoidArena checkpoints contain publisher-local tokenizer paths, but no
tokenizer files. This module checks gated access before paid launch and binds
the exact small tokenizer-only files to a fixed upstream revision. It never
records or forwards the Hugging Face credential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://huggingface.co/google/paligemma-3b-pt-224"
SOURCE_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
EXPECTED_FILES = frozenset(
    {
        "added_tokens.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
)
COMPAT_TOKENIZER_PATH = Path("/ai/Yichi/taowen/ckpts/checkpoints/paligemma-3b-pt-224")
PUBLISHER_TOKENIZER_REFS = {
    "humanoidarena_pi05_g1_dex3_sonic": str(COMPAT_TOKENIZER_PATH),
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi": (
        "/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints/paligemma-3b-pt-224"
    ),
}


class _HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme.lower() != "https":
            raise ValueError("g1_pi_tokenizer_insecure_redirect")
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if (
            redirected is not None
            and urllib.parse.urlsplit(new_url).netloc
            != urllib.parse.urlsplit(request.full_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def _request(url: str, *, token: str, method: str):
    if urllib.parse.urlsplit(url).scheme.lower() != "https":
        raise ValueError("g1_pi_tokenizer_insecure_source")
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token}, method=method
    )
    return urllib.request.build_opener(_HTTPSOnly()).open(request, timeout=60)


def _token(token_file: Path) -> str:
    path = Path(token_file)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 4096
        or path.stat().st_mode & 0o007
    ):
        raise ValueError("g1_pi_tokenizer_credential_unavailable")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\x00" in value:
        raise ValueError("g1_pi_tokenizer_credential_unavailable")
    return value


def _inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_pi_tokenizer_inventory_unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("files") if isinstance(value, dict) else None
    if not isinstance(value, dict) or (
        value.get("schema_version") != "g1_paligemma_tokenizer_inventory.v1"
        or value.get("source_repository") != SOURCE_REPOSITORY
        or value.get("source_revision") != SOURCE_REVISION
        or value.get("model_license") != "gemma"
        or value.get("rights_review_required") is not True
        or not isinstance(rows, list)
        or len(rows) != len(EXPECTED_FILES)
        or {row.get("path") for row in rows if isinstance(row, dict)} != EXPECTED_FILES
    ):
        raise ValueError("g1_pi_tokenizer_inventory_invalid")
    for row in rows:
        if (
            not isinstance(row, dict)
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] <= 0
        ):
            raise ValueError("g1_pi_tokenizer_inventory_invalid")
        digest = row.get("sha256") or row.get("git_blob_sha1")
        expected_length = 64 if "sha256" in row else 40
        if (
            ("sha256" in row) == ("git_blob_sha1" in row)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{" + str(expected_length) + r"}", digest) is None
        ):
            raise ValueError("g1_pi_tokenizer_inventory_invalid")
    return value


def _url(inventory: dict[str, Any], name: str) -> str:
    return (
        SOURCE_REPOSITORY
        + "/resolve/"
        + inventory["source_revision"]
        + "/"
        + urllib.parse.quote(name, safe="")
    )


def _identity(path: Path, row: dict[str, Any]) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_pi_tokenizer_file_missing_or_symlink:" + row["path"])
    # SHA-1 is the upstream Git blob identifier, not a security digest.
    digest = hashlib.sha256() if "sha256" in row else hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    if size != row["size_bytes"]:
        raise ValueError("g1_pi_tokenizer_file_size_mismatch:" + row["path"])
    if "git_blob_sha1" in row:
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    expected = row.get("sha256") or row.get("git_blob_sha1")
    if digest.hexdigest() != expected:
        raise ValueError("g1_pi_tokenizer_file_digest_mismatch:" + row["path"])
    return digest.hexdigest(), size


def verify_tokenizer_assets(*, inventory_path: Path, asset_dir: Path) -> dict[str, Any]:
    inventory = _inventory(inventory_path)
    root = Path(asset_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_pi_tokenizer_asset_directory_invalid")
    files = []
    for row in inventory["files"]:
        path = root / row["path"]
        digest, size = _identity(path, row)
        files.append({"path": row["path"], "digest": digest, "size_bytes": size})
    return {
        "status": "tokenizer_bytes_verified",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "inventory_sha256": "sha256:"
        + hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest(),
        "files": files,
        "inference_executed": False,
    }


def preflight_tokenizer_access(*, inventory_path: Path, token_file: Path) -> dict[str, Any]:
    """Read-only all-file HEAD check; no GPU and no credential in the receipt."""

    inventory = _inventory(inventory_path)
    token = _token(token_file)
    checked = []
    for row in inventory["files"]:
        try:
            with _request(_url(inventory, row["path"]), token=token, method="HEAD") as response:
                if (
                    response.status != 200
                    or urllib.parse.urlsplit(response.geturl()).scheme.lower() != "https"
                ):
                    raise ValueError("g1_pi_tokenizer_access_unverified:" + row["path"])
        except urllib.error.HTTPError as exc:
            raise ValueError("g1_pi_tokenizer_access_http_" + str(exc.code)) from None
        checked.append(row["path"])
    return {
        "status": "tokenizer_access_verified_no_download",
        "source_revision": SOURCE_REVISION,
        "files_checked": checked,
        "credential_recorded": False,
        "gpu_allocated": False,
    }


def download_tokenizer_assets(
    *, inventory_path: Path, token_file: Path, asset_dir: Path
) -> dict[str, Any]:
    """Acquire only the pinned tokenizer files outside a paid GPU run."""

    inventory = _inventory(inventory_path)
    token = _token(token_file)
    root = Path(asset_dir)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("g1_pi_tokenizer_asset_directory_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for row in inventory["files"]:
        destination = root / row["path"]
        if destination.exists() or destination.is_symlink():
            _identity(destination, row)
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".g1-pi-tokenizer-", dir=root, delete=False
            ) as stream:
                temporary = Path(stream.name)
                try:
                    with _request(
                        _url(inventory, row["path"]), token=token, method="GET"
                    ) as response:
                        if (
                            response.status != 200
                            or urllib.parse.urlsplit(response.geturl()).scheme.lower() != "https"
                        ):
                            raise ValueError("g1_pi_tokenizer_download_unverified:" + row["path"])
                        size = 0
                        while block := response.read(1024 * 1024):
                            size += len(block)
                            if size > row["size_bytes"]:
                                raise ValueError("g1_pi_tokenizer_download_exceeds_pinned_size")
                            stream.write(block)
                except urllib.error.HTTPError as exc:
                    raise ValueError("g1_pi_tokenizer_access_http_" + str(exc.code)) from None
            _identity(temporary, row)
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify_tokenizer_assets(inventory_path=inventory_path, asset_dir=root)


def stage_provider_tokenizer(
    *,
    inventory_path: Path,
    bundled_dir: Path,
    destination: Path = COMPAT_TOKENIZER_PATH,
) -> dict[str, Any]:
    """Copy verified private bundle bytes to the pinned publisher compat path."""

    receipt = verify_tokenizer_assets(inventory_path=inventory_path, asset_dir=bundled_dir)
    target = Path(destination)
    if not target.is_absolute() or any(
        parent.is_symlink() for parent in (target, *target.parents) if parent != Path("/")
    ):
        raise ValueError("g1_pi_tokenizer_compat_path_invalid")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    inventory = _inventory(inventory_path)
    for row in inventory["files"]:
        path = target / row["path"]
        if path.exists() or path.is_symlink():
            _identity(path, row)
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".g1-pi-tokenizer-", dir=target, delete=False
            ) as stream:
                temporary = Path(stream.name)
                with (Path(bundled_dir) / row["path"]).open("rb") as source:
                    shutil.copyfileobj(source, stream)
            _identity(temporary, row)
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    verify_tokenizer_assets(inventory_path=inventory_path, asset_dir=target)
    return {**receipt, "status": "tokenizer_compat_path_ready", "compat_path": str(target)}


def require_policy_tokenizer_reference(policy_dir: Path, candidate_id: str) -> None:
    """Reject a new publisher-local path before starting a paid policy server."""

    expected = PUBLISHER_TOKENIZER_REFS.get(candidate_id)
    if expected is None:
        return
    path = Path(policy_dir) / "policy_preprocessor.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_pi_tokenizer_preprocessor_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    refs = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "tokenizer_name":
                    refs.append(item)
                collect(item)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(value)
    if refs != [expected]:
        raise ValueError("g1_pi_tokenizer_publisher_reference_changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2] / "configs/g1_paligemma_tokenizer_inventory.v1.json"
        ),
    )
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--access-check", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.access_check:
        if args.token_file is None:
            parser.error("--access-check requires --token-file")
        result = preflight_tokenizer_access(
            inventory_path=args.inventory,
            token_file=args.token_file,
        )
    elif args.verify_only:
        if args.asset_dir is None:
            parser.error("--verify-only requires --asset-dir")
        result = verify_tokenizer_assets(
            inventory_path=args.inventory,
            asset_dir=args.asset_dir,
        )
    else:
        if args.token_file is None or args.asset_dir is None:
            parser.error("download requires --token-file and --asset-dir")
        result = download_tokenizer_assets(
            inventory_path=args.inventory,
            token_file=args.token_file,
            asset_dir=args.asset_dir,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
