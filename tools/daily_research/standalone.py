"""Export the reviewed research subtree; never install, schedule or call a provider."""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

FILES = (
    "README.md", "RENDER.md", "KNOWLEDGE.md", "SKILLS.md", "ADAPTIVE.md", "standalone.py", "runner.py", "knowledge.py",
    "contracts.py", "freshness.py", "capabilities.py", "discovery.py", "verification.py", "recovery.py", "qa_retry.py", "adaptive.py", "adaptive_runtime.py", "search.py", "allocation.py", "expansion.py", "exa_transport.py", "findall.py", "SEARCH.md", "site_universe.py",
    "history.py", "adaptive-test.config.example.json", "adaptive-daily.config.example.json", "perplexity-daily.config.example.json",
    "consumer.py", "publication.py", "publisher.mjs", "verification-digest.mjs", "requirements.txt", "standalone.config.example.json",
    "capabilities/blueprint-evidence-qualification/SKILL.md",
    "capabilities/blueprint-evidence-qualification/references/prospect-contract.md",
    "capabilities/deep-research/SKILL.md", "capabilities/deep-research/LICENSE",
    "firestore.py", "firestore_bridge.mjs", "contact_research.mjs", "render.py", "render_worker.mjs", "render.control.example.json",
    "config.example.json", "knowledge.config.example.json", "knowledge.v3.config.example.json",
    "daily-research.v2.schema.json", "daily-research.v3.schema.json",
    "knowledge-snapshot.v1.schema.json", "knowledge-refresh-policy.v1.schema.json",
    "systemd/blueprint-researcher-daily.service", "systemd/blueprint-researcher-daily.timer",
    "operators/research-oct2-control.py", "operators/research-perplexity-canary.py",
    "operators/research-perplexity-canary.mjs", "operators/paid-expansion-direction.py", "operators/README.md",
)
PREFIX = "tools/daily_research/"
# The canonical stdlib FindAll closure from src/blueprint_pipeline/. The WebApp installer
# admits only tools/daily_research/ members, so the release projects it under this
# prefix; findall.runtime() imports the canonical package first and this copy second.
BLUEPRINT_RUNTIME_FILES = (
    "__init__.py", "safe_outbound_http.py", "paid_resource_admission.py",
    "parallel_findall.py", "parallel_findall_execution.py", "parallel_findall_owner.py",
    "parallel_findall_admission.py",
)
RUNTIME_PREFIX = PREFIX + "pipeline_runtime/blueprint_pipeline/"


def git(repository, *args):
    return subprocess.run(["git", "--no-replace-objects", "-C", str(repository), *args], check=True,
                          capture_output=True).stdout


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def build(revision, destination, repository="."):
    """Require an exact commit and a new consumer-local directory; no worktree reads."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable commit SHA")
    if git(repository, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("Revision must identify a commit")
    payload = {}
    for name in FILES:
        path = PREFIX + name
        entry = git(repository, "ls-tree", revision, "--", path).decode().strip()
        if not entry.startswith("100644 blob ") and not entry.startswith("100755 blob "):
            raise ValueError("Missing regular source file: " + path)
        payload[path] = git(repository, "show", revision + ":" + path)
    # Project the canonical import closure into the portable research subtree.
    # This performs no installation, credential lookup or security change.
    for name in BLUEPRINT_RUNTIME_FILES:
        source = "src/blueprint_pipeline/" + name
        entry = git(repository, "ls-tree", revision, "--", source).decode().strip()
        if not entry.startswith("100644 blob ") and not entry.startswith("100755 blob "):
            raise ValueError("Missing regular source file: " + source)
        payload[RUNTIME_PREFIX + name] = git(repository, "show", revision + ":" + source)
    config = json.loads(payload[PREFIX + "standalone.config.example.json"])
    if config.get("enabled") is not False:
        raise ValueError("Standalone example must be disabled")
    manifest = {
        "schema_version": "blueprint.research-standalone.v1", "source_commit": revision,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())},
        "activation_performed": False, "example_enabled": False,
        "credential_environment_name": "OPENAI_API_KEY",
        "credential_binding_verified": False, "persistent_host_verified": False,
        "release_directory": "/opt/blueprint/researcher/releases/" + revision,
        "state_directory": "/var/lib/blueprint/researcher",
    }
    payload["manifest.json"] = encoded(manifest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(payload.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o644
            archive.addfile(entry, io.BytesIO(data))
    data = buffer.getvalue()
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt = {"source_commit": revision, "archive": "blueprint-research.tar",
               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    for name, content in ((receipt["archive"], data), ("receipt.json", encoded(receipt))):
        fd = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as output:
                output.write(content)
                output.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, help="New local directory under an existing parent")
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    print(json.dumps(build(args.revision, args.output, args.repository), sort_keys=True))


if __name__ == "__main__":
    main()
