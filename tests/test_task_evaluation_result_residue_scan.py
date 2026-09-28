# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_result_residue_scan.py
"""The reference search reads what stays in a run for every file of the run it names."""

from __future__ import annotations

import json

from blueprint_pipeline import task_evaluation_result_residue_scan as scan


def test_the_stream_search_carries_a_token_and_reads_escaped_slashes() -> None:
    """A token cut by a chunk boundary is read whole, ``\\/`` reads as ``/``, and a token longer
    than any path keeps only its tail, so memory stays bounded."""

    import io

    def tokens(raw: bytes, chunk: int = 8) -> set[str]:
        found: set[str] = set()
        for part in scan.stream_tokens(io.BytesIO(raw), chunk_bytes=chunk):
            found |= part
        return found

    assert "logs/worker.log" in tokens(b'{"log": "logs/worker.log"}')
    assert "logs/worker.log" in tokens(b'{"log": "logs\\/worker.log"}', chunk=11)
    assert "logs/worker.log" in tokens(b"a logs/worker.log b", chunk=5)
    long = tokens(b"x" * (3 * scan._MAX_TOKEN_CHARS) + b"/run-1/logs/worker.log", chunk=4096)
    assert any(token.endswith("/run-1/logs/worker.log") for token in long)
    assert all(len(token) <= scan._MAX_TOKEN_CHARS for token in long)
    # A binary names nothing.
    assert tokens(b"\x00" + b"logs/worker.log") == set()


def test_every_way_a_kept_document_names_a_run_file_keeps_it() -> None:
    """A run file may be named absolutely (the run's own name may recur deeper in the path),
    relative to the evidence root, or relative to the run; each keeps it."""

    import io

    name = "run-7"
    value = {"nested": f"/var/lib/canaries/{name}/work/{name}/state.npz",
             "rooted": [f"{name}/logs/worker.log"], "relative": "provider/outputs.zip",
             "beside": "notes.txt", "above": "stage/state.npz"}

    def strings(raw: bytes) -> set[str]:
        return set().union(*scan.stream_tokens(io.BytesIO(raw)))

    named = set(scan.named_paths(strings(json.dumps(value).encode()), "work/stage/index.json", name))

    assert {f"work/{name}/state.npz", "logs/worker.log", "provider/outputs.zip", "work/stage/notes.txt",
            "work/stage/state.npz"} <= named
    # Free text and JSON lines name files too.
    assert "logs/worker.log" in strings(b"see logs/worker.log, then retry\n")
    assert f"/x/{name}/a.bin" in strings(b'{"a": 1}\n{"b": "/x/' + name.encode() + b'/a.bin"}\n')
