

def test_persist_survives_values_json_cannot_encode() -> None:
    """A receipt that cannot be written destroys the diagnosis of a paid run.

    `_persist` is called from a `finally`. Without `default=str` a value json
    cannot encode raises *inside* the handler, replacing the real exception and
    leaving the run with no receipt at all. The construction and controls
    workers already pass `default=str`; this one did not.
    """

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_policy_worker import _persist

    class _Unencodable:
        def __repr__(self) -> str:
            return "<warp array>"

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_policy_result.v1.json"
        _persist(target, {"status": "blocked", "stray": _Unencodable()})

        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["status"] == "blocked"
    assert written["stray"] == "<warp array>"
    assert written["result_digest"].startswith("sha256:")
