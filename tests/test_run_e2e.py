from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.run_e2e import main


def test_run_e2e_cli_accepts_local_no_llm_provider(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run_end_to_end(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "preflight_status": "passed",
            "pipeline_status": "completed",
            "pipeline_lanes": ["qualification"],
        }

    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_end_to_end", fake_run_end_to_end)

    assert main(["--capture-root", str(tmp_path), "--provider", "local"]) == 0
    assert seen == {"capture_root": str(tmp_path), "provider": "local", "resume_completed_stages": False}
