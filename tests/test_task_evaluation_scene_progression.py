"""Persistent intent reaches real factory output without allocating a provider."""
import json
import time
from pathlib import Path

from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_progression_state as state
from blueprint_pipeline import public_scene_host_input_intake
from tests.test_task_evaluation_public_scene_attempt_factory import context as context, write


def configuration(context, monkeypatch):
    args, _ = context
    root = args["intent_path"].parent.parent
    release = json.loads(args["release_binding_path"].read_text())
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: release["source_commit"])
    bindings = root.parent / "bindings"
    binding = json.loads(args["source_binding_path"].read_text())
    write(bindings / (binding["binding_id"] + ".json"), binding)
    config = dict(schema_version=engine.CONFIG_SCHEMA, intent_root=str(root),
        public_source_binding_root=str(bindings), machinery_path=str(args["machinery_path"]),
        release_binding_path=str(args["release_binding_path"]),
        factory_output_root=str(root.parent / "progression-output"), trusted_clients=["blueprint-webapp"],
        submission_enabled=False)
    return write(root.parent / "progression-config.json", config, "config_digest")


def test_real_factory_is_idempotent_across_worker_restart(context, monkeypatch):
    config = configuration(context, monkeypatch)
    first = engine.process_scene_intents(config_path=config)
    assert first["results"][0]["phase"] == "publication_ready", first
    assert first["provider_allocation_performed"] is False
    second = engine.process_scene_intents(config_path=config)
    assert second == first
    directory = context[0]["intent_path"].parent
    intent = json.loads((directory / "intent.json").read_text())
    progress = state.load_progression(directory, intent)
    factory = json.loads(Path(progress["state"]["factory"]["path"]).read_text())
    assert factory["original_source_reinstalled"] is False
    assert factory["provider_mutation_performed"] is False


def test_missing_source_and_expired_owner_do_not_reserve(context, monkeypatch):
    config = configuration(context, monkeypatch)
    value = json.loads(config.read_text())
    value["public_source_binding_root"] = str(config.parent / "missing")
    write(config, value, "config_digest")
    before = list(context[0]["intent_path"].parent.joinpath("attempts").glob("*.json"))
    result = engine.process_scene_intents(config_path=config)
    assert result["results"][0]["status"] == "awaiting_source"
    result = engine.process_scene_intents(config_path=config, now=time.time() + 10000)
    assert result["results"][0]["phase"] == "authority"
    assert before == list(context[0]["intent_path"].parent.joinpath("attempts").glob("*.json"))


def test_projection_recovers_event_before_pointer_crash(context, monkeypatch):
    config = configuration(context, monkeypatch)
    engine.process_scene_intents(config_path=config)
    directory = context[0]["intent_path"].parent
    intent = json.loads((directory / "intent.json").read_text())
    expected = state.load_progression(directory, intent)
    (directory / "progression.json").unlink()
    assert state.load_progression(directory, intent) == expected
