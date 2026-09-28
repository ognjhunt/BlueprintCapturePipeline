from pydantic import BaseModel, ConfigDict, Field

from tests.test_agent_execution_sessions import make_task, runtime
from blueprint_pipeline.agent_execution.contracts import AgentTool


class DefaultedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    uncertainty: list[str] = Field(default_factory=list)


def test_managed_provider_schema_is_strict_without_mutating_admitted_schema(tmp_path):
    task = make_task(output_schema=DefaultedOutput.model_json_schema())
    original = task.task_digest
    assert "uncertainty" not in task.output_schema["required"]
    executor, _ = runtime(tmp_path)
    payload = executor._create_payload(task)
    strict = payload["agent"]["text"]["format"]["schema"]
    assert set(strict["required"]) == set(strict["properties"])
    assert "uncertainty" not in task.output_schema["required"]
    assert task.snapshot().task_digest == original


def test_zero_argument_function_has_explicit_strict_required_list(tmp_path):
    tool = AgentTool("read_fixture", "1", "Read fixture", {"type": "object", "properties": {},
        "additionalProperties": False}, "read_only", lambda *_: {})
    task = make_task((tool,))
    executor, _ = runtime(tmp_path, tools=(tool,))
    payload = executor._create_payload(task)
    assert payload["agent"]["tools"][0]["parameters"]["required"] == []
    assert "required" not in tool.input_schema
