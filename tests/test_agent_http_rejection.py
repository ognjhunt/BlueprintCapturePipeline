from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from blueprint_pipeline.agent_execution.openai_transport import OpenAIAgentsHTTP, AgentTransportError


def test_preserves_protocol_error_metadata_without_provider_message_or_credentials():
    transport = OpenAIAgentsHTTP(api_key="private-key", project_id="project")
    class Opener:
        def open(self, *_args, **_kwargs):
            raise HTTPError("https://api.openai.com/v1/agents/sessions", 400, "invalid", {}, BytesIO(json.dumps({
                "error": {"type": "invalid_request_error", "code": "invalid_json_schema", "param": "agent.text.format",
                          "message": "schema invalid; private-key https://private.invalid/secret"}}).encode()))
    transport._opener = Opener()
    with pytest.raises(AgentTransportError) as caught:
        transport.request("POST", "/agents/sessions", body={})
    assert caught.value.diagnostics == {"http_status": 400, "type": "invalid_request_error", "code": "invalid_json_schema",
                                        "param": "agent.text.format", "category": "schema_validation"}
    assert "private" not in json.dumps(caught.value.diagnostics)
