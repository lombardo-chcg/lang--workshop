import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class RunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.roots = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.roots.append((run_id, kwargs))


class AgentTests(unittest.TestCase):
    def test_application_tools_and_first_call_budget(self):
        calls = []

        def generate(model, messages, **kwargs):
            calls.append((messages, kwargs["tools"]))
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Hello"))])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(ChatOpenAI, "_generate", generate):
            agent.run_agent("Hello", user_id="rep_amills")

        messages, tools = calls[0]
        self.assertEqual({tool["function"]["name"] for tool in tools}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info", "get_current_rep",
        })
        for tool in tools:
            self.assertNotIn("runtime", tool["function"]["parameters"].get("properties", {}))
        payload = json.dumps({"messages": messages_to_dict(messages), "tools": tools})
        self.assertLess(len(payload), 7000)
        self.assertLess(sum(len(message.content) for message in messages if message.type == "system"), 1200)

    def test_runtime_tools_and_tracing_configuration(self):
        responses = iter([
            AIMessage(content="", tool_calls=[{"name": "get_current_rep", "args": {}, "id": "rep"}]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email", "id": "email", "args": {
                    "prospect": {"name": "Prospect", "email": "prospect@example.com"},
                    "subject": "Demo", "body": "Let's meet.",
                },
            }]),
            AIMessage(content="Sent"),
        ])
        recorder = RunRecorder()

        def generate(model, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        original_invoke = agent.gtm_agent.invoke

        def invoke(inputs, config):
            return original_invoke(inputs, config={**config, "callbacks": [recorder]})

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(ChatOpenAI, "_generate", generate), patch.object(agent.gtm_agent, "invoke", side_effect=invoke):
            result = agent.run_agent("Email the prospect", user_id="rep_amills", environment="test", thread_id="thread-1")

        outputs = [json.loads(message.content) for message in result["messages"] if isinstance(message, ToolMessage)]
        rep = agent.data_service.get_rep("rep_amills")
        self.assertEqual(outputs[0], {"rep": rep, "found": True})
        self.assertEqual(outputs[1]["status"], "sent")
        self.assertEqual(outputs[1]["from"], rep["email"])
        self.assertEqual(outputs[1]["from_name"], rep["name"])
        self.assertEqual(outputs[1]["to"], "prospect@example.com")
        self.assertEqual(outputs[1]["subject"], "Demo")
        self.assertEqual(outputs[1]["body"], "Let's meet.")
        self.assertEqual(result["reply"], "Sent")
        self.assertEqual(len(recorder.roots), 1)
        run_id, recorded = recorder.roots[0]
        self.assertEqual(run_id, uuid.UUID(result["run_id"]))
        self.assertEqual(recorded["name"], "GTM Assistant")
        self.assertEqual({key: recorded["metadata"][key] for key in (
            "thread_id", "user_id", "environment", "request_intent",
        )}, {
            "thread_id": "thread-1", "user_id": "rep_amills",
            "environment": "test", "request_intent": "send_email",
        })


if __name__ == "__main__":
    unittest.main()
