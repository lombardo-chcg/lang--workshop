import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class RunAgentTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_email_uses_injected_rep_without_identity_tool(self):
        for rep in agent.REP_IDS[:2]:
            with self.subTest(user_id=rep["rep_id"]):
                calls = []

                def generate(model, messages, **kwargs):
                    calls.append(messages)
                    identity = next(
                        message.content for message in messages
                        if isinstance(message, SystemMessage) and rep["email"] in message.content
                    )
                    self.assertIn(rep["name"], identity)
                    signer = identity.split("The signed-in rep is ", 1)[1].split(" (", 1)[0]
                    if len(calls) == 1:
                        response = AIMessage(content="", tool_calls=[{
                            "name": "get_prospect", "args": {"prospect_id": "LEAD-12853"}, "id": "lookup",
                        }])
                    elif len(calls) == 2:
                        prospect = json.loads(messages[-1].content)["prospect"]
                        response = AIMessage(content="", tool_calls=[{
                            "name": "send_prospect_email",
                            "args": {
                                "prospect": prospect, "subject": "Demo",
                                "body": f"Let's schedule a demo.\n\n{signer}",
                            },
                            "id": "send",
                        }])
                    else:
                        response = AIMessage(content="Email sent.")
                    return ChatResult(generations=[ChatGeneration(message=response)])

                with patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=generate):
                    result = agent.run_agent("Email LEAD-12853 to schedule a demo.", user_id=rep["rep_id"])
                tools = [message for message in result["messages"] if isinstance(message, ToolMessage)]
                self.assertEqual([message.name for message in tools], ["get_prospect", "send_prospect_email"])
                sent = json.loads(tools[-1].content)
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], rep["email"])
                self.assertEqual(sent["from_name"], rep["name"])
                self.assertTrue(sent["body"].endswith(rep["name"]))
                self.assertEqual(len(calls), 3)

    def test_explicit_identity_lookup_remains_available(self):
        for user_id in ("rep_amills", "unknown-rep"):
            with self.subTest(user_id=user_id):
                responses = iter([
                    AIMessage(content="", tool_calls=[{
                        "name": "get_current_rep", "args": {}, "id": "identity",
                    }]),
                    AIMessage(content="Identity checked."),
                ])
                with patch.object(ChatOpenAI, "_generate", side_effect=lambda *a, **k: ChatResult(
                    generations=[ChatGeneration(message=next(responses))]
                )):
                    result = agent.run_agent("Who am I signed in as? Use the identity tool.", user_id=user_id)
                tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
                rep = agent.data_service.get_rep(user_id)
                self.assertEqual(json.loads(tool_result.content), {"rep": rep, "found": rep is not None})

    def test_missing_rep_preserves_input_and_sender_behavior(self):
        captured = {}

        def invoke(inputs, config):
            captured.update(inputs)
            sent = agent.send_prospect_email.func(
                prospect={"name": "Prospect", "email": "prospect@example.com"},
                subject="Demo", body="Let's schedule a demo.",
                runtime=agent.ToolRuntime(
                    state={}, context=None, config=config, stream_writer=lambda _: None,
                    tool_call_id="send", store=None,
                ),
            )
            self.assertEqual(sent["status"], "sent")
            self.assertIsNone(sent["from"])
            self.assertIsNone(sent["from_name"])
            return {"messages": [AIMessage(content="Email sent.")]}

        with patch.object(agent.gtm_agent, "invoke", side_effect=invoke):
            agent.run_agent("Email the prospect.", user_id="unknown-rep")
        self.assertEqual(captured["messages"], [{"role": "user", "content": "Email the prospect."}])


if __name__ == "__main__":
    unittest.main()
