import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HAS_MODEL_CREDENTIALS = bool(os.environ.get("OPENAI_API_KEY"))

with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "LEAD-TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
        }

    def send_email(self, prospect=None):
        return agent.send_prospect_email.func(
            prospect=self.prospect if prospect is None else prospect,
            subject="Demo invitation",
            body="Would you like a demo?",
            runtime=self.runtime,
        )

    def assert_sent(self, result):
        rep = agent.data_service.get_rep("rep_amills")
        self.assertEqual(result["status"], "sent")
        self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["to_name"], self.prospect["name"])
        self.assertEqual(result["from"], rep["email"])
        self.assertEqual(result["from_name"], rep["name"])
        self.assertEqual(result["subject"], "Demo invitation")
        self.assertEqual(result["body"], "Would you like a demo?")

    def test_disqualified_source_blocks_email(self):
        record = {**self.prospect, "disqualified": True}
        before = copy.deepcopy(record)
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-TEST": record}, clear=True):
            for flag in (False, True, None):
                with self.subTest(contact_disqualified=flag):
                    result = self.send_email({**self.prospect, "disqualified": flag})
                    self.assertEqual(result, {
                        "status": "blocked",
                        "error": "Prospect is marked disqualified; email was not sent.",
                    })
        self.assertEqual(record, before)

    def test_eligible_source_sends_email(self):
        record = {**self.prospect, "disqualified": False}
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-TEST": record}, clear=True):
            self.assert_sent(self.send_email())

    def test_missing_source_preserves_send(self):
        with patch.dict(agent.data_service.PROSPECTS, {}, clear=True):
            for prospect_id in ("LEAD-UNKNOWN", None, ""):
                with self.subTest(prospect_id=prospect_id):
                    self.assert_sent(self.send_email({**self.prospect, "prospect_id": prospect_id}))
            prospect = {k: v for k, v in self.prospect.items() if k != "prospect_id"}
            self.assert_sent(self.send_email(prospect))

    def test_missing_email_still_fails(self):
        result = self.send_email({"prospect_id": "LEAD-TEST"})
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(HAS_MODEL_CREDENTIALS, "Requires OpenAI model credentials")
    def test_agent_explains_blocked_send(self):
        result = agent.run_agent(
            "Send LEAD-50001 an email inviting them to book a demo.",
            user_id="rep_amills",
        )
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not sent|did not send|didn't send|cannot send|can't send|unable to send|could not send|couldn't send")
        for message in result["messages"]:
            if getattr(message, "name", None) == "send_prospect_email":
                self.assertEqual(json.loads(message.content)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
