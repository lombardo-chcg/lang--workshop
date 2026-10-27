import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.ids = {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
        self.result = agent.ProspectScore(
            score=75,
            justification="Fit assessment",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.scorer = Mock()
        self.scorer.invoke.return_value = self.result
        scorer_patch = patch.object(agent, "_scoring_llm", self.scorer)
        scorer_patch.start()
        self.addCleanup(scorer_patch.stop)

    def scoring_input(self):
        message = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = message.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        return json.loads(offering), json.loads(profile)

    def test_score_resolves_ids_and_serializes_result(self):
        self.assertEqual(agent.score_prospect.invoke(self.ids), self.result.model_dump())
        offering, profile = self.scoring_input()
        self.assertEqual(offering, agent.data_service.get_offering(self.ids["offering_id"]))
        self.assertEqual(profile, agent.build_prospect_profile.invoke(
            {"prospect_id": self.ids["prospect_id"]},
        )["prospect_profile"])

    def test_cached_profile_uses_authoritative_stack_without_changing_cache(self):
        cached = {"prospect_id": self.ids["prospect_id"], "tech_stack": ["stale"]}
        agent.data_service._PROFILES[self.ids["prospect_id"]] = cached
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Kafka"]):
            agent.score_prospect.invoke(self.ids)
        self.assertEqual(self.scoring_input()[1]["tech_stack"], ["Kafka"])
        self.assertEqual(cached["tech_stack"], ["stale"])

    def test_missing_and_invalid_offerings_do_not_score(self):
        valid = agent.data_service.get_offering(self.ids["offering_id"])
        invalid = [None, {}]
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            record = dict(valid)
            record.pop(field)
            invalid.append(record)
        for record in invalid:
            with self.subTest(record=record), patch.object(
                agent.data_service, "get_offering", return_value=record,
            ):
                self.assertEqual(agent.score_prospect.invoke(self.ids), {
                    "score": None, "error": "Cannot score without a valid offering.",
                })
        self.scorer.invoke.assert_not_called()

    def test_missing_prospect_does_not_score(self):
        self.assertEqual(agent.score_prospect.invoke({
            **self.ids, "prospect_id": "missing",
        }), {"score": None, "error": "Prospect not found."})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
        self.scorer.invoke.assert_not_called()

    def test_profile_enrichment_storage_and_cache_shared_with_scoring(self):
        pid = self.ids["prospect_id"]
        with patch.object(agent.data_service, "fetch_engagement_history", return_value=["engaged"]), \
             patch.object(agent.data_service, "fetch_account_details", return_value=["account"]):
            built = agent.build_prospect_profile.invoke({"prospect_id": pid})
        self.assertTrue(built["found"])
        self.assertEqual(built["prospect_profile"]["engagement_history"], ["engaged"])
        self.assertEqual(built["prospect_profile"]["account_details"], ["account"])
        self.assertEqual(agent.data_service.get_profile_from_db(pid)["prospect_profile"], built["prospect_profile"])
        with patch.object(agent.data_service, "get_prospect_record", side_effect=AssertionError("cache missed")):
            agent.score_prospect.invoke(self.ids)
            self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": pid}), built)
        self.assertEqual(self.scoring_input()[1], built["prospect_profile"])

    def run_script(self, calls, request):
        responses = [AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": str(i),
        }]) for i, (name, args) in enumerate(calls)]
        responses.append(AIMessage(content="Done"))
        graph = agent.create_deep_agent(
            model=ToolCallingModel(responses=responses),
            tools=[agent.get_current_rep, agent.score_prospect, agent.update_prospect_info,
                   agent.build_prospect_profile, agent.lookup_offering],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, "gtm_agent", graph):
            return agent.run_agent(request, user_id="rep_jchen")

    def test_agent_can_score_directly_with_ids_and_report_invalid_ids(self):
        for ids, expected in (
            (self.ids, self.result.model_dump()),
            ({**self.ids, "offering_id": "missing"}, {"score": None, "error": "Cannot score without a valid offering."}),
            ({**self.ids, "prospect_id": "missing"}, {"score": None, "error": "Prospect not found."}),
        ):
            with self.subTest(ids=ids):
                result = self.run_script(
                    [("get_current_rep", {}), ("score_prospect", ids)],
                    f"Score {ids['prospect_id']} against {ids['offering_id']}.",
                )
                messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual([m.name for m in messages], ["get_current_rep", "score_prospect"])
                self.assertEqual(json.loads(messages[-1].content), expected)

    def test_agent_updates_before_scoring(self):
        pid = self.ids["prospect_id"]
        records = deepcopy(agent.data_service.PROSPECTS)
        with patch.object(agent.data_service, "PROSPECTS", records):
            result = self.run_script([
                ("get_current_rep", {}),
                ("update_prospect_info", {"prospect_id": pid, "technology": "Kafka"}),
                ("score_prospect", self.ids),
            ], f"First add Kafka to {pid}'s tech stack, then score against OFFER-10005.")
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in messages], ["get_current_rep", "update_prospect_info", "score_prospect"])
        self.assertTrue(json.loads(messages[1].content)["updated"])
        self.assertEqual(json.loads(messages[2].content), self.result.model_dump())
        self.assertEqual(self.scoring_input()[1]["tech_stack"], records[pid]["tech_stack"])


if __name__ == "__main__":
    unittest.main()
