import copy
import json
import unittest
from unittest.mock import patch

from langsmith import tracing_context

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "LANGSMITH_TRACING": "false"}):
    with patch("dotenv.load_dotenv"):
        from gtm_agent import data_service
        from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-12853"
        self.record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.cache_patch = patch.object(data_service, "_PROFILES", {})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def test_contact_allowlist(self):
        for billing in (True, False):
            with self.subTest(billing=billing):
                record = {**self.record, "future_private_field": "private"}
                if not billing:
                    record.pop("billing_qualification")
                with patch.object(data_service, "get_prospect_record", return_value=record):
                    result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                self.assertEqual(result, {
                    "prospect": {
                        "prospect_id": self.prospect_id,
                        **{k: record[k] for k in ("name", "email", "disqualified")},
                    },
                    "found": True,
                })

    def test_new_profile_preserves_nonbilling_fields_and_cache(self):
        for billing in (True, False):
            with self.subTest(billing=billing):
                data_service._PROFILES.clear()
                record = copy.deepcopy(self.record)
                if not billing:
                    record.pop("billing_qualification")
                expected = {"prospect_id": self.prospect_id, **record}
                expected.pop("billing_qualification", None)
                with patch.object(data_service, "get_prospect_record", return_value=record):
                    result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                self.assertEqual(result, {"prospect_profile": expected, "found": True})
                self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
                self.assertEqual(
                    agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}), result
                )
                self.assertEqual("billing_qualification" in record, billing)

    def test_cached_profile_is_sanitized_before_trace_returns(self):
        cached = {"prospect_id": self.prospect_id, **self.record, "custom_field": "keep"}
        data_service._PROFILES[self.prospect_id] = cached
        expected = {k: v for k, v in cached.items() if k != "billing_qualification"}
        outputs = []
        with tracing_context(enabled="local"):
            result = data_service.get_profile_from_db(
                self.prospect_id,
                langsmith_extra={"on_end": lambda run: outputs.append(copy.deepcopy(run.outputs))},
            )
        self.assertEqual(outputs, [{"prospect_profile": expected}])
        self.assertEqual(result, {"prospect_profile": expected})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
            {"prospect_profile": expected, "found": True},
        )

    def test_missing_prospect_shapes(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})

    def test_scoring_payload_allowlist_and_authoritative_tech_stack(self):
        score = agent.ProspectScore(
            score=75, justification="Fit assessment",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        offering = data_service.get_offering("OFFER-10001")
        for billing in (True, False):
            for has_id in (True, False):
                with self.subTest(billing=billing, has_id=has_id):
                    profile = {**self.record, "tech_stack": ["caller-supplied"]}
                    if not billing:
                        profile.pop("billing_qualification")
                    if has_id:
                        profile["prospect_id"] = self.prospect_id
                    original = copy.deepcopy(profile)
                    with patch.object(agent, "_scoring_llm") as llm:
                        llm.invoke.return_value = score
                        result = agent.score_prospect.invoke({
                            "prospect_profile": profile, "offering": offering,
                        })
                    self.assertEqual(result, score.model_dump())
                    content = llm.invoke.call_args.args[0][1]["content"]
                    offering_json, profile_json = content.removeprefix("Offering:\n").split(
                        "\n\nProspect profile:\n"
                    )
                    self.assertEqual(json.loads(offering_json), offering)
                    expected = {k: profile[k] for k in (
                        "annual_revenue", "tech_stack", "account_details", "engagement_history"
                    ) if k in profile}
                    if has_id:
                        expected["tech_stack"] = self.record["tech_stack"]
                    self.assertEqual(json.loads(profile_json), expected)
                    self.assertEqual(profile, original)

    def test_invalid_offering_preserves_error(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            with self.subTest(offering=offering):
                with patch.object(agent, "_scoring_llm") as llm:
                    self.assertEqual(agent.score_prospect.invoke({
                        "prospect_profile": self.record, "offering": offering,
                    }), {"score": None, "error": "Cannot score without a valid offering."})
                    llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
