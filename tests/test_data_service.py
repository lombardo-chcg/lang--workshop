import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_persists_stack_and_preserves_result(self):
        original = list(data_service.fetch_tech_stack("LEAD-71001"))
        self.assertNotIn("Kafka", original)

        result = data_service.update_prospect_info("LEAD-71001", "Kafka")

        self.assertEqual(result, {
            "updated": True, "found": True, "tech_stack": original + ["Kafka"],
        })
        self.assertIn("Kafka", data_service.fetch_tech_stack("LEAD-71001"))
        self.assertEqual(data_service.update_prospect_info("LEAD-71001", "Kafka"), result)
        self.assertEqual(data_service.fetch_tech_stack("LEAD-71001").count("Kafka"), 1)

    def test_update_refreshes_cached_profile(self):
        cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        self.assertNotIn("Kafka", cached["prospect_profile"]["tech_stack"])

        gtm_agent.update_prospect_info.invoke({"prospect_id": "LEAD-71001", "technology": "Kafka"})
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})

        self.assertTrue(rebuilt["found"])
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])

    def test_scoring_uses_updated_saved_stack_even_with_old_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})["prospect_profile"]
        offering = data_service.get_offering("OFFER-10002")
        offering = {**offering, "required_tech_stack": offering["required_tech_stack"] + ["Kafka"]}
        data_service.update_prospect_info("LEAD-71001", "Kafka")

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        messages = scoring_llm.invoke.call_args.args[0]
        scoring_profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
        missing = set(offering["required_tech_stack"]) - set(scoring_profile["tech_stack"])
        self.assertEqual(missing, set())
        self.assertNotIn("Kafka", profile["tech_stack"])

    def test_unknown_prospect_preserves_failure_result(self):
        self.assertEqual(data_service.update_prospect_info("unknown", "Kafka"), {
            "updated": False, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
