from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APITestCase

from app.views_factcheck_api import (
    _format_sources_paragraph,
    _rank_and_cap_sources,
    _split_title_url,
)


CLAIM_TEXT = "Did the FG approve a ₦70,000 minimum wage in 2024?"


# ============================================================
# 🌐 Web view (HTML) – user submits a statement/claim
# ============================================================

class FactcheckWebViewClaimTests(TestCase):
    """
    Tests the HTML factcheck view with a real user claim.
    """

    @patch("app.views.get_external_api_answer")
    def test_web_factcheck_with_claim(self, mock_get_answer):
        """
        When the user posts a normal claim:
        - we call get_external_api_answer with that claim
        - we render preview.html with the verdict/explanation
        - we include at least one source URL
        """
        mock_get_answer.return_value = (
            "Verdict: False\n\nExplanation: Test summary for minimum wage claim.",
            ["https://example.com/news/fg-minimum-wage"],
        )

        url = reverse("result")  # ensure urls.py has: path("result/", views.result, name="result")
        response = self.client.post(url, {"query": CLAIM_TEXT})

        self.assertEqual(response.status_code, 200)

        # The pipeline should have been called with the real statement
        mock_get_answer.assert_called_once_with(CLAIM_TEXT)

        # Check that the verdict and explanation show up in the HTML
        self.assertContains(response, "Verdict: False")
        self.assertContains(response, "Test summary for minimum wage claim.")

        # Check that the source appears somewhere in the rendered page
        self.assertContains(response, "https://example.com/news/fg-minimum-wage")


# ============================================================
# 🧪 API – same claim, human mode & JSON mode
# ============================================================

class FactcheckAPIClaimTests(APITestCase):
    """
    Tests the API endpoint with a real user claim.
    """

    @patch("app.views_factcheck_api.get_external_api_answer_english")
    def test_api_human_mode_with_claim(self, mock_core_answer):
        """
        Human mode (default) with a real statement/claim.
        """
        mock_core_answer.return_value = {
            "result": (
                "Verdict: Misleading\n\nExplanation: Test explanation for FG minimum wage claim.\n\n"
                "Sources:\n- Premium Times — https://premiumtimesng.com/test-minimum-wage"
            )
        }

        url = reverse("api-factcheck")  # ensure urls.py has: path("api/factcheck/", FactcheckAPIView.as_view(), name="api-factcheck")
        payload = {"query": CLAIM_TEXT}

        response = self.client.post(url, payload, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        # Basic shape
        self.assertIn("query", data)
        self.assertIn("result", data)
        self.assertIn("latency_ms", data)

        self.assertEqual(data["query"], CLAIM_TEXT)
        self.assertIn("Verdict: Misleading", data["result"])
        self.assertIn("Test explanation for FG minimum wage claim.", data["result"])
        self.assertIn("https://premiumtimesng.com/test-minimum-wage", data["result"])

        mock_core_answer.assert_called_once_with(CLAIM_TEXT, json_mode=False)

    @patch("app.views_factcheck_api.get_external_api_answer_english")
    def test_api_json_mode_with_claim(self, mock_core_answer):
        """
        JSON mode (?mode=json) with the same statement/claim.
        """
        mock_core_answer.return_value = {
            "result": (
                "Verdict: False\n\nExplanation: Test JSON explanation: FG has not officially "
                "approved a ₦70,000 minimum wage.\n\nSources:\n- Premium Times — "
                "https://premiumtimesng.com/test-minimum-wage"
            ),
            "result_json": {
                "verdict": "False",
                "explanation": "Test JSON explanation: FG has not officially approved a ₦70,000 minimum wage.",
                "citations": [
                    {
                        "title": "Official FG circular",
                        "url": "https://gov.ng/circulars/minimum-wage-test",
                        "tier": "government",
                    },
                    {
                        "title": "Premium Times report",
                        "url": "https://premiumtimesng.com/test-minimum-wage",
                        "tier": "major_news",
                    },
                ],
                "confidence": 0.92,
            },
        }

        url = reverse("api-factcheck") + "?mode=json"
        payload = {"query": CLAIM_TEXT}

        response = self.client.post(url, payload, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        # Shape
        self.assertIn("query", data)
        self.assertIn("result_json", data)
        self.assertIn("latency_ms", data)

        self.assertEqual(data["query"], CLAIM_TEXT)

        rj = data["result_json"]
        self.assertEqual(rj["verdict"], "False")
        self.assertIn("Test JSON explanation", rj["explanation"])
        self.assertAlmostEqual(rj["confidence"], 0.92, places=2)

        # Citations
        self.assertEqual(len(rj["citations"]), 2)
        self.assertEqual(rj["citations"][0]["tier"], "government")
        self.assertEqual(rj["citations"][0]["url"], "https://gov.ng/circulars/minimum-wage-test")

        mock_core_answer.assert_called_once_with(CLAIM_TEXT, json_mode=True)


# ============================================================
# 🧪 Regression – get_external_api_answer may return dict sources
# ============================================================


class SourceHelpersMixedTypesTests(TestCase):
    """
    get_external_api_answer (app/views.py) returns sources as a list of
    {'title', 'url'} dicts. The API helpers must tolerate BOTH dicts and
    the older "Title - URL" strings without raising AttributeError.
    """

    def test_split_title_url_dict(self):
        d = {"title": "Premium Times", "url": "https://premiumtimesng.com/x"}
        self.assertEqual(_split_title_url(d), ("Premium Times", "https://premiumtimesng.com/x"))

    def test_split_title_url_str(self):
        self.assertEqual(
            _split_title_url("Premium Times - https://premiumtimesng.com/x"),
            ("Premium Times", "https://premiumtimesng.com/x"),
        )

    def test_rank_and_cap_mixed_type_list(self):
        mixed = [
            {"title": "Gov site", "url": "https://gov.ng/circulars/x"},
            "Nairametrics - https://facebook.com/Nairametrics/post",
            {"title": "BBC", "url": "https://bbc.com/news/123"},
        ]
        ranked = _rank_and_cap_sources(mixed, cap=5)
        self.assertIsInstance(ranked, list)
        self.assertTrue(ranked)
        for item in ranked:
            self.assertIsInstance(item, str)
            self.assertIn(" - ", item)

    def test_format_sources_paragraph_mixed(self):
        mixed = [
            {"title": "IGP", "url": "https://inec.gov.ng/official"},
            "Instagram - https://instagram.com/reel/x",
        ]
        block = _format_sources_paragraph(mixed, max_n=3)
        self.assertIn("IGP", block)
        self.assertIn("https://inec.gov.ng/official", block)
        self.assertIn("Instagram", block)
        self.assertIn("https://instagram.com/reel/x", block)


class FactcheckAPIHumanModeDictSourcesTests(APITestCase):
    """
    End-to-end: human mode should survive when get_external_api_answer
    returns dict sources (the ./views.py pipeline). Previously this raised
    AttributeError ('dict' object has no attribute 'strip') -> 500.
    """

    @patch("app.views_factcheck_api.get_external_api_answer")
    def test_api_human_mode_with_dict_sources(self, mock_get_answer):
        mock_get_answer.return_value = (
            "Verdict: Misleading\n\nExplanation: Test explanation.",
            [
                {"title": "Gov site", "url": "https://gov.ng/circulars/wage"},
                {"title": "Instagram post", "url": "https://instagram.com/p/abc"},
            ],
        )

        url = reverse("api-factcheck")
        response = self.client.post(url, {"query": CLAIM_TEXT}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("result", data)
        # The result should contain a renderable "Title - URL" source line
        self.assertIn("https://gov.ng/circulars/wage", data["result"])
