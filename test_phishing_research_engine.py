"""
Testy jednostkowe dla PhishingResearchEngine.
Testowane moduły:
  - mask_links() i _mask_links_in_dict()
  - scrape_url() z mockowanymi odpowiedziami
  - call_llm() z mockowanym klientem OpenAI
  - stage1_synthesize(), stage2_profile(), stage3_generate()
  - run_pipeline() – pełny pipeline na mockach

CEL EDUKACYJNY I BADAWCZY: testy wyłącznie na potrzeby pracy inżynierskiej.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from phishing_research_engine import (
    SIMULATED_LINK_PLACEHOLDER,
    AnalysisResult,
    PhishingResearchEngine,
    ScrapedPage,
    save_report,
)


# ---------------------------------------------------------------------------
# Pomocnicze dane testowe
# ---------------------------------------------------------------------------

SAMPLE_FACTS = {
    "profession": "Software Engineer",
    "technologies": ["Python", "Linux", "Git"],
    "writing_style": "technical, informal",
    "interests": ["open source", "kernel development"],
    "key_facts": ["works at tech company", "active on GitHub"],
}

SAMPLE_PROFILE = {
    "psychological_profile": "Motivated by open-source recognition and technical challenges.",
    "attack_vectors": [
        {
            "type": "technical",
            "description": "Fake security advisory for a repository the target contributes to.",
            "rationale": "Target cares deeply about code quality.",
        },
        {
            "type": "recruitment",
            "description": "Fake recruiter from top tech company.",
            "rationale": "Target values career growth.",
        },
        {
            "type": "hobby",
            "description": "Fake invitation to exclusive open-source conference.",
            "rationale": "Target is passionate about open source.",
        },
    ],
}

SAMPLE_EMAIL = {
    "subject": "Security alert for your repository",
    "sender_name": "GitHub Security Team",
    "html_body": (
        "<p>Please review this link: "
        "[SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY]</p>"
    ),
    "plain_body": (
        "Please review: [SIMULATED_PHISHING_LINK_FOR_EDUCATION_ONLY]"
    ),
}


# ---------------------------------------------------------------------------
# Pomocnik: tworzenie mockowanej odpowiedzi OpenAI
# ---------------------------------------------------------------------------

def _mock_openai_response(content: dict) -> MagicMock:
    """Tworzy mockowany obiekt odpowiedzi OpenAI."""
    msg = MagicMock()
    msg.content = json.dumps(content)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Testy maskowania linków
# ---------------------------------------------------------------------------

class TestMaskLinks(unittest.TestCase):
    def setUp(self):
        self.engine = PhishingResearchEngine.__new__(PhishingResearchEngine)

    def test_http_url_replaced(self):
        text = "Odwiedź http://example.com/login teraz."
        result = self.engine.mask_links(text)
        self.assertNotIn("http://example.com", result)
        self.assertIn(SIMULATED_LINK_PLACEHOLDER, result)

    def test_https_url_replaced(self):
        text = "Kliknij https://secure.bank.com/reset?token=abc123"
        result = self.engine.mask_links(text)
        self.assertNotIn("https://secure.bank.com", result)
        self.assertIn(SIMULATED_LINK_PLACEHOLDER, result)

    def test_multiple_urls_replaced(self):
        text = "A: http://a.com, B: https://b.org/path"
        result = self.engine.mask_links(text)
        self.assertEqual(result.count(SIMULATED_LINK_PLACEHOLDER), 2)

    def test_no_url_unchanged(self):
        text = "Brak żadnego linku w tym tekście."
        self.assertEqual(self.engine.mask_links(text), text)

    def test_mask_links_in_dict_strings(self):
        data = {
            "subject": "Check http://evil.com now",
            "body": "Visit https://phish.org/page",
        }
        masked = self.engine._mask_links_in_dict(data)
        for val in masked.values():
            self.assertNotIn("http", val)
            self.assertIn(SIMULATED_LINK_PLACEHOLDER, val)

    def test_mask_links_in_dict_nested(self):
        data = {
            "outer": {
                "inner": "Go to http://nested.com",
            }
        }
        masked = self.engine._mask_links_in_dict(data)
        self.assertIn(SIMULATED_LINK_PLACEHOLDER, masked["outer"]["inner"])

    def test_mask_links_in_dict_list(self):
        data = {"items": ["http://a.com", "no link here", "https://b.com"]}
        masked = self.engine._mask_links_in_dict(data)
        self.assertNotIn("http://a.com", masked["items"])
        self.assertEqual(masked["items"][1], "no link here")
        self.assertNotIn("https://b.com", masked["items"])


# ---------------------------------------------------------------------------
# Testy ScrapedPage
# ---------------------------------------------------------------------------

class TestScrapedPage(unittest.TestCase):
    def test_is_ok_with_text(self):
        page = ScrapedPage(url="http://x.com", text="some content")
        self.assertTrue(page.is_ok)

    def test_is_ok_with_error(self):
        page = ScrapedPage(url="http://x.com", text="", error="404")
        self.assertFalse(page.is_ok)

    def test_is_ok_empty_text_no_error(self):
        page = ScrapedPage(url="http://x.com", text="")
        self.assertFalse(page.is_ok)


# ---------------------------------------------------------------------------
# Testy scrapingu
# ---------------------------------------------------------------------------

class TestScrapeUrl(unittest.TestCase):
    def setUp(self):
        self.engine = PhishingResearchEngine.__new__(PhishingResearchEngine)

    @patch("phishing_research_engine.trafilatura.fetch_url")
    @patch("phishing_research_engine.trafilatura.extract")
    def test_scrape_success(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>content</html>"
        mock_extract.return_value = "Extracted text content"

        page = self.engine.scrape_url("http://test.com")

        self.assertTrue(page.is_ok)
        self.assertEqual(page.text, "Extracted text content")
        self.assertIsNone(page.error)

    @patch("phishing_research_engine.trafilatura.fetch_url")
    def test_scrape_fetch_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        page = self.engine.scrape_url("http://test.com")
        self.assertFalse(page.is_ok)
        self.assertIsNotNone(page.error)

    @patch("phishing_research_engine.trafilatura.fetch_url")
    @patch("phishing_research_engine.trafilatura.extract")
    def test_scrape_empty_extraction(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html></html>"
        mock_extract.return_value = None
        page = self.engine.scrape_url("http://test.com")
        self.assertFalse(page.is_ok)

    @patch("phishing_research_engine.trafilatura.fetch_url")
    def test_scrape_exception_handled(self, mock_fetch):
        mock_fetch.side_effect = Exception("Connection timeout")
        page = self.engine.scrape_url("http://test.com")
        self.assertFalse(page.is_ok)
        self.assertIn("timeout", page.error.lower())

    @patch("phishing_research_engine.trafilatura.fetch_url")
    @patch("phishing_research_engine.trafilatura.extract")
    def test_scrape_multiple_urls(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>x</html>"
        mock_extract.side_effect = ["Text A", "Text B", None]

        pages = self.engine.scrape_urls(["http://a.com", "http://b.com", "http://c.com"])
        self.assertEqual(len(pages), 3)
        self.assertTrue(pages[0].is_ok)
        self.assertTrue(pages[1].is_ok)
        self.assertFalse(pages[2].is_ok)


# ---------------------------------------------------------------------------
# Testy call_llm
# ---------------------------------------------------------------------------

class TestCallLLM(unittest.TestCase):
    def _make_engine(self, llm_response: dict) -> PhishingResearchEngine:
        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()
        engine.client.chat.completions.create.return_value = _mock_openai_response(
            llm_response
        )
        return engine

    def test_call_llm_returns_dict(self):
        engine = self._make_engine({"key": "value"})
        result = engine.call_llm("system", "user")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")

    def test_call_llm_json_fallback(self):
        """Jeśli model zwróci JSON osadzony w tekście, powinno być wyodrębnione."""
        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()
        msg = MagicMock()
        msg.content = 'Sure! Here is the JSON: {"answer": 42}'
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        engine.client.chat.completions.create.return_value = resp

        result = engine.call_llm("system", "user")
        self.assertEqual(result.get("answer"), 42)

    def test_call_llm_exception_returns_error(self):
        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()
        engine.client.chat.completions.create.side_effect = Exception("LLM down")
        result = engine.call_llm("system", "user")
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Testy etapów pipeline
# ---------------------------------------------------------------------------

class TestPipelineStages(unittest.TestCase):
    def _make_engine(self, llm_response: dict) -> PhishingResearchEngine:
        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()
        engine.client.chat.completions.create.return_value = _mock_openai_response(
            llm_response
        )
        return engine

    def test_stage1_returns_facts(self):
        engine = self._make_engine(SAMPLE_FACTS)
        result = engine.stage1_synthesize("Some scraped text about a developer.")
        self.assertEqual(result["profession"], "Software Engineer")
        self.assertIn("Python", result["technologies"])

    def test_stage2_returns_profile(self):
        engine = self._make_engine(SAMPLE_PROFILE)
        result = engine.stage2_profile(SAMPLE_FACTS)
        self.assertIn("psychological_profile", result)
        self.assertEqual(len(result["attack_vectors"]), 3)

    def test_stage3_returns_email(self):
        engine = self._make_engine(SAMPLE_EMAIL)
        result = engine.stage3_generate(SAMPLE_PROFILE, vector_index=0)
        self.assertIn("subject", result)
        self.assertIn("html_body", result)
        self.assertIn("plain_body", result)

    def test_stage3_links_are_masked(self):
        """Upewnij się, że etap 3 zawsze maskuje linki."""
        email_with_real_link = {
            "subject": "Check this",
            "sender_name": "Admin",
            "html_body": "<a href='http://evil.com/steal'>Click</a>",
            "plain_body": "Go to http://evil.com/steal now",
        }
        engine = self._make_engine(email_with_real_link)
        result = engine.stage3_generate(SAMPLE_PROFILE, vector_index=0)
        self.assertNotIn("http://evil.com", result.get("html_body", ""))
        self.assertNotIn("http://evil.com", result.get("plain_body", ""))

    def test_stage3_vector_index_out_of_bounds(self):
        """Indeks poza zakresem – powinien użyć ostatniego wektora."""
        engine = self._make_engine(SAMPLE_EMAIL)
        # Nie powinno rzucić wyjątku
        result = engine.stage3_generate(SAMPLE_PROFILE, vector_index=99)
        self.assertIsInstance(result, dict)

    def test_stage3_empty_vectors(self):
        """Brak wektorów ataku – powinno działać bez błędów."""
        engine = self._make_engine(SAMPLE_EMAIL)
        result = engine.stage3_generate({"psychological_profile": "x", "attack_vectors": []})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Test pełnego pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline(unittest.TestCase):
    @patch("phishing_research_engine.trafilatura.fetch_url")
    @patch("phishing_research_engine.trafilatura.extract")
    def test_full_pipeline(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>content</html>"
        mock_extract.return_value = "Developer profile text with Python and Linux experience."

        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()

        # Trzy kolejne wywołania LLM zwracają fakty, profil, e-mail
        engine.client.chat.completions.create.side_effect = [
            _mock_openai_response(SAMPLE_FACTS),
            _mock_openai_response(SAMPLE_PROFILE),
            _mock_openai_response(SAMPLE_EMAIL),
        ]

        result = engine.run_pipeline(["http://test.com/profile"])

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.stage1_facts["profession"], "Software Engineer")
        self.assertEqual(len(result.stage2_profile["attack_vectors"]), 3)
        self.assertIn("subject", result.stage3_email)

    @patch("phishing_research_engine.trafilatura.fetch_url")
    def test_pipeline_no_pages_scraped(self, mock_fetch):
        mock_fetch.return_value = None  # Wszystkie strony niedostępne

        engine = PhishingResearchEngine.__new__(PhishingResearchEngine)
        engine.model = "test-model"
        engine.client = MagicMock()

        result = engine.run_pipeline(["http://unreachable.com"])

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.stage1_facts, {})  # pipeline przerwany


# ---------------------------------------------------------------------------
# Test zapisu raportu
# ---------------------------------------------------------------------------

class TestSaveReport(unittest.TestCase):
    def test_save_report_creates_file(self):
        import os
        import tempfile

        result = AnalysisResult(
            urls=["http://test.com"],
            stage1_facts=SAMPLE_FACTS,
            stage2_profile=SAMPLE_PROFILE,
            stage3_email=SAMPLE_EMAIL,
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_report(result, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("_disclaimer", data)
            self.assertIn("stage1_facts", data)
            self.assertIn("stage3_email", data)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Uruchomienie testów
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
