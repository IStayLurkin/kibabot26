from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from osint_bot.services.models import OSINTRequest
from osint_bot.services.osint_service import OSINTService


class FakeLLM:
    async def summarize_findings(self, prompt: str) -> str:
        return "LLM summary"


class OSINTServiceTests(unittest.TestCase):
    def test_text_summary_flow(self) -> None:
        service = OSINTService(llm_service=FakeLLM())
        result = asyncio.run(
            service.handle_request(OSINTRequest(target_type="text", target_value="hello world", mode="summarize_only"))
        )
        self.assertEqual(result.summary, "LLM summary")
        self.assertTrue(result.allowed)

    def test_blocked_request(self) -> None:
        service = OSINTService(llm_service=FakeLLM())
        result = asyncio.run(
            service.handle_request(OSINTRequest(target_type="text", target_value="steal credentials", mode="summarize_only"))
        )
        self.assertFalse(result.allowed)

    @patch("osint_bot.services.adapters.rdap_lookup")
    @patch("osint_bot.services.adapters.ssl_lookup")
    @patch("osint_bot.services.adapters.dns_lookup")
    def test_authorized_domain_flow(self, mock_dns, mock_ssl, mock_rdap) -> None:
        mock_dns.return_value = ["1.1.1.1"]
        mock_ssl.return_value = {
            "subject_cn": "example.com",
            "issuer_cn": "Example CA",
            "not_before": "yesterday",
            "not_after": "tomorrow",
        }
        mock_rdap.return_value = ("https://rdap.org/domain/example.com", {"handle": "ABC", "status": ["active"]})

        service = OSINTService(llm_service=FakeLLM())
        result = asyncio.run(
            service.handle_request(
                OSINTRequest(
                    target_type="domain",
                    target_value="example.com",
                    mode="owned_asset_check",
                    authorization=True,
                )
            )
        )
        self.assertEqual(result.summary, "LLM summary")
        self.assertIn("DNS A records: 1.1.1.1", result.findings)


if __name__ == "__main__":
    unittest.main()
