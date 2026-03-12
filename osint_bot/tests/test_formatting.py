from __future__ import annotations

import unittest

from osint_bot.services.formatting import build_discord_payload, render_result_text
from osint_bot.services.models import OSINTResult


class FormattingTests(unittest.TestCase):
    def test_render_includes_sections(self) -> None:
        result = OSINTResult(
            summary="Summary",
            findings=["one"],
            sources=["source"],
            warnings=["warning"],
        )
        text = render_result_text(result)
        self.assertIn("Findings:", text)
        self.assertIn("Sources:", text)
        self.assertIn("Warnings:", text)

    def test_build_payload_attaches_long_results(self) -> None:
        result = OSINTResult(
            summary="S" * 2000,
            findings=[],
            sources=[],
            warnings=[],
        )
        message, attachment = build_discord_payload(result)
        self.assertIn("attached", message.lower())
        self.assertIsNotNone(attachment)


if __name__ == "__main__":
    unittest.main()
