from __future__ import annotations

import unittest

from osint_bot.core.policy import evaluate_request


class PolicyTests(unittest.TestCase):
    def test_blocks_intrusive_language(self) -> None:
        decision = evaluate_request("text", "need credentials for target", authorization=False)
        self.assertFalse(decision.allowed)
        self.assertIn("blocked", decision.blocked_reason.lower())

    def test_warns_on_active_checks_without_authorization(self) -> None:
        decision = evaluate_request("domain", "example.com", authorization=False)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.warnings)

    def test_allows_authorized_active_checks(self) -> None:
        decision = evaluate_request("domain", "example.com", authorization=True)
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
