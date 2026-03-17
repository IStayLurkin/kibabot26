from __future__ import annotations

import unittest

from osint_bot.bot import OSINTBot


class StartupTests(unittest.TestCase):
    def test_bot_initializes_isolated_service(self) -> None:
        bot = OSINTBot()
        self.assertEqual(bot.command_prefix, "!")
        self.assertIsNotNone(getattr(bot, "osint_service", None))


if __name__ == "__main__":
    unittest.main()

