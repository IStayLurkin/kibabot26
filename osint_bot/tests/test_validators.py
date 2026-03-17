from __future__ import annotations

import unittest

from osint_bot.services.validators import (
    validate_domain,
    validate_ip,
    validate_url,
    validate_username,
)


class ValidatorTests(unittest.TestCase):
    def test_validate_domain(self) -> None:
        self.assertEqual(validate_domain("https://Example.com/path"), "example.com")

    def test_validate_url(self) -> None:
        self.assertEqual(validate_url("https://example.com"), "https://example.com")

    def test_validate_ip(self) -> None:
        self.assertEqual(validate_ip("8.8.8.8"), "8.8.8.8")

    def test_validate_username(self) -> None:
        self.assertEqual(validate_username("@user_name"), "@user_name")

    def test_invalid_domain_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_domain("bad domain")


if __name__ == "__main__":
    unittest.main()
