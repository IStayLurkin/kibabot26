from __future__ import annotations

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackageBoundaryTests(unittest.TestCase):
    def test_osint_bot_does_not_import_main_bot_modules(self) -> None:
        forbidden = {"cogs", "core", "services", "database", "tasks"}
        allowed_root = "osint_bot"

        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    self.assertTrue(
                        top == allowed_root or top not in forbidden,
                        f"{path} imports forbidden module {node.module}",
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertTrue(
                            top == allowed_root or top not in forbidden,
                            f"{path} imports forbidden module {alias.name}",
                        )


if __name__ == "__main__":
    unittest.main()
