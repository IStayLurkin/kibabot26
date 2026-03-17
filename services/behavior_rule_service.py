from __future__ import annotations

import re

from database.behavior_rules_repository import (
    add_behavior_rule,
    clear_behavior_rules,
    list_behavior_rules,
    remove_behavior_rule,
    replace_behavior_rule,
    update_behavior_rule,
)


class BehaviorRuleService:
    async def get_rules(self) -> list[dict]:
        return await list_behavior_rules()

    async def get_enabled_rule_texts(self) -> list[str]:
        rules = await self.get_rules()
        return [rule["rule_text"] for rule in rules if rule["enabled"]]

    async def add_rule(self, rule_text: str, created_by: str = "") -> tuple[bool, str]:
        cleaned = self.normalize_rule_text(rule_text)
        if not cleaned:
            return False, "Provide a non-empty rule."

        await add_behavior_rule(cleaned, created_by=created_by)
        return True, f"Rule set: {cleaned}"

    async def delete_rule(self, rule_id: int) -> tuple[bool, str]:
        deleted = await remove_behavior_rule(rule_id)
        if not deleted:
            return False, f"I couldn't find a rule with ID `{rule_id}`."
        return True, f"Removed rule `{rule_id}`."

    async def edit_rule(self, rule_id: int, rule_text: str) -> tuple[bool, str]:
        cleaned = self.normalize_rule_text(rule_text)
        if not cleaned:
            return False, "Provide a non-empty replacement rule."

        updated = await update_behavior_rule(rule_id, cleaned)
        if not updated:
            return False, f"I couldn't find a rule with ID `{rule_id}`."
        return True, f"Updated rule `{rule_id}`: {cleaned}"

    async def replace_rule(self, old_rule_text: str, new_rule_text: str, created_by: str = "") -> tuple[bool, str]:
        cleaned_old = self.normalize_rule_text(old_rule_text)
        cleaned_new = self.normalize_rule_text(new_rule_text)
        if not cleaned_old or not cleaned_new:
            return False, "Provide both the current rule text and the replacement rule text."

        updated = await replace_behavior_rule(cleaned_old, cleaned_new, created_by=created_by)
        if not updated:
            return False, "I couldn't find that existing rule to replace."
        return True, f"Rule updated: {cleaned_new}"

    async def reset_rules(self) -> tuple[bool, str]:
        await clear_behavior_rules()
        return True, "Cleared all custom behavior rules."

    async def get_rules_text(self) -> str:
        rules = await self.get_rules()
        if not rules:
            return "No custom behavior rules are set."

        lines = ["Behavior rules:"]
        for rule in rules:
            lines.append(f"{rule['id']}. {rule['rule_text']}")
        return "\n".join(lines)

    def looks_like_rule_request(self, text: str) -> bool:
        lowered = text.strip().lower()
        prefixes = (
            "create a rule",
            "set a rule",
            "add a rule",
            "make a rule",
            "new rule",
            "rule:",
            "set rule",
            "create rule",
        )
        return lowered.startswith(prefixes)

    def looks_like_rule_edit_request(self, text: str) -> bool:
        lowered = text.strip().lower()
        prefixes = (
            "edit rule",
            "edit said rule",
            "change rule",
            "replace rule",
            "update rule",
        )
        return lowered.startswith(prefixes)

    def extract_rule_text(self, text: str) -> str:
        cleaned = text.strip()
        lowered = cleaned.lower()
        prefixes = (
            "create a rule",
            "set a rule",
            "add a rule",
            "make a rule",
            "new rule",
            "set rule",
            "create rule",
        )

        for prefix in prefixes:
            if lowered.startswith(prefix):
                extracted = cleaned[len(prefix):].strip(" .:-")
                return self.normalize_rule_text(extracted)

        if lowered.startswith("rule:"):
            return self.normalize_rule_text(cleaned[5:].strip())

        return self.normalize_rule_text(cleaned)

    def normalize_rule_text(self, rule_text: str) -> str:
        cleaned = " ".join(rule_text.strip().split())
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def extract_rule_replacement(self, text: str) -> tuple[str, str]:
        matches = []
        for pattern in (r'"([^"]+)"', r"'([^']+)'"):
            matches.extend(re.findall(pattern, text))

        if len(matches) >= 2:
            old_rule = self.normalize_rule_text(matches[0])
            new_rule = self.normalize_rule_text(matches[1])
            return old_rule, new_rule

        lowered = text.lower()
        if " to " in lowered:
            prefixes = (
                "edit rule",
                "edit said rule",
                "change rule",
                "replace rule",
                "update rule",
            )
            cleaned = text.strip()
            for prefix in prefixes:
                if lowered.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip(" :.-")
                    break
            parts = cleaned.split(" to ", 1)
            if len(parts) == 2:
                return self.normalize_rule_text(parts[0].strip(" :.-")), self.normalize_rule_text(parts[1].strip(" :.-"))

        return "", ""
