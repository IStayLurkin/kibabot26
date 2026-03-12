from __future__ import annotations

import json

from osint_bot.core.constants import SAFE_USAGE_POLICY
from osint_bot.core.policy import evaluate_request
from osint_bot.services import adapters
from osint_bot.services.models import OSINTRequest, OSINTResult
from osint_bot.services.validators import (
    normalize_domain,
    validate_domain,
    validate_ip,
    validate_url,
    validate_username,
)


class OSINTService:
    def __init__(self, llm_service=None) -> None:
        self.llm_service = llm_service

    async def handle_request(self, request: OSINTRequest) -> OSINTResult:
        decision = evaluate_request(request.target_type, request.target_value, request.authorization)
        if not decision.allowed:
            return OSINTResult(
                summary="Request blocked.",
                findings=[],
                sources=[],
                warnings=[],
                blocked_reason=decision.blocked_reason,
            )

        if request.target_type == "domain":
            return await self._handle_domain(request, decision.warnings)
        if request.target_type == "url":
            return await self._handle_url(request, decision.warnings)
        if request.target_type == "ip":
            return await self._handle_ip(request, decision.warnings)
        if request.target_type == "username":
            return await self._handle_username(request, decision.warnings)
        if request.target_type == "text":
            return await self._handle_text(request, decision.warnings)

        return OSINTResult(
            summary="Unsupported target type.",
            findings=[],
            sources=[],
            warnings=decision.warnings,
            blocked_reason="Unsupported target type.",
        )

    async def policy_text(self) -> OSINTResult:
        return OSINTResult(
            summary="OSINT bot safety policy",
            findings=[SAFE_USAGE_POLICY],
            sources=[],
            warnings=[],
        )

    async def _handle_domain(self, request: OSINTRequest, warnings: list[str]) -> OSINTResult:
        domain = validate_domain(request.target_value)
        findings: list[str] = [f"Normalized domain: {domain}"]
        sources: list[str] = []
        raw_sections: dict[str, str] = {}

        source, rdap_data = adapters.rdap_lookup(domain)
        sources.append(source)
        if rdap_data:
            findings.append(f"RDAP handle: {rdap_data.get('handle', 'unknown')}")
            status = rdap_data.get("status") or []
            findings.append(f"RDAP status: {', '.join(status) if status else 'unknown'}")
            raw_sections["rdap"] = json.dumps(rdap_data, indent=2)[:4000]
        else:
            findings.append("RDAP data unavailable.")

        if request.authorization:
            try:
                ips = adapters.dns_lookup(domain)
                findings.append("DNS A records: " + (", ".join(ips) if ips else "none"))
                raw_sections["dns"] = json.dumps({"ips": ips}, indent=2)
            except Exception as exc:
                findings.append(f"DNS lookup failed: {exc}")

            try:
                ssl_data = adapters.ssl_lookup(domain)
                findings.extend(
                    [
                        f"SSL subject CN: {ssl_data['subject_cn']}",
                        f"SSL issuer CN: {ssl_data['issuer_cn']}",
                        f"SSL valid to: {ssl_data['not_after']}",
                    ]
                )
                raw_sections["ssl"] = json.dumps(ssl_data, indent=2)
            except Exception as exc:
                findings.append(f"SSL lookup failed: {exc}")

        summary = await self._summarize("domain", domain, findings, warnings)
        return OSINTResult(summary=summary, findings=findings, sources=sources, warnings=warnings, raw_sections=raw_sections)

    async def _handle_url(self, request: OSINTRequest, warnings: list[str]) -> OSINTResult:
        url = validate_url(request.target_value)
        domain = normalize_domain(url)
        findings = [f"Normalized URL: {url}", f"Host: {domain}"]
        sources = [url]
        raw_sections: dict[str, str] = {}

        if request.authorization:
            try:
                metadata = adapters.http_metadata(url)
                findings.extend(
                    [
                        f"HTTP status: {metadata['status']}",
                        f"Server header: {metadata['server']}",
                        f"Content type: {metadata['content_type']}",
                        f"Final URL: {metadata['final_url']}",
                    ]
                )
                raw_sections["http"] = json.dumps(metadata, indent=2)
            except Exception as exc:
                findings.append(f"HTTP metadata lookup failed: {exc}")

        summary = await self._summarize("url", url, findings, warnings)
        return OSINTResult(summary=summary, findings=findings, sources=sources, warnings=warnings, raw_sections=raw_sections)

    async def _handle_ip(self, request: OSINTRequest, warnings: list[str]) -> OSINTResult:
        ip_value = validate_ip(request.target_value)
        findings = [f"Validated IP: {ip_value}"]
        summary = await self._summarize("ip", ip_value, findings, warnings)
        return OSINTResult(summary=summary, findings=findings, sources=[], warnings=warnings)

    async def _handle_username(self, request: OSINTRequest, warnings: list[str]) -> OSINTResult:
        username = validate_username(request.target_value)
        clean_username = username.removeprefix("@")
        findings = [
            f"Normalized username: @{clean_username}",
            "Public enrichment only. No private-platform scraping is performed.",
            f"Suggested next step: manually review public profiles matching @{clean_username}.",
        ]
        summary = await self._summarize("username", clean_username, findings, warnings)
        return OSINTResult(summary=summary, findings=findings, sources=[], warnings=warnings)

    async def _handle_text(self, request: OSINTRequest, warnings: list[str]) -> OSINTResult:
        text = request.target_value.strip()
        if not text:
            raise ValueError("No text provided.")
        findings = [text[:1200]]
        summary = await self._summarize("text", text[:280], findings, warnings)
        return OSINTResult(summary=summary, findings=findings, sources=[], warnings=warnings)

    async def _summarize(
        self,
        target_type: str,
        target_value: str,
        findings: list[str],
        warnings: list[str],
    ) -> str:
        if self.llm_service is None:
            return f"Safe {target_type} summary for {target_value}"

        prompt = (
            f"Target type: {target_type}\n"
            f"Target value: {target_value}\n"
            f"Warnings: {warnings or ['none']}\n"
            "Findings:\n- " + "\n- ".join(findings)
        )
        try:
            return await self.llm_service.summarize_findings(prompt)
        except Exception:
            return f"Safe {target_type} summary for {target_value}"
