from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WhoisSummary:
    domain: str
    rdap_source: str
    raw_json: dict


class OSINTService:
    """
    Safe public-info OSINT only.
    Preserved original architecture with 2026 Agentic Dossier additions.
    """

    def __init__(self, performance_tracker=None) -> None:
        self.performance_tracker = performance_tracker
        self._rdap_cache: dict[str, tuple[float, dict | None]] = {}
        self._dns_cache: dict[str, tuple[float, str]] = {}
        self._ssl_cache: dict[str, tuple[float, str]] = {}
        self.cache_ttl_seconds = 300

    async def lookup_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "No query provided."

        return (
            "Safe OSINT mode is enabled.\n"
            f"Query received: {query}\n"
            "Use specific commands like !whois <domain> for domain intelligence, "
            "or provide public text/data to summarize."
        )

    # --- NEW: AGENTIC DOSSIER ADDITION ---
    async def run_dossier(self, target: str) -> str:
        """
        2026 Expansion: Runs lookups in parallel for a target.
        Leverages 3090 Ti concurrent execution.
        """
        started_at = time.perf_counter()
        
        # Gathering your original lookup methods in parallel
        tasks = [
            self.dns_lookup(target),
            self.ssl_lookup(target),
            self.whois_lookup(target)
        ]
        results = await asyncio.gather(*tasks)
        
        self._record_duration("osint.run_dossier", started_at)
        return "\n\n--- AGENTIC DOSSIER REPORT ---\n\n" + "\n\n".join(results)

    async def whois_lookup(self, domain: str) -> str:
        domain = self._normalize_domain(domain)
        if not domain:
            return "Invalid domain."

        rdap_url = f"https://rdap.org/domain/{urllib.parse.quote(domain)}"
        raw = await self._fetch_json(rdap_url)

        if raw is None:
            return f"Could not retrieve RDAP data for {domain}."

        summary = self._format_rdap_summary(domain, rdap_url, raw)
        return summary

    async def dns_lookup(self, domain: str) -> str:
        domain = self._normalize_domain(domain)
        if not domain:
            return "Invalid domain."

        cached = self._get_cached(self._dns_cache, domain)
        if cached is not None:
            return cached

        started_at = time.perf_counter()
        result = await asyncio.to_thread(self._dns_lookup_sync, domain)
        self._set_cached(self._dns_cache, domain, result)
        self._record_duration("osint.dns_lookup", started_at)
        return result

    async def ssl_lookup(self, domain: str) -> str:
        domain = self._normalize_domain(domain)
        if not domain:
            return "Invalid domain."

        cached = self._get_cached(self._ssl_cache, domain)
        if cached is not None:
            return cached

        started_at = time.perf_counter()
        result = await asyncio.to_thread(self._ssl_lookup_sync, domain)
        self._set_cached(self._ssl_cache, domain, result)
        self._record_duration("osint.ssl_lookup", started_at)
        return result

    def summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return "No text provided."
        if len(cleaned) <= 500:
            return f"Summary:\n{cleaned}"
        return f"Summary:\n{cleaned[:500]}..."

    async def _fetch_json(self, url: str) -> dict | None:
        cached = self._get_cached(self._rdap_cache, url)
        if cached is not None:
            return cached

        started_at = time.perf_counter()
        data = await asyncio.to_thread(self._fetch_json_sync, url)
        self._set_cached(self._rdap_cache, url, data)
        self._record_duration("osint.fetch_json", started_at)
        return data

    def _fetch_json_sync(self, url: str) -> dict | None:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "KibaBot/1.0"
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = response.read().decode("utf-8", errors="replace")
                return json.loads(data)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            logger.exception("Failed to fetch JSON from %s", url)
            return None

    def _dns_lookup_sync(self, domain: str) -> str:
        try:
            _, _, ips = socket.gethostbyname_ex(domain)
            if not ips:
                return f"No A records resolved for {domain}."
            return f"DNS A records for {domain}: {', '.join(ips)}"
        except socket.gaierror as exc:
            logger.exception("DNS lookup failed for %s", domain)
            return f"DNS lookup failed for {domain}: {exc}"

    def _ssl_lookup_sync(self, domain: str) -> str:
        context = ssl.create_default_context()

        try:
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as secure_sock:
                    cert = secure_sock.getpeercert()

            subject = dict(x[0] for x in cert.get("subject", []))
            issuer = dict(x[0] for x in cert.get("issuer", []))
            not_before = cert.get("notBefore", "unknown")
            not_after = cert.get("notAfter", "unknown")

            return (
                f"SSL info for {domain}\n"
                f"Subject CN: {subject.get('commonName', 'unknown')}\n"
                f"Issuer CN: {issuer.get('commonName', 'unknown')}\n"
                f"Valid From: {not_before}\n"
                f"Valid To: {not_after}"
            )
        except Exception as exc:
            logger.exception("SSL lookup failed for %s", domain)
            return f"SSL lookup failed for {domain}: {exc}"

    def _normalize_domain(self, value: str) -> str:
        value = value.strip().lower()
        value = value.removeprefix("http://").removeprefix("https://")
        value = value.split("/")[0]
        value = value.split(":")[0]
        return value

    def _format_rdap_summary(self, domain: str, source: str, raw: dict) -> str:
        status = raw.get("status", [])
        events = raw.get("events", [])
        nameservers = raw.get("nameservers", [])

        event_lines: list[str] = []
        for event in events[:5]:
            action = event.get("eventAction", "unknown")
            date = event.get("eventDate", "unknown")
            event_lines.append(f"- {action}: {date}")

        ns_lines: list[str] = []
        for nameserver in nameservers[:5]:
            name = nameserver.get("ldhName")
            if name:
                ns_lines.append(f"- {name}")

        lines = [
            f"RDAP summary for {domain}",
            f"Source: {source}",
            f"Handle: {raw.get('handle', 'unknown')}",
            f"Status: {', '.join(status) if status else 'unknown'}",
            "",
            "Recent RDAP events:",
        ]
        lines.extend(event_lines or ["- none found"])
        lines.extend([
            "",
            "Nameservers:",
        ])
        lines.extend(ns_lines or ["- none found"])
        return "\n".join(lines)

    def _record_duration(self, name: str, started_at: float) -> None:
        if self.performance_tracker is None:
            return

        self.performance_tracker.record_service_call(
            name,
            (time.perf_counter() - started_at) * 1000,
        )

    def _get_cached(self, cache: dict, key: str):
        cached = cache.get(key)
        if cached is None:
            return None

        expires_at, value = cached
        if expires_at < time.time():
            cache.pop(key, None)
            return None

        return value

    def _set_cached(self, cache: dict, key: str, value):
        cache[key] = (time.time() + self.cache_ttl_seconds, value)