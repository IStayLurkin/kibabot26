from __future__ import annotations

import json
import logging
import socket
import ssl
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

    Intended uses:
    - public domain RDAP lookup
    - basic DNS resolution
    - SSL certificate subject/issuer peek
    - summarization of supplied public text
    """

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

        try:
            _, _, ips = socket.gethostbyname_ex(domain)
            if not ips:
                return f"No A records resolved for {domain}."
            return f"DNS A records for {domain}: {', '.join(ips)}"
        except socket.gaierror as exc:
            logger.exception("DNS lookup failed for %s", domain)
            return f"DNS lookup failed for {domain}: {exc}"

    async def ssl_lookup(self, domain: str) -> str:
        domain = self._normalize_domain(domain)
        if not domain:
            return "Invalid domain."

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

    def summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return "No text provided."
        if len(cleaned) <= 500:
            return f"Summary:\n{cleaned}"
        return f"Summary:\n{cleaned[:500]}..."

    async def _fetch_json(self, url: str) -> dict | None:
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
            *event_lines if event_lines else ["- none found"],
            "",
            "Nameservers:",
            *ns_lines if ns_lines else ["- none found"],
        ]
        return "\n".join(lines)