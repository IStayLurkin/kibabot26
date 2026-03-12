from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request

from osint_bot.core.config import OSINT_REQUEST_TIMEOUT_SECONDS


def fetch_json(url: str) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": "OSINTBot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=OSINT_REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return json.loads(payload)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def rdap_lookup(domain: str) -> tuple[str, dict | None]:
    source = f"https://rdap.org/domain/{urllib.parse.quote(domain)}"
    return source, fetch_json(source)


def dns_lookup(host: str) -> list[str]:
    _name, _aliases, ips = socket.gethostbyname_ex(host)
    return ips


def ssl_lookup(host: str) -> dict[str, str]:
    context = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=OSINT_REQUEST_TIMEOUT_SECONDS) as sock:
        with context.wrap_socket(sock, server_hostname=host) as secure_sock:
            cert = secure_sock.getpeercert()

    subject = dict(item[0] for item in cert.get("subject", []))
    issuer = dict(item[0] for item in cert.get("issuer", []))
    return {
        "subject_cn": subject.get("commonName", "unknown"),
        "issuer_cn": issuer.get("commonName", "unknown"),
        "not_before": cert.get("notBefore", "unknown"),
        "not_after": cert.get("notAfter", "unknown"),
    }


def http_metadata(url: str) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OSINTBot/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=OSINT_REQUEST_TIMEOUT_SECONDS) as response:
        server = response.headers.get("Server", "unknown")
        content_type = response.headers.get("Content-Type", "unknown")
        return {
            "final_url": response.geturl(),
            "status": str(response.status),
            "server": server,
            "content_type": content_type,
        }
