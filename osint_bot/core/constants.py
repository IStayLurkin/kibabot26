from __future__ import annotations

SAFE_USAGE_POLICY = (
    "Allowed use: public-source enrichment and owned/authorized asset checks only.\n"
    "Blocked use: private data gathering, covert collection, credential use, phishing, malware, "
    "intrusive automation, or requests targeting assets without authorization."
)

DISCORD_MESSAGE_SOFT_LIMIT = 1800

SUPPORTED_TARGET_TYPES = {
    "domain",
    "url",
    "ip",
    "username",
    "text",
}

OWNED_ASSET_TARGET_TYPES = {
    "domain",
    "url",
    "ip",
}
