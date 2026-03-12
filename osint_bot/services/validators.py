from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse


DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))+$")
USERNAME_RE = re.compile(r"^@?[A-Za-z0-9_.-]{2,64}$")


def normalize_domain(value: str) -> str:
    value = value.strip().lower()
    value = value.removeprefix("http://").removeprefix("https://")
    value = value.split("/")[0].split(":")[0]
    return value


def validate_domain(value: str) -> str:
    domain = normalize_domain(value)
    if not DOMAIN_RE.match(domain):
        raise ValueError("Invalid domain.")
    return domain


def validate_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL.")
    return value.strip()


def validate_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError as exc:
        raise ValueError("Invalid IP address.") from exc


def validate_username(value: str) -> str:
    username = value.strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Invalid username.")
    return username
