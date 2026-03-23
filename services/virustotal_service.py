import asyncio
import aiohttp
from core.config import VIRUSTOTAL_API_KEY
from core.logging_config import get_logger

logger = get_logger(__name__)

_VT_BASE = "https://www.virustotal.com/api/v3"
_POLL_ATTEMPTS = 10
_POLL_DELAY = 3


async def _submit_url(session: aiohttp.ClientSession, url: str) -> str:
    """Submit URL to VirusTotal, return analysis ID."""
    headers = {"x-apikey": VIRUSTOTAL_API_KEY, "Content-Type": "application/x-www-form-urlencoded"}
    async with session.post(f"{_VT_BASE}/urls", data={"url": url}, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["data"]["id"]


async def _poll_result(session: aiohttp.ClientSession, analysis_id: str) -> dict:
    """Poll analysis until complete, return result dict."""
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    for _ in range(_POLL_ATTEMPTS):
        async with session.get(f"{_VT_BASE}/analyses/{analysis_id}", headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            status = data.get("data", {}).get("attributes", {}).get("status")
            if status == "completed":
                return data
        await asyncio.sleep(_POLL_DELAY)
    logger.warning("[virustotal] Poll exhausted all %d attempts without completed status for analysis %s", _POLL_ATTEMPTS, analysis_id)
    return {}


async def is_safe(url: str) -> bool:
    """Return True only if VirusTotal reports 0 malicious and 0 suspicious detections."""
    if not VIRUSTOTAL_API_KEY:
        logger.warning("[virustotal] No API key configured — skipping scan, treating as unsafe")
        return False
    try:
        async with aiohttp.ClientSession() as session:
            analysis_id = await _submit_url(session, url)
            result = await _poll_result(session, analysis_id)
            stats = result.get("data", {}).get("attributes", {}).get("stats", {})
            return stats.get("malicious", 1) == 0 and stats.get("suspicious", 1) == 0
    except Exception:
        logger.exception("[virustotal] Scan failed for %s — treating as unsafe", url)
        return False
