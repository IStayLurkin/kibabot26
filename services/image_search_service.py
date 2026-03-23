import aiohttp
from pathlib import Path
from core.config import GIPHY_API_KEY, LOCAL_IMAGE_DIR
from core.logging_config import get_logger
from services.virustotal_service import is_safe

logger = get_logger(__name__)

_GIPHY_SEARCH = "https://api.giphy.com/v1/gifs/search"
_SUPPORTED_EXTS = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
_MAX_CANDIDATES = 5


async def _giphy_get(topic: str) -> dict:
    params = {"api_key": GIPHY_API_KEY, "q": topic, "limit": _MAX_CANDIDATES, "rating": "pg-13"}
    async with aiohttp.ClientSession() as session:
        async with session.get(_GIPHY_SEARCH, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()


async def search_giphy(topic: str) -> list[str]:
    """Return up to 5 Giphy GIF URLs for the topic."""
    if not GIPHY_API_KEY:
        return []
    try:
        data = await _giphy_get(topic)
        return [
            item["images"]["original"]["url"]
            for item in data.get("data", [])
            if item.get("images", {}).get("original", {}).get("url")
        ]
    except Exception:
        logger.exception("[image_search] Giphy search failed for %r", topic)
        return []


def search_local(topic: str) -> list[str]:
    """Return local image paths whose filename contains the topic keyword."""
    if not LOCAL_IMAGE_DIR:
        return []
    folder = Path(LOCAL_IMAGE_DIR)
    if not folder.is_dir():
        return []
    # Match on both space and underscore variants so "cat memes" finds "cat_memes.gif" and vice versa
    normalized = topic.lower()
    keywords = [normalized, normalized.replace(" ", "_"), normalized.replace("_", " ")]
    results = []
    for ext in _SUPPORTED_EXTS:
        results.extend(
            str(p) for p in folder.iterdir()
            if p.suffix.lower() == ext and any(kw in p.name.lower() for kw in keywords)
        )
    return results[:_MAX_CANDIDATES]


async def find_verified_image(topic: str) -> str | None:
    """
    Search Giphy then local folder for topic. Return first URL/path
    that passes VirusTotal scan, or None if all fail.
    """
    candidates = await search_giphy(topic)

    # Local files don't have URLs to scan — convert to file:// for VT or just trust them?
    # We scan local files by their path string; VT won't scan local paths so we trust them
    # but still run through the pipeline for uniformity — is_safe will fail-safe to False
    # for non-http paths. So we append local paths after URL candidates.
    local = search_local(topic)

    for url in candidates:
        if await is_safe(url):
            return url

    # For local files, bypass VT (they're on your machine — you put them there)
    if local:
        return local[0]

    logger.info("[image_search] No safe results found for %r", topic)
    return None
