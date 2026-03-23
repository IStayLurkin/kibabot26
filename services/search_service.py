import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECONDS = 5


class SearchService:
    def __init__(self, base_url: str, max_results: int = 3):
        self._base_url = base_url.rstrip("/")
        self._max_results = max_results

    async def search(self, query: str) -> list[dict]:
        """Search SearXNG for query. Returns list of {title, snippet, url} dicts. Never raises."""
        url = f"{self._base_url}/search"
        params = {"q": query, "format": "json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=_SEARCH_TIMEOUT_SECONDS),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("SearXNG returned status %d for query %r", resp.status, query)
                        return []
                    data = await resp.json()
                    results = data.get("results", [])
                    return [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("content", ""),
                            "url": r.get("url", ""),
                        }
                        for r in results[: self._max_results]
                        if r.get("title") or r.get("content")
                    ]
        except Exception as exc:
            logger.warning("SearXNG search failed for query %r: %s", query, exc)
            return []

    async def search_many(self, queries: list[str]) -> list[dict]:
        """Run multiple queries in parallel. Returns combined results list."""
        if not queries:
            return []
        results_per_query = await asyncio.gather(*[self.search(q) for q in queries])
        combined = []
        for results in results_per_query:
            combined.extend(results)
        return combined
