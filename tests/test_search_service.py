import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.search_service import SearchService


@pytest.fixture
def service():
    return SearchService(base_url="http://localhost:8080", max_results=3)


@pytest.mark.asyncio
async def test_search_returns_formatted_results(service):
    fake_response = {
        "results": [
            {"title": "Result One", "content": "Snippet one.", "url": "http://example.com/1"},
            {"title": "Result Two", "content": "Snippet two.", "url": "http://example.com/2"},
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("latest AI news")

    assert len(results) == 2
    assert results[0]["title"] == "Result One"
    assert results[0]["snippet"] == "Snippet one."
    assert results[0]["url"] == "http://example.com/1"


@pytest.mark.asyncio
async def test_search_respects_max_results(service):
    fake_response = {
        "results": [
            {"title": f"Result {i}", "content": f"Snippet {i}.", "url": f"http://example.com/{i}"}
            for i in range(10)
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_returns_empty_on_http_error(service):
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert results == []


@pytest.mark.asyncio
async def test_search_returns_empty_on_network_error(service):
    import aiohttp
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert results == []


@pytest.mark.asyncio
async def test_search_multiple_queries_parallel(service):
    fake_response = {
        "results": [
            {"title": "Result", "content": "Snippet.", "url": "http://example.com/1"},
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        all_results = await service.search_many(["query one", "query two"])

    # Each query returns 1 result, 2 queries total
    assert len(all_results) == 2
