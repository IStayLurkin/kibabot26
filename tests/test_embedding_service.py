import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.embedding_service import EmbeddingService


@pytest.fixture
def svc():
    return EmbeddingService(base_url="http://localhost:11434", model="nomic-embed-text")


@pytest.mark.asyncio
async def test_embed_returns_list_of_floats(svc):
    fake_response = {"embedding": [0.1, 0.2, 0.3] * 256}  # 768 dims
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=fake_response)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await svc.embed("hello world")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_returns_empty_on_error(svc):
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await svc.embed("hello world")

    assert result == []


@pytest.mark.asyncio
async def test_embed_many_returns_list_of_embeddings(svc):
    fake_embedding = [0.1, 0.2, 0.3] * 256
    fake_response = {"embedding": fake_embedding}
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=fake_response)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        results = await svc.embed_many(["hello", "world"])

    assert len(results) == 2
    assert all(len(e) == 768 for e in results)
