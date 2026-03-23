import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch
from services.vector_memory_service import VectorMemoryService, _cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


@pytest.mark.asyncio
async def test_store_memory_calls_embed_and_db():
    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=[0.1] * 768)
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.store_vector_memory", new_callable=AsyncMock) as mock_store:
        svc = VectorMemoryService(embedding_service=mock_embed_svc)
        await svc.store(mock_db, user_id="123", content="Brandon loves Python")

    mock_embed_svc.embed.assert_called_once_with("Brandon loves Python")
    mock_store.assert_called_once()


@pytest.mark.asyncio
async def test_store_memory_skips_on_empty_embedding():
    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=[])  # embed failed
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.store_vector_memory", new_callable=AsyncMock) as mock_store:
        svc = VectorMemoryService(embedding_service=mock_embed_svc)
        await svc.store(mock_db, user_id="123", content="some text")

    mock_store.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_returns_top_k_by_similarity():
    import struct

    def pack(v):
        return struct.pack(f"{len(v)}f", *v)

    # Query vector pointing in direction of memory_a
    query_vec = [1.0, 0.0] + [0.0] * 766
    memory_a_vec = [1.0, 0.0] + [0.0] * 766   # cos sim = 1.0 (identical)
    memory_b_vec = [0.0, 1.0] + [0.0] * 766   # cos sim = 0.0 (orthogonal)

    class FakeRow:
        def __init__(self, content, embedding_blob):
            self._data = {"content": content, "embedding": embedding_blob}
        def __getitem__(self, key):
            return self._data[key]

    fake_rows = [
        FakeRow("Memory A", pack(memory_a_vec)),
        FakeRow("Memory B", pack(memory_b_vec)),
    ]

    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=query_vec)
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.get_all_vector_memories", new_callable=AsyncMock, return_value=fake_rows):
        svc = VectorMemoryService(embedding_service=mock_embed_svc, top_k=1)
        results = await svc.retrieve(mock_db, user_id="123", query="some query")

    assert len(results) == 1
    assert results[0] == "Memory A"
