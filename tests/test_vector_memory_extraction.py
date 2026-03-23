import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.memory_service import maybe_store_episodic_memory


@pytest.mark.asyncio
async def test_stores_memory_when_llm_says_yes():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(return_value={
        "should_store": True,
        "content": "Brandon is building a Discord bot in Python"
    })
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="I'm building a Discord bot",
        bot_reply="That's cool!"
    )

    mock_vms.store.assert_called_once_with(mock_db, user_id="123", content="Brandon is building a Discord bot in Python")


@pytest.mark.asyncio
async def test_skips_when_llm_says_no():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(return_value={"should_store": False, "content": ""})
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="hey",
        bot_reply="hey"
    )

    mock_vms.store.assert_not_called()


@pytest.mark.asyncio
async def test_skips_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(side_effect=RuntimeError("LLM down"))
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    # Should not raise
    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="some text",
        bot_reply="some reply"
    )

    mock_vms.store.assert_not_called()


@pytest.mark.asyncio
async def test_skips_short_messages():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock()
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="ok",
        bot_reply="ok"
    )

    mock_llm.extract_episodic_memory.assert_not_called()
    mock_vms.store.assert_not_called()
