import pytest

@pytest.mark.asyncio
async def test_delete_user_history_runs_without_error():
    from database.chat_memory import delete_user_history
    await delete_user_history("999999", "888888")

@pytest.mark.asyncio
async def test_delete_channel_history_runs_without_error():
    from database.chat_memory import delete_channel_history
    await delete_channel_history("888888")
