import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import subprocess


def make_ctx():
    ctx = MagicMock()
    ctx.send = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_update_does_not_restart_on_pull_failure():
    """If git pull returns non-zero, bot should NOT restart."""
    from cogs.dev_commands import DevCommands

    bot = MagicMock()
    cog = DevCommands(bot)
    ctx = make_ctx()

    failed_result = MagicMock()
    failed_result.returncode = 1
    failed_result.stdout = ""
    failed_result.stderr = "CONFLICT (content): Merge conflict in bot.py"

    with patch("subprocess.run", return_value=failed_result), \
         patch("os.execv") as mock_exec:
        await cog.update.callback(cog, ctx)

    mock_exec.assert_not_called()
    ctx.send.assert_called()
    call_text = ctx.send.call_args[0][0]
    assert "failed" in call_text.lower() or "conflict" in call_text.lower() or "CONFLICT" in call_text


@pytest.mark.asyncio
async def test_update_restarts_on_successful_pull():
    """If git pull succeeds, bot should call os.execv."""
    from cogs.dev_commands import DevCommands

    bot = MagicMock()
    cog = DevCommands(bot)
    ctx = make_ctx()

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = "Already up to date."
    success_result.stderr = ""

    with patch("subprocess.run", return_value=success_result), \
         patch("os.execv") as mock_exec, \
         patch("sys.executable", "/usr/bin/python"), \
         patch("sys.argv", ["bot.py"]):
        await cog.update.callback(cog, ctx)

    mock_exec.assert_called_once()
