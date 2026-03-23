import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.virustotal_service import is_safe


@pytest.mark.asyncio
async def test_is_safe_returns_true_for_clean_url():
    """Clean URL (0 malicious/suspicious) returns True."""
    mock_analysis = {
        "data": {"attributes": {"stats": {"malicious": 0, "suspicious": 0}}}
    }
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, return_value="scan123"), \
         patch("services.virustotal_service._poll_result", new_callable=AsyncMock, return_value=mock_analysis):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is True


@pytest.mark.asyncio
async def test_is_safe_returns_false_for_malicious_url():
    """Malicious URL returns False."""
    mock_analysis = {
        "data": {"attributes": {"stats": {"malicious": 3, "suspicious": 0}}}
    }
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, return_value="scan123"), \
         patch("services.virustotal_service._poll_result", new_callable=AsyncMock, return_value=mock_analysis):
        result = await is_safe("https://evil.example.com/bad.gif")
    assert result is False


@pytest.mark.asyncio
async def test_is_safe_returns_false_on_exception():
    """Network error returns False (fail safe)."""
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is False


@pytest.mark.asyncio
async def test_is_safe_returns_false_when_no_api_key():
    """Missing API key returns False immediately."""
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", ""):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is False


@pytest.mark.asyncio
async def test_is_safe_returns_false_on_poll_timeout():
    """Poll timeout (_poll_result returns {}) is treated as unsafe (False)."""
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, return_value="scan123"), \
         patch("services.virustotal_service._poll_result", new_callable=AsyncMock, return_value={}):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is False
