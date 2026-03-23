import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from services.image_search_service import search_giphy, search_local, find_verified_image


@pytest.mark.asyncio
async def test_search_giphy_returns_urls():
    """Giphy search returns list of gif URLs."""
    mock_response = {
        "data": [
            {"images": {"original": {"url": "https://media.giphy.com/media/abc/giphy.gif"}}},
            {"images": {"original": {"url": "https://media.giphy.com/media/def/giphy.gif"}}},
        ]
    }
    with patch("services.image_search_service.GIPHY_API_KEY", "fake_key"), \
         patch("services.image_search_service._giphy_get", new_callable=AsyncMock, return_value=mock_response):
        result = await search_giphy("cat memes")
    assert len(result) == 2
    assert result[0] == "https://media.giphy.com/media/abc/giphy.gif"


@pytest.mark.asyncio
async def test_search_giphy_returns_empty_on_error():
    """Giphy error returns empty list."""
    with patch("services.image_search_service.GIPHY_API_KEY", "fake_key"), \
         patch("services.image_search_service._giphy_get", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await search_giphy("cat memes")
    assert result == []


def test_search_local_matches_by_keyword(tmp_path):
    """Local search finds files whose name contains the keyword."""
    (tmp_path / "cat_funny.gif").touch()
    (tmp_path / "dog_photo.jpg").touch()
    (tmp_path / "cat_meme.png").touch()

    with patch("services.image_search_service.LOCAL_IMAGE_DIR", str(tmp_path)):
        result = search_local("cat")
    assert len(result) == 2
    assert all("cat" in Path(p).name for p in result)


def test_search_local_returns_empty_when_no_dir():
    """Local search returns empty list when LOCAL_IMAGE_DIR is not set."""
    with patch("services.image_search_service.LOCAL_IMAGE_DIR", ""):
        result = search_local("cat")
    assert result == []


@pytest.mark.asyncio
async def test_find_verified_image_returns_first_safe():
    """find_verified_image returns first URL that passes VT scan."""
    candidates = ["https://media.giphy.com/media/abc/giphy.gif", "https://media.giphy.com/media/def/giphy.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, return_value=True):
        result = await find_verified_image("cat memes")
    assert result == candidates[0]


@pytest.mark.asyncio
async def test_find_verified_image_skips_unsafe():
    """find_verified_image skips unsafe URLs and returns next safe one."""
    candidates = ["https://evil.example.com/bad.gif", "https://media.giphy.com/media/safe/giphy.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, side_effect=[False, True]):
        result = await find_verified_image("cat")
    assert result == "https://media.giphy.com/media/safe/giphy.gif"


@pytest.mark.asyncio
async def test_find_verified_image_returns_none_when_all_unsafe():
    """Returns None when all candidates fail VT scan."""
    candidates = ["https://bad1.com/a.gif", "https://bad2.com/b.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, return_value=False):
        result = await find_verified_image("cat")
    assert result is None


@pytest.mark.asyncio
async def test_find_verified_image_returns_local_when_giphy_empty(tmp_path):
    """When Giphy returns nothing, local file fallback is returned."""
    local_file = tmp_path / "cat_funny.gif"
    local_file.touch()
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=[]), \
         patch("services.image_search_service.search_local", return_value=[str(local_file)]):
        result = await find_verified_image("cat")
    assert result == str(local_file)
