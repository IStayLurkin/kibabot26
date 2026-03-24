import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from services.cogvideo_service import CogVideoService


@pytest.fixture
def service():
    return CogVideoService()


def test_service_initializes_with_no_pipeline(service):
    assert service.pipeline is None
    assert service.current_model is None


def test_get_vram_usage_returns_int(service):
    with patch.object(service, "_hardware") as mock_hw:
        mock_hw.get_vram_usage_mb.return_value = 12000
        assert service._get_vram_usage() == 12000


def test_purge_vram_clears_pipeline(service):
    service.pipeline = MagicMock()
    service.current_model = "2b"
    with patch("torch.cuda.empty_cache"), patch("torch.cuda.ipc_collect"), patch("gc.collect"):
        service._purge_vram()
    assert service.pipeline is None


@pytest.mark.asyncio
async def test_generate_returns_none_on_exception(service):
    with patch.object(service, "_purge_vram"), \
         patch("asyncio.get_running_loop") as mock_loop:
        mock_executor = AsyncMock(side_effect=RuntimeError("CUDA OOM"))
        mock_loop.return_value.run_in_executor = mock_executor
        result = await service.generate("2b", "a cat running", None)
    assert result is None
