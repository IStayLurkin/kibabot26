import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from services.animatediff_service import AnimateDiffService


@pytest.fixture
def service():
    return AnimateDiffService()


def test_service_initializes_with_no_pipeline(service):
    assert service.pipeline is None


def test_purge_vram_clears_pipeline(service):
    service.pipeline = MagicMock()
    with patch("torch.cuda.empty_cache"), patch("torch.cuda.ipc_collect"), patch("gc.collect"):
        service._purge_vram()
    assert service.pipeline is None


@pytest.mark.asyncio
async def test_generate_returns_none_on_exception(service):
    with patch.object(service, "_purge_vram"), \
         patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(side_effect=RuntimeError("OOM"))
        result = await service.generate("a dog running", None)
    assert result is None
