import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_enhance_prompt_returns_string():
    from services.llm_service import LLMService
    svc = LLMService()
    svc.generate_text = AsyncMock(return_value="a majestic wolf in neon cyberpunk city, detailed, 8k")
    result = await svc.enhance_image_prompt("wolf city")
    assert isinstance(result, str)
    assert len(result) > len("wolf city")

@pytest.mark.asyncio
async def test_enhance_prompt_falls_back_on_error():
    from services.llm_service import LLMService
    svc = LLMService()
    svc.generate_text = AsyncMock(side_effect=Exception("LLM down"))
    result = await svc.enhance_image_prompt("wolf city")
    assert result == "wolf city"
