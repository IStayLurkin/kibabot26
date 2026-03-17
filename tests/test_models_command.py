import pytest
from unittest.mock import patch, MagicMock

def test_get_ollama_running_models_returns_list():
    from services.hardware_service import HardwareService
    svc = HardwareService()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = b'{"models": [{"name": "kiba:latest", "size": 9000000000}]}'
    with patch('urllib.request.urlopen', return_value=mock_resp):
        result = svc.get_ollama_running_models()
    assert isinstance(result, list)
    assert result[0]["name"] == "kiba:latest"

def test_get_ollama_running_models_returns_empty_on_error():
    from services.hardware_service import HardwareService
    svc = HardwareService()
    with patch('urllib.request.urlopen', side_effect=Exception("connection refused")):
        result = svc.get_ollama_running_models()
    assert result == []
