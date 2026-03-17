import unittest
from unittest.mock import MagicMock, patch
from services.llm_service import LLMService


class ProviderChainTests(unittest.TestCase):
    def _make_service(self, provider="ollama"):
        with patch("services.llm_service.LLM_PROVIDER", provider):
            svc = LLMService()
        svc.provider = provider
        return svc

    def test_default_chain_is_ollama_first(self):
        svc = self._make_service("ollama")
        chain = svc._build_provider_chain()
        self.assertEqual(chain[0], "ollama")
        self.assertNotIn("openai", chain)

    def test_hf_primary_chain(self):
        svc = self._make_service("hf")
        chain = svc._build_provider_chain()
        self.assertEqual(chain[0], "hf")
        self.assertNotIn("openai", chain)

    def test_runtime_service_overrides_chain(self):
        svc = self._make_service("ollama")
        mock_runtime = MagicMock()
        mock_runtime.get_active_llm_provider.return_value = "hf"
        svc.model_runtime_service = mock_runtime
        chain = svc._build_provider_chain()
        self.assertEqual(chain[0], "hf")
        self.assertNotIn("openai", chain)

    def test_openai_provider_raises(self):
        svc = self._make_service("ollama")
        with self.assertRaises(RuntimeError):
            svc._get_client_for_provider("openai")

    def test_fallback_uses_correct_model_per_provider(self):
        with patch("services.llm_service.OLLAMA_MODEL", "kiba:latest"), \
             patch("services.llm_service.HF_MODEL", "dolphin-llama3"):
            svc = self._make_service("ollama")
            svc.model_runtime_service = None
            self.assertEqual(svc._get_model_for_provider("ollama"), "kiba:latest")
            self.assertEqual(svc._get_model_for_provider("hf"), "dolphin-llama3")

    def test_active_provider_model_comes_from_runtime(self):
        svc = self._make_service("ollama")
        mock_runtime = MagicMock()
        mock_runtime.get_active_llm_provider.return_value = "ollama"
        mock_runtime.get_active_llm_model.return_value = "kiba:latest"
        svc.model_runtime_service = mock_runtime
        self.assertEqual(svc._get_model_for_provider("ollama"), "kiba:latest")

    def test_fallback_provider_does_not_use_runtime_model(self):
        """When falling back to hf, should use HF_MODEL not the ollama active model."""
        with patch("services.llm_service.HF_MODEL", "dolphin-llama3"):
            svc = self._make_service("ollama")
            mock_runtime = MagicMock()
            mock_runtime.get_active_llm_provider.return_value = "ollama"
            mock_runtime.get_active_llm_model.return_value = "kiba:latest"
            svc.model_runtime_service = mock_runtime
            # hf is not the active provider, so should fall through to HF_MODEL
            self.assertEqual(svc._get_model_for_provider("hf"), "dolphin-llama3")

    def test_generate_reply_falls_back_on_ollama_failure(self):
        """If ollama raises, the chain should attempt hf before raising RuntimeError."""
        with patch("services.llm_service.OLLAMA_MODEL", "kiba:latest"), \
             patch("services.llm_service.HF_MODEL", "dolphin-llama3"):
            svc = self._make_service("ollama")
            svc.model_runtime_service = None

            call_log = []

            def fake_create(provider, *, model, messages, temperature, max_tokens):
                call_log.append(provider)
                if provider == "ollama":
                    raise ConnectionError("ollama not running")
                raise ConnectionError("hf also down")

            svc._create_chat_completion = fake_create

            with self.assertRaises(RuntimeError) as ctx:
                svc._generate_reply_sync("user", "hello", {}, [])

            self.assertIn("ollama", call_log)
            self.assertIn("hf", call_log)
            self.assertIn("All LLM providers failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
