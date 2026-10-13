import unittest
import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import chatbot


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "local-a": {
                "key": "local-a", "name": "Local A", "provider": "local",
                "note": "local", "downloaded": False, "file": None,
                "available": False, "unavailable_reason": "모델 파일 미다운로드",
            },
            "api:sample:model-a": {
                "key": "api:sample:model-a", "name": "API A", "provider": "api",
                "note": "api", "downloaded": False, "file": None,
                "available": True, "unavailable_reason": None,
                "base_url": "https://example.com/v1", "api_key": "sk-test-secret",
                "model_id": "model-a", "enable_thinking": None,
            },
        }

    def test_default_skips_unavailable_model(self):
        with patch.object(chatbot, "DEFAULT_MODEL", "local-a"):
            self.assertEqual(chatbot.get_default_model_key(self.catalog), "api:sample:model-a")
        with patch.object(chatbot, "DEFAULT_MODEL", "api:sample:model-a"):
            self.assertEqual(chatbot.get_default_model_key(self.catalog), "api:sample:model-a")

    def test_explicit_unavailable_selection_does_not_fallback(self):
        with self.assertRaises(chatbot.HTTPException) as raised:
            chatbot.select_model("local-a", self.catalog)
        self.assertEqual(raised.exception.status_code, 400)

    def test_local_loader_does_not_switch_models(self):
        with patch.object(chatbot, "LLAMA_OK", True), patch.object(chatbot, "find_gguf_file", return_value=None):
            with self.assertRaises(FileNotFoundError):
                chatbot.get_llm("qwen3-8b")

    def test_status_hides_server_api_configuration(self):
        with patch.object(chatbot, "get_model_catalog", return_value=self.catalog):
            response = TestClient(chatbot.app).get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["default_model"], "api:sample:model-a")
        self.assertNotIn("sk-test-secret", response.text)
        self.assertNotIn("api_key", response.text)
        self.assertNotIn("base_url", response.text)

    def test_chat_rejects_unavailable_selection_before_streaming(self):
        with patch.object(chatbot, "get_model_catalog", return_value=self.catalog):
            response = TestClient(chatbot.app).post("/chat", json={"message": "hello", "model": "local-a"})
        self.assertEqual(response.status_code, 400)

    def test_chat_uses_catalog_default_without_external_request(self):
        with patch.object(chatbot, "get_model_catalog", return_value=self.catalog), \
             patch.object(chatbot, "stream_api_completion", return_value=iter(["hello"])) as stream:
            response = TestClient(chatbot.app).post("/chat", json={"message": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('"token": "hello"', response.text)
        self.assertIn('"provider": "api"', response.text)
        self.assertIn('"model_name": "API A"', response.text)
        self.assertEqual(stream.call_args.args[1]["key"], "api:sample:model-a")
        self.assertIn("model-a", stream.call_args.args[0][0]["content"])

    def test_prompt_follows_question_language_and_names_configured_model(self):
        messages = chatbot.build_messages("你是什么模型", [], model_id="qwen3.8-max")
        system = messages[0]["content"]
        self.assertIn("같은 언어", system)
        self.assertIn("qwen3.8-max", system)
        self.assertNotIn("한국어 챗봇", system)

    def test_explicit_chat_language_overrides_question_language(self):
        chinese = chatbot.build_messages("안녕하세요", [], language="zh")[0]["content"]
        korean = chatbot.build_messages("你好", [], language="ko")[0]["content"]
        self.assertIn("请用简体中文回答", chinese)
        self.assertIn("한국어로 답변하세요", korean)

    def test_chat_passes_selected_language_to_model(self):
        with patch.object(chatbot, "get_model_catalog", return_value=self.catalog), \
             patch.object(chatbot, "stream_api_completion", return_value=iter(["안녕하세요"])) as stream:
            response = TestClient(chatbot.app).post(
                "/chat", json={"message": "你好", "model": "api:sample:model-a", "language": "ko"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("한국어로 답변하세요", stream.call_args.args[0][0]["content"])

    def test_chat_rejects_unknown_language(self):
        response = TestClient(chatbot.app).post(
            "/chat", json={"message": "hello", "language": "en"}
        )
        self.assertEqual(response.status_code, 422)

    def test_auth_error_is_actionable_without_exposing_key(self):
        key = "sk-test-secret"
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": {"message": "Invalid API-key provided"}})
        ))
        with patch.object(chatbot.httpx, "Client", return_value=client):
            with self.assertRaises(RuntimeError) as raised:
                list(chatbot.stream_api_completion([{"role": "user", "content": "test"}], self.catalog["api:sample:model-a"]))
        self.assertIn("401", str(raised.exception))
        self.assertNotIn(key, str(raised.exception))

    def test_verify_api_model_uses_short_non_thinking_probe(self):
        model = {**self.catalog["api:sample:model-a"], "enable_thinking": True}
        with patch.object(chatbot, "stream_api_completion", return_value=iter(["OK"])) as stream:
            self.assertTrue(chatbot.verify_api_model(model))
        self.assertEqual(stream.call_args.args[1]["enable_thinking"], False)
        self.assertTrue(model["enable_thinking"])

    def test_verify_endpoint_reports_result_without_exposing_key(self):
        with patch.object(chatbot, "get_model_catalog", return_value=self.catalog), \
             patch.object(chatbot, "verify_api_model", return_value=True) as probe:
            response = TestClient(chatbot.app).post(
                "/verify_model", json={"model": "api:sample:model-a"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "api:sample:model-a")
        probe.assert_called_once()

    def test_api_key_must_be_a_string(self):
        provider = [{
            "id": "qianwen", "name": "Qwen",
            "base_url": "https://example.com/v1",
            "api_key": 123,
            "models": ["test-model"],
        }]
        with patch.object(chatbot, "API_MODELS_FILE") as config_file:
            config_file.exists.return_value = True
            config_file.read_text.return_value = json.dumps(provider)
            with self.assertRaises(ValueError) as raised:
                chatbot.get_api_models()
        error = str(raised.exception)
        self.assertIn("api_key", error)

    def test_provider_key_is_read_directly_and_private(self):
        provider = [{
            "id": "sample", "name": "Sample", "base_url": "https://example.com/v1",
            "api_key": "sk-test-private",
            "models": ["model-a"],
        }]
        with patch.object(chatbot, "API_MODELS_FILE") as config_file, \
             patch.dict("os.environ", {"TEST_FICL_API_KEY": "sk-env-ignored", "MODEL_API_KEY": "sk-env-ignored"}):
            config_file.exists.return_value = True
            config_file.read_text.return_value = json.dumps(provider)
            model = chatbot.get_api_models()["api:sample:model-a"]
        self.assertTrue(model["available"])
        self.assertEqual(model["api_key"], "sk-test-private")
        self.assertNotIn("api_key_env", model)
        self.assertNotIn("sk-test-private", chatbot.public_model_info({
            **model, "note": "api", "downloaded": False, "file": None,
            "unavailable_reason": None,
        }).values())

        seen = {}
        def respond(request):
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch.object(chatbot.httpx, "Client", return_value=client):
            result = list(chatbot.stream_api_completion([{"role": "user", "content": "test"}], model))
        self.assertEqual(result, ["OK"])
        self.assertEqual(seen["authorization"], "Bearer sk-test-private")

    def test_legacy_api_environment_does_not_create_models(self):
        with patch.object(chatbot, "API_MODELS_FILE") as config_file, \
             patch.dict("os.environ", {"MODEL_API_MODELS": "legacy-model", "MODEL_API_KEY": "sk-legacy"}):
            config_file.exists.return_value = False
            self.assertEqual(chatbot.get_api_models(), {})


if __name__ == "__main__":
    unittest.main()
