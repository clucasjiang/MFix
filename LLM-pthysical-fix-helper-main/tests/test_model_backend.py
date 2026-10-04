import base64
import copy
import json
import unittest
from pathlib import Path

import httpx
from PIL import Image

from src.grok import Grok, OpenAI, ModelError
from src.model_backend import create_model
from src.models import ImageFrame, Session
from test_loop import WorkspaceTemporaryDirectory, response


class SelectionTests(unittest.TestCase):
    def test_user_chat_gpt_key_selects_openai_and_explicit_grok_remains_available(self):
        values = {"CHAT_GPT_KEY": "test-openai", "XAI_API_KEY": "test-grok"}
        model = create_model(None, values)
        self.assertEqual((model.provider, model.api_key, model.model, model.reasoning_effort),
                         ("openai", "test-openai", "gpt-6.1-sol", "low"))
        model = create_model(None, values, provider="grok")
        self.assertEqual((model.provider, model.api_key, model.reasoning_effort), ("grok", "test-grok", "medium"))

    def test_provider_keys_are_separate_and_cli_options_override_config(self):
        values = {"CHAT_GPT_KEY": "alias", "OPENAI_API_KEY": "primary", "OPENAI_MODEL": "configured",
                  "OPENAI_REASONING_EFFORT": "high", "OPENAI_SEARCH": "auto"}
        model = create_model(None, values, provider="openai", model="gpt-4.1", reasoning_effort="auto", search="off")
        self.assertEqual((model.api_key, model.model, model.reasoning_effort, model.search),
                         ("primary", "gpt-4.1", "auto", "off"))
        self.assertEqual(create_model(None, values, provider="grok").api_key, "")
        values["OPENAI_API_KEY"] = "  "
        self.assertEqual(create_model(None, values).api_key, "alias")

    def test_invalid_modes_fail_before_any_request(self):
        with self.assertRaises(ValueError): create_model(None, {}, provider="unknown")
        with self.assertRaises(ValueError): Grok(None, "key", search="unbounded")
        with self.assertRaises(ValueError): OpenAI(None, "key", reasoning_effort="none")


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = WorkspaceTemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "current.png"
        Image.new("RGB", (160, 120), "blue").save(self.path)
        self.frame = ImageFrame(self.path, 160, 120)

    def payload(self, outputs=None):
        return {"id": "response-test", "status": "completed", "output": outputs or [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": response(False).model_dump_json()}]}],
            "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80},
                      "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 10}}}

    async def test_openai_request_uses_own_endpoint_key_schema_and_low_reasoning(self):
        calls = []
        def handler(request):
            self.assertEqual(request.url.host, "api.openai.com")
            self.assertEqual(request.headers["Authorization"], "Bearer test-openai")
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=self.payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAI(client, "test-openai", search="off")
            session = Session()
            await model.diagnose("What is this?", self.frame, session)
        body = calls[0]
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["text"]["format"]["schema"]["additionalProperties"], False)
        self.assertFalse(body["store"])
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(body["prompt_cache_key"], session.session_id)
        self.assertNotIn("tools", body)
        self.assertEqual(body["tool_choice"], "none")
        image = body["input"][-1]["content"][1]
        self.assertEqual(base64.b64decode(image["image_url"].split(",")[1]), self.path.read_bytes())
        self.assertEqual(model.last_metrics["cached_tokens"], 80)
        self.assertEqual(model.last_metrics["reasoning_tokens"], 10)
        self.assertNotIn("test-openai", json.dumps(model.last_metrics))
        self.assertEqual(session.model_identity, ("openai", "gpt-6.1-sol"))

    async def test_latest_photo_keeps_text_and_encrypted_outputs_without_mutating_saved_history(self):
        outputs = [{"type": "reasoning", "encrypted_content": "keep-exact"},
                   {"type": "web_search_call", "id": "search", "status": "completed"},
                   *self.payload()["output"]]
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=self.payload(outputs))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAI(client, "key")
            session = Session()
            await model.diagnose("My reading was 5 volts", self.frame, session)
            original = copy.deepcopy(session.history)
            await model.diagnose("What next?", self.frame, session)
        self.assertEqual(session.history[:len(original)], original)
        body = calls[1]
        self.assertEqual(body["input"][1:4], outputs)
        self.assertIn("My reading was 5 volts", body["input"][0]["content"][0]["text"])
        images = [block for item in body["input"] if item.get("role") == "user"
                  for block in item["content"] if block.get("type") == "input_image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(body["max_tool_calls"], 1)

    async def test_full_photo_history_option_and_nonreasoning_model(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=self.payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAI(client, "key", model="gpt-4.1", history_images="all")
            session = Session()
            for text in ("first", "second"):
                await model.diagnose(text, self.frame, session)
        self.assertNotIn("reasoning", calls[0])
        images = [block for item in calls[1]["input"] if item.get("role") == "user"
                  for block in item["content"] if block.get("type") == "input_image"]
        self.assertEqual(len(images), 2)

    async def test_openai_stored_turns_resend_instructions(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=self.payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAI(client, "key", state_mode="stored")
            session = Session()
            await model.diagnose("first", self.frame, session)
            await model.diagnose("second", self.frame, session)
        self.assertEqual(calls[1]["previous_response_id"], "response-test")
        self.assertIn("instructions", calls[1])
        self.assertNotIn("include", calls[1])

    async def test_cross_provider_state_is_rejected_before_sending_and_reset_restores_access(self):
        calls = []
        def handler(request):
            calls.append(request.url.host)
            return httpx.Response(200, json=self.payload())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = Session()
            await Grok(client, "grok-key").diagnose("first", self.frame, session)
            with self.assertRaises(ValueError):
                await OpenAI(client, "openai-key").diagnose("second", self.frame, session)
            self.assertEqual(calls, ["api.x.ai"])
            session.reset()
            await OpenAI(client, "openai-key").diagnose("new", self.frame, session)
        self.assertEqual(calls, ["api.x.ai", "api.openai.com"])

    async def test_auth_failure_does_not_echo_key_or_retry_or_mutate_session(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, json={"error": {"message": "secret-key rejected"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = Session()
            with self.assertRaises(ModelError) as caught:
                await OpenAI(client, "secret-key").diagnose("first", self.frame, session)
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertIn("OpenAI (401)", str(caught.exception))
        self.assertEqual(len(calls), 1)
        self.assertEqual(session.history, [])
        self.assertIsNone(session.model_identity)


if __name__ == "__main__":
    unittest.main()
