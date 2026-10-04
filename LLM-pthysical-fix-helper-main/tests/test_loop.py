import asyncio
import base64
import json
import io
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from uuid import uuid4

import httpx
from PIL import Image
from pydantic import ValidationError

from src.camera import Camera, save_debug
from src.grok import Grok, GrokError
from src.models import BoundingBox, DiagnosticResponse, ImageFrame, Session, State
from src.operator import Operator
from src.speech import DEFAULT_VOICE_ID, Speech as ElevenLabsSpeech, SpeechError


class WorkspaceTemporaryDirectory:
    """Use ordinary inherited Windows permissions for sandbox test artifacts."""
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.path = self.root / ("test-artifacts-" + uuid4().hex)
        self.path.mkdir()
        self.name = str(self.path)
    def __enter__(self): return self.name
    def __exit__(self, *args): self.cleanup()
    def cleanup(self):
        target = self.path.resolve()
        if target.parent != self.root or not target.name.startswith("test-artifacts-"):
            raise ValueError("Test cleanup path escaped the workspace")
        if target.exists(): shutil.rmtree(target)


def response(target=True, **kwargs):
    fields = dict(speech="Check this connector." if target else "What is the model?",
                  target_present=target, target_label="connector" if target else None,
                  bbox=BoundingBox(x1=10, y1=20, x2=30, y2=40) if target else None,
                  confidence=0.95 if target else None, status="inspect")
    fields.update(kwargs)
    return DiagnosticResponse(**fields)


class Recorder:
    def __init__(self): self.starts = 0
    def start(self): self.starts += 1
    def stop(self): return Path("input.wav")
    def abort(self): pass


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.started, self.complete = asyncio.Event(), asyncio.Event()
    async def capture_and_process(self):
        self.started.set()
        await self.complete.wait()
        return self.frame


class Speech:
    def __init__(self):
        self.transcribed, self.spoken = asyncio.Event(), asyncio.Event()
        self.complete = asyncio.Event()
        self.texts = []
    async def transcribe(self, audio):
        self.transcribed.set()
        return "Why won't it turn on?"
    async def speak(self, text):
        self.texts.append(text)
        self.spoken.set()
        await self.complete.wait()


class Model:
    def __init__(self, result): self.result, self.calls = result, 0
    async def diagnose(self, transcript, frame, session):
        self.calls += 1
        session.previous_response_id = f"response-{self.calls}"
        if isinstance(self.result, Exception): raise self.result
        return self.result


class Pointer:
    def __init__(self):
        self.sent, self.complete = asyncio.Event(), asyncio.Event()
        self.boxes = []
        self.ack = True
        self.cancelled = 0
    async def send_bbox(self, *box):
        self.boxes.append(box)
        self.sent.set()
    async def wait_until_complete(self): await self.complete.wait()
    async def cancel(self):
        self.cancelled += 1
        return self.ack


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = WorkspaceTemporaryDirectory()
        self.path = Path(self.temp.name) / "original.png"
        Image.new("RGB", (100, 80), "white").save(self.path)
        self.camera = FakeCamera(ImageFrame(self.path, 100, 80))
        self.speech, self.pointer, self.recorder = Speech(), Pointer(), Recorder()
        self.model = Model(response())
        self.op = Operator(self.recorder, self.camera, self.speech, self.model,
                           self.pointer, notify=lambda text: None, output_timeout=1,
                           pointer_timeout=0.2)

    async def asyncTearDown(self):
        self.camera.complete.set()
        self.speech.complete.set()
        self.pointer.complete.set()
        await self.op.close()
        self.temp.cleanup()

    async def begin(self):
        self.assertTrue(self.op.space_down())
        await asyncio.wait_for(self.camera.started.wait(), 1)
        self.assertTrue(self.op.space_up())
        self.camera.complete.set()

    async def test_speech_provider_failure_does_not_retry_tts_fallback(self):
        async def fail(text):
            self.speech.texts.append(text)
            raise SpeechError("ElevenLabs speech playback (402): choose another voice")
        self.speech.speak = fail
        self.model.result = response(False)
        self.camera.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(len(self.speech.texts), 1)
        self.assertEqual(self.op.state, State.READY)
        self.assertIn("ElevenLabs", self.op.last_error)
        self.assertEqual(self.pointer.boxes, [])

    async def test_image_starts_during_hold_and_stt_overlaps_processing(self):
        self.op.space_down()
        await asyncio.wait_for(self.camera.started.wait(), 1)
        self.assertEqual(self.op.state, State.RECORDING)
        self.assertFalse(self.op.space_down())
        self.op.space_up()
        await asyncio.wait_for(self.speech.transcribed.wait(), 1)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.recorder.starts, 1)
        self.camera.complete.set()
        self.pointer.complete.set()
        self.speech.complete.set()
        await self.op.wait_until_idle()
        self.assertEqual(self.model.calls, 1)

    async def test_unchanged_box_and_wait_for_both_outputs(self):
        await self.begin()
        await asyncio.wait_for(self.pointer.sent.wait(), 1)
        await asyncio.wait_for(self.speech.spoken.wait(), 1)
        self.assertEqual(self.pointer.boxes, [(10, 20, 30, 40)])
        self.assertFalse(self.op.space_down())
        self.assertFalse(self.op.space_up())
        self.assertFalse(self.op.reset_session())
        self.speech.complete.set()
        await asyncio.sleep(0)
        self.assertEqual(self.op.state, State.WAITING_FOR_POINTER)
        self.pointer.complete.set()
        await self.op.wait_until_idle()
        self.assertEqual(self.op.state, State.READY)
        self.assertTrue(self.op.session.turns[0]["output_complete"])
        self.assertEqual(self.op.session.previous_response_id, "response-1")
        self.assertTrue(self.op.reset_session())
        self.assertIsNone(self.op.session.previous_response_id)
        self.assertEqual(self.op.session.turns, [])

    async def test_pointer_finishes_first_still_waits_for_speech(self):
        await self.begin()
        await asyncio.wait_for(self.pointer.sent.wait(), 1)
        self.pointer.complete.set()
        await asyncio.sleep(0.02)
        self.assertNotEqual(self.op.state, State.READY)
        self.assertFalse(self.op.space_down())
        self.speech.complete.set()
        await self.op.wait_until_idle()
        self.assertEqual(self.op.state, State.READY)

    async def test_no_target_never_calls_pointer(self):
        self.model.result = response(False)
        await self.begin()
        await asyncio.wait_for(self.speech.spoken.wait(), 1)
        self.assertEqual(self.pointer.boxes, [])
        self.assertNotEqual(self.op.state, State.READY)
        self.speech.complete.set()
        await self.op.wait_until_idle()
        self.assertEqual(self.op.state, State.READY)

    async def test_bad_box_and_low_confidence_request_clarification(self):
        for bad in [response(bbox=BoundingBox(x1=10, y1=20, x2=100, y2=40)), response(confidence=0.1)]:
            self.model.result = bad
            self.speech.complete.set()
            await self.begin()
            await self.op.wait_until_idle()
            self.assertFalse(self.op.last_response.target_present)
            self.assertEqual(self.pointer.boxes, [])
            self.assertNotIn("Check this connector.", self.speech.texts)
            self.assertTrue(self.op.session.pending_feedback)
            if bad.confidence < self.op.min_confidence:
                with Image.open(self.op.debug_path) as debug:
                    self.assertEqual(debug.getpixel((10, 30)), (255, 0, 0))

    async def test_malformed_model_response_never_moves(self):
        self.model.result = ValueError("Malformed structured output")
        self.speech.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(self.pointer.boxes, [])
        self.assertIn("Malformed", self.op.last_error)
        self.assertEqual(self.op.state, State.READY)

    async def assert_pointer_failure_carries_on(self):
        events = []
        self.op.on_event = lambda kind, **data: events.append(kind)
        self.speech.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(self.op.state, State.READY)
        self.assertIsNone(self.op.last_error)
        self.assertNotIn("error", events)  # nothing shown in the conversation
        self.assertEqual(self.speech.texts, ["Check this connector."])
        self.assertIn("could not mark connector", self.op.session.pending_feedback[0])

    async def test_pointer_timeout_carries_on(self):
        self.op.pointer_timeout = 0.01
        await self.assert_pointer_failure_carries_on()

    async def test_interrupt_pointer_wait_requires_ack(self):
        await self.begin()
        await asyncio.wait_for(self.pointer.sent.wait(), 1)
        self.pointer.ack = False
        await self.op.interrupt()
        self.assertEqual(self.op.state, State.ERROR)
        self.assertFalse(self.op.space_down())
        self.pointer.ack = True
        await self.op.interrupt()
        self.assertEqual(self.op.state, State.READY)

    async def test_interrupt_drains_capture_before_ready(self):
        self.op.space_down()
        await self.camera.started.wait()
        interruption = asyncio.create_task(self.op.interrupt())
        await asyncio.sleep(0.01)
        self.assertFalse(self.op.space_down())
        self.camera.complete.set()
        await interruption
        self.assertEqual(self.op.state, State.READY)
        self.assertEqual(self.model.calls, 0)

    async def test_stt_failure_drains_capture(self):
        async def fail(audio): raise RuntimeError("STT failed")
        self.speech.transcribe = fail
        self.speech.complete.set()
        self.op.space_down()
        self.op.space_up()
        await asyncio.sleep(0.01)
        self.assertFalse(self.op.space_down())
        self.camera.complete.set()
        await self.op.wait_until_idle()
        self.assertEqual(self.pointer.boxes, [])
        self.assertEqual(self.op.state, State.READY)

    async def test_debug_is_separate_and_fixture_dimensions_are_real(self):
        camera = Camera(Path(self.temp.name) / "artifacts", self.path)
        frame = await camera.capture_and_process()
        clean = frame.path.read_bytes()
        debug = save_debug(frame, response().bbox, "connector")
        self.assertEqual((frame.width, frame.height), (100, 80))
        self.assertNotEqual(frame.path, debug)
        self.assertEqual(frame.path.read_bytes(), clean)
        with Image.open(debug) as preview:
            self.assertEqual(preview.getpixel((10, 30)), (255, 0, 0))

    async def test_camera_failure_and_processing_timeout_never_move(self):
        async def fail(): raise RuntimeError("OpenCV: camera disconnected")
        self.camera.capture_and_process = fail
        self.speech.complete.set()
        self.op.space_down()
        self.op.space_up()
        await self.op.wait_until_idle()
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.pointer.boxes, [])
        self.assertEqual(self.op.last_error, "I couldn't take a photo. Check the camera and try again.")
        self.op.processing_timeout = 0.01
        async def slow(audio): await asyncio.Event().wait()
        self.speech.transcribe = slow
        self.op.stage = 1
        self.op.space_down()
        self.op.space_up()
        await self.op.wait_until_idle()
        self.assertIn("TimeoutError", self.op.last_error)
        self.assertEqual(self.op.state, State.READY)

    async def test_microphone_cleanup_failure_keeps_turns_locked(self):
        def fail(): raise RuntimeError("Microphone close failed")
        self.recorder.abort = fail
        self.speech.complete.set()
        self.pointer.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(self.op.state, State.ERROR)
        self.assertFalse(self.op.space_down())
        self.recorder.abort = lambda: None
        await self.op.interrupt()
        self.assertEqual(self.op.state, State.READY)

    async def test_stages_one_two_only_acquire_inputs(self):
        self.op.stage = 1
        self.op.space_down()
        self.assertIsNone(self.op._image_task)
        self.op.space_up()
        await self.op.wait_until_idle()
        self.op.stage = 2
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.pointer.boxes, [])
        self.assertEqual(self.speech.texts, [])

    async def test_grok_timeout_returns_ready_without_pointing(self):
        async def slow(*args): await asyncio.Event().wait()
        self.model.diagnose = slow
        self.op.grok_timeout = 0.01
        self.speech.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        self.assertEqual(self.pointer.boxes, [])
        self.assertEqual(self.op.state, State.READY)
        self.assertIn("TimeoutError", self.op.last_error)

    async def test_immediate_release_then_cancel_cleans_unstarted_turn(self):
        self.camera.complete.set()
        self.op.space_down()
        self.op.space_up()
        await self.op.interrupt()
        self.assertEqual(self.op.state, State.READY)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.pointer.boxes, [])

    async def test_send_failure_carries_on(self):
        async def fail(*args): raise RuntimeError("Pointer connection lost after send")
        self.pointer.send_bbox = fail
        await self.assert_pointer_failure_carries_on()

    async def test_camera_failure_reuses_the_previous_photo(self):
        self.speech.complete.set()
        self.pointer.complete.set()
        await self.begin()
        await self.op.wait_until_idle()
        async def fail(): raise RuntimeError("OpenCV: camera disconnected")
        self.camera.capture_and_process = fail
        self.op.space_down()
        self.op.space_up()
        await self.op.wait_until_idle()
        self.assertIsNone(self.op.last_error)
        self.assertEqual(self.model.calls, 2)
        self.assertEqual(self.op.last_frame.path, self.path)
        self.assertEqual(self.op.state, State.READY)


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_payment_error_is_readable_and_does_not_play_audio(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(402, json={"detail": {"code": "paid_plan_required", "message": "Free users cannot use library voices via the API."}}, headers={"x-trace-id": "private-trace"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("sounddevice.play") as play:
                with self.assertRaises(SpeechError) as caught:
                    await ElevenLabsSpeech(client, "test-key").speak("Hi")
                self.assertIn("Choose another voice", str(caught.exception))
                self.assertNotIn("private-trace", str(caught.exception))
                self.assertEqual(len(requests), 1)
                play.assert_not_called()

    async def test_voice_list_missing_permission_offers_manual_id(self):
        def handler(request):
            return httpx.Response(401, json={"detail": {"code": "unauthorized", "status": "missing_permissions", "message": "Missing permission"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(SpeechError, "voices_read.*voice ID"):
                await ElevenLabsSpeech(client, "test-key").list_voices()

    async def test_sdk_transcription_uses_requested_settings(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "text": " Check this connector. ", "language_code": "eng",
                "language_probability": 0.99, "words": [],
            })
        audio_file = io.BytesIO(b"recorded audio")
        audio_file.name = "input.wav"
        audio = MagicMock(spec=Path)
        audio.open.return_value = audio_file
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            speech = ElevenLabsSpeech(client, "test-key")
            self.assertEqual(await speech.transcribe(audio), "Check this connector.")
        request = requests[0]
        self.assertEqual(request.url.path, "/v1/speech-to-text")
        self.assertEqual(request.headers["xi-api-key"], "test-key")
        body = request.content.decode().lower()
        self.assertIn('name="model_id"\r\n\r\nscribe_v2', body)
        self.assertIn('name="tag_audio_events"\r\n\r\ntrue', body)
        self.assertIn('name="diarize"\r\n\r\ntrue', body)
        self.assertNotIn('name="language_code"', body)  # SDK omits None for auto detection.
        self.assertTrue(audio_file.closed)

    async def test_sdk_tts_uses_v4_english_and_user_voice(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, content=b"\x00\x00\x01\x00",
                                  headers={"content-type": "application/octet-stream"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            speech = ElevenLabsSpeech(client, "test-key")
            with patch("sounddevice.play") as play, patch("sounddevice.get_stream", return_value=SimpleNamespace(active=False)), patch("sounddevice.stop") as stop:
                await speech.speak("Check this connector. [whispers] Carefully.")
                play.assert_called_once()
                self.assertEqual(play.call_args.kwargs["samplerate"], 16000)
                stop.assert_called_once()
        request = requests[0]
        self.assertEqual(request.url.path, f"/v1/text-to-speech/{DEFAULT_VOICE_ID}")
        self.assertEqual(request.url.params["output_format"], "pcm_16000")
        payload = json.loads(request.content)
        self.assertEqual(payload["model_id"], "eleven_v4")
        self.assertEqual(payload["language_code"], "en")
        self.assertEqual(payload["text"], "Check this connector. [whispers] Carefully.")

    async def test_sdk_empty_transcript_and_malformed_audio_fail(self):
        def handler(request):
            if request.url.path.endswith("speech-to-text"):
                return httpx.Response(200, json={"text": " ", "language_code": "eng",
                                                "language_probability": 0.99, "words": []})
            return httpx.Response(200, content=b"x")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            speech = ElevenLabsSpeech(client, "test-key")
            audio = MagicMock(spec=Path)
            audio.open.return_value = io.BytesIO(b"audio")
            with self.assertRaises(ValueError): await speech.transcribe(audio)
            with patch("sounddevice.play") as play:
                with self.assertRaises(ValueError): await speech.speak("Test")
                play.assert_not_called()

    async def test_tts_cancellation_stops_playback(self):
        def handler(request): return httpx.Response(200, content=b"\x00\x00")
        started = asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            speech = ElevenLabsSpeech(client, "test-key")
            with patch("sounddevice.play", side_effect=lambda *a, **kw: started.set()), patch("sounddevice.get_stream", return_value=SimpleNamespace(active=True)), patch("sounddevice.stop") as stop:
                task = asyncio.create_task(speech.speak("Test"))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                stop.assert_called_once()


class SchemaTests(unittest.TestCase):
    def test_strict_coordinates_and_missing_values(self):
        for x in [True, "10", 10.5, None]:
            with self.assertRaises(ValidationError):
                BoundingBox(x1=x, y1=20, x2=30, y2=40)
        with self.assertRaises(ValidationError): BoundingBox(x1=1, y1=2, x2=3)
        for box in [(1, 2, 1, 3), (-1, 2, 3, 4), (1, 2, 100, 3), (1, 2, 3, 80)]:
            with self.assertRaises(ValueError):
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]).validate_image(100, 80)

    def test_inconsistent_target_and_nan_rejected(self):
        with self.assertRaises(ValidationError): response(False, bbox=BoundingBox(x1=1,y1=2,x2=3,y2=4))
        with self.assertRaises(ValidationError): response(confidence=float("nan"))


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_search_pixels_state_chain_and_reset(self):
        with WorkspaceTemporaryDirectory() as temp:
            path = Path(temp) / "scene.png"
            Image.new("RGB", (200, 150)).save(path)
            frame, session, calls = ImageFrame(path, 200, 150), Session(), []
            def handler(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, json={"id": f"resp-{len(calls)}", "status": "completed", "output": [
                    {"type": "web_search_call", "id": "search-1", "status": "completed"},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": response().model_dump_json()}]},
                ]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                grok = Grok(client, "test-key", state_mode="stored")
                await grok.diagnose("first", frame, session)
                await grok.diagnose("second", frame, session)
                session.reset()
                await grok.diagnose("new problem", frame, session)
            self.assertEqual(calls[0]["model"], "grok-4.7")
            self.assertEqual(calls[0]["tools"], [{"type": "web_search"}])
            self.assertEqual(calls[0]["text"]["format"]["type"], "json_schema")
            self.assertNotIn("previous_response_id", calls[0])
            self.assertEqual(calls[1]["previous_response_id"], "resp-1")
            self.assertIn("instructions", calls[0])
            self.assertNotIn("instructions", calls[1])
            self.assertIn("instructions", calls[2])
            self.assertNotIn("previous_response_id", calls[2])
            blocks = calls[0]["input"][0]["content"]
            self.assertIn("200 x 150", blocks[0]["text"])
            self.assertEqual(base64.b64decode(blocks[1]["image_url"].split(",")[1]), path.read_bytes())

    async def test_client_history_keeps_research_and_rejects_incomplete(self):
        with WorkspaceTemporaryDirectory() as temp:
            path = Path(temp) / "scene.png"
            Image.new("RGB", (100, 80)).save(path)
            frame, session, calls = ImageFrame(path, 100, 80), Session(), []
            output = [{"type": "reasoning", "encrypted_content": "opaque"},
                      {"type": "web_search_call", "id": "search-1", "status": "completed"},
                      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": response(False).model_dump_json()}]}]
            def handler(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, json={"id": "resp-1", "status": "completed" if len(calls) < 3 else "incomplete", "output": output})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                grok = Grok(client, "test-key")
                await grok.diagnose("first", frame, session)
                await grok.diagnose("second", frame, session)
                history_length = len(session.history)
                with self.assertRaises(ValueError): await grok.diagnose("third", frame, session)
            self.assertFalse(calls[1]["store"])
            self.assertEqual(calls[1]["input"][1:4], output)
            self.assertEqual(len(session.history), history_length)

    async def test_provider_error_body_is_visible_without_mutating_history(self):
        with WorkspaceTemporaryDirectory() as temp:
            path = Path(temp) / "scene.png"
            Image.new("RGB", (100, 80)).save(path)
            session = Session()
            def handler(request):
                return httpx.Response(400, json={"code": "400", "error": "Argument not supported: instructions and previous_response_id together"})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(GrokError, "Grok \\(400\\).*instructions"):
                    await Grok(client, "test-key").diagnose("Hi", ImageFrame(path, 100, 80), session)
            self.assertEqual(session.history, [])


if __name__ == "__main__": unittest.main()
