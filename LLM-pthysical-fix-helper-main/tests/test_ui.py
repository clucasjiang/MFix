import asyncio
import json
import tkinter as tk
import unittest
import httpx
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.camera import Camera
from src.camera_capture import CAMERA_NAME, take_photo
from src.devices import CameraDevice, MicrophoneDevice, list_microphones, parse_cameras
from src.models import State
from src.grok import Grok
from src.speech import Speech as LiveSpeech
from src.credentials import save_settings
from src.operator import Operator
from src.ui import DesktopApp
from test_loop import FakeCamera, Model, Pointer, Recorder, Speech, WorkspaceTemporaryDirectory, response


class DeviceTests(unittest.TestCase):
    def test_camera_discovery_keeps_unique_video_paths_and_ignores_audio(self):
        log = '''[dshow] "USB Camera" (video)
[dshow] Alternative name "@device_pnp_camera1"
[dshow] "Microphone" (audio)
[dshow] Alternative name "@device_cm_audio1"
[dshow] "USB Camera" (video)
[dshow] Alternative name "@device_pnp_camera2"'''
        self.assertEqual(parse_cameras(log), [CameraDevice("USB Camera", "@device_pnp_camera1"),
                                              CameraDevice("USB Camera", "@device_pnp_camera2")])

    def test_microphone_list_keeps_input_indices_and_host_api(self):
        devices = [{"name": "Output only", "hostapi": 0, "max_input_channels": 0},
                   {"name": "USB mic", "hostapi": 1, "max_input_channels": 1}]
        with patch("sounddevice.query_devices", return_value=devices), patch("sounddevice.query_hostapis", return_value=[{"name": "MME"}, {"name": "WASAPI"}]):
            self.assertEqual(list_microphones(), [MicrophoneDevice(1, "USB mic", "WASAPI")])

    def test_selected_camera_reaches_capture_without_forcing_generic_format(self):
        with WorkspaceTemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = []
            @contextmanager
            def staging(**kwargs):
                path = root / "staging"
                path.mkdir()
                yield str(path)
            def capture(command, **kwargs):
                commands.append(command)
                Image.new("RGB", (96, 64), "white").save(command[-1])
            with patch("src.camera_capture.TemporaryDirectory", staging), patch("src.camera_capture.subprocess.run", side_effect=capture):
                result = take_photo(root, camera_name="@device_pnp_cam", camera_label="Built-in camera", resolution=None)
                self.assertIn("video=@device_pnp_cam", commands[-1])
                self.assertNotIn("-video_size", commands[-1])
                self.assertNotIn("mjpeg", commands[-1])
                self.assertEqual((result["image_width"], result["image_height"]), (96, 64))
                self.assertTrue(Path(result["original_path"]).is_file())
                self.assertFalse(Path(result["original_path"]).with_name("grid.png").exists())
                take_photo(root, camera_name="@device_pnp_ugreen", camera_label=CAMERA_NAME)
                self.assertIn("mjpeg", commands[-1])
                self.assertIn("1920x1080", commands[-1])


class UiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.temp = WorkspaceTemporaryDirectory()
        self.path = Path(self.temp.name) / "capture.png"
        Image.new("RGB", (160, 120), "white").save(self.path)
        self.camera = Camera(Path(self.temp.name), self.path)
        self.recorder, self.speech, self.pointer = Recorder(), Speech(), Pointer()
        self.recorder.device = None
        self.op = Operator(self.recorder, self.camera, self.speech, Model(response(False)), self.pointer,
                           notify=lambda text: None)
        self.app = DesktopApp(self.op, root=self.root)
        self.op.notify = lambda text: None

    async def asyncTearDown(self):
        self.speech.complete.set()
        self.pointer.complete.set()
        await asyncio.gather(*self.app._tasks, return_exceptions=True)
        await self.op.close()
        self.root.destroy()
        self.temp.cleanup()

    async def test_switching_to_openai_starts_new_session_and_changes_speaker_label(self):
        async with httpx.AsyncClient() as client:
            self.op.grok = Grok(client, "grok-key")
            old_session = self.op.session.session_id
            self.op.session.history.append({"old": "provider state"})
            self.app._apply_model_settings({"CHAT_GPT_KEY": "openai-key"}, provider="openai",
                                          model="gpt-6.1-sol", effort="low", search="off")
            self.assertNotEqual(self.op.session.session_id, old_session)
            self.assertEqual(self.op.session.history, [])
            self.assertEqual(self.op.grok.provider, "openai")
            self.assertIn("OpenAI", self.app.model_text.get())
            self.app.on_event("response", response=response(False))
            self.assertIn("OPENAI", self.app.conversation.get("1.0", "end"))

    async def test_model_switch_while_busy_is_rejected_and_timings_are_visible(self):
        self.op.state = State.OUTPUTTING
        with self.assertRaises(ValueError):
            self.app._apply_model_settings({}, provider="openai", model="gpt-6.1-sol", effort="low", search="off")
        self.app.on_event("timings", timings={"stt_seconds": 1, "model_seconds": 3, "output_seconds": 2,
                                              "total_after_release_seconds": 6})
        self.assertIn("Model 3.0s", self.app.timing_text.get())

    async def test_settings_save_sends_model_and_voice_keys_to_secure_storage(self):
        from test_credentials import MemoryStore
        store = MemoryStore()
        path = Path(self.temp.name) / ".env"
        async with httpx.AsyncClient() as client:
            self.op.grok = Grok(client, "dummy-grok")
            self.op.speech = LiveSpeech(client, "dummy-old-voice", device=3)
            self.op.speech._sdk = object()  # The changed key must discard the old SDK.
            self.app.show_model_settings()
            dialog = next(widget for widget in self.root.winfo_children() if isinstance(widget, tk.Toplevel))
            form = dialog.winfo_children()[0]
            entries = [widget for widget in form.winfo_children() if isinstance(widget, tk.ttk.Entry)
                       and not isinstance(widget, tk.ttk.Combobox)]
            entries[0].insert(0, "dummy-new-model")
            entries[1].insert(0, "dummy-new-voice")
            save = next(widget for widget in form.winfo_children() if isinstance(widget, tk.ttk.Button))
            def secure_save(unused_path, changes):
                return save_settings(path, changes, store=store, environ={})
            with patch("src.ui.load_settings", return_value={"XAI_API_KEY": "dummy-grok"}), \
                    patch("src.ui.save_settings", side_effect=secure_save):
                save.invoke()
            self.assertFalse(dialog.winfo_exists())
            self.assertEqual(self.op.grok.api_key, "dummy-new-model")
            self.assertEqual(self.op.speech.api_key, "dummy-new-voice")
            self.assertIsNone(self.op.speech._sdk)
            self.assertEqual(self.op.speech.device, 3)
            self.assertNotIn("dummy", path.read_text())

    async def test_grok_message_transcript_and_debug_image_visible(self):
        self.app.on_event("transcript", text="Which connector should I check?")
        self.app.on_event("response", response=response())
        self.app.on_event("image", frame=SimpleNamespace(path=self.path, width=160, height=120))
        self.assertIn("Check this connector.", self.app.conversation.get("1.0", "end"))
        self.assertIn("Which connector", self.app.conversation.get("1.0", "end"))
        self.assertIn("connector", self.app.target_text.get())
        self.assertIn("160 × 120", self.app.image_note.get())
        self.assertEqual(self.app._source_image.size, (160, 120))

    async def test_devices_apply_only_when_ready(self):
        self.camera.fixture = None
        self.app._mic_choices["USB mic"] = 2
        self.app._camera_choices["Laptop camera"] = CameraDevice("Laptop camera", "@device_camera")
        self.app.mic_text.set("USB mic")
        self.app._choose_mic()
        self.assertEqual(self.recorder.device, 2)
        self.app.camera_text.set("Laptop camera")
        self.app._choose_camera()
        self.assertEqual(self.camera.camera_name, "@device_camera")
        self.assertIsNone(self.camera.resolution)
        self.op._state(State.PROCESSING)
        self.app._mic_choices["Other mic"] = 3
        self.app.mic_text.set("Other mic")
        self.app._choose_mic()
        self.assertEqual(self.recorder.device, 2)
        self.assertEqual(str(self.app.mic_combo.cget("state")), "disabled")
        self.assertEqual(str(self.app.camera_combo.cget("state")), "disabled")

    async def test_voice_selection_and_manual_id_apply_only_when_ready(self):
        self.app._voice_choices = {"My voice [voice123]": "voice123"}
        self.app.voice_text.set("My voice [voice123]")
        self.app._choose_voice()
        self.assertEqual(self.speech.voice_id, "voice123")
        self.app.voice_text.set("manual456")
        self.app._choose_voice()
        self.assertEqual(self.speech.voice_id, "manual456")
        self.op._state(State.PROCESSING)
        self.app.voice_text.set("blocked789")
        self.app._choose_voice()
        self.assertEqual(self.speech.voice_id, "manual456")

    async def test_native_popup_focus_does_not_resolve_python_widget_or_interrupt(self):
        with patch.object(self.root, "focus_displayof", side_effect=KeyError("popdown")) as lookup:
            self.app._focus_out(None)
            lookup.assert_not_called()
        self.op._state(State.RECORDING)
        native_root = SimpleNamespace(tk=SimpleNamespace(call=lambda *args: ".camera.popdown"), _w=".")
        with patch.object(self.app, "root", native_root), patch.object(self.app, "interrupt") as interrupt:
            self.app._check_recording_focus()
            interrupt.assert_not_called()
        native_root.tk.call = lambda *args: ""
        with patch.object(self.app, "root", native_root), patch.object(self.app, "interrupt") as interrupt:
            self.app._check_recording_focus()
            interrupt.assert_called_once()

    async def test_refresh_handles_partial_failure_and_preserves_selection(self):
        self.camera.fixture = None
        self.recorder.device = 7
        with patch("src.ui.list_microphones", return_value=[MicrophoneDevice(7, "USB mic", "WASAPI")]), patch("src.ui.list_cameras", return_value=[CameraDevice(CAMERA_NAME, "@device_ugreen")]):
            await self.app._discover()
        self.assertEqual(self.app._mic_choices[self.app.mic_text.get()], 7)
        self.assertEqual(self.app.camera_text.get(), CAMERA_NAME)
        with patch("src.ui.list_microphones", side_effect=RuntimeError("Audio service unavailable")), patch("src.ui.list_cameras", return_value=[]), patch("builtins.print"):
            await self.app._discover()
        self.assertEqual(self.recorder.device, 7)
        self.assertIn("Microphone discovery failed", self.app.device_note.get())
        self.assertIn("unavailable", self.app.camera_text.get())

    async def test_mouse_and_keyboard_share_one_turn_admission(self):
        self.app._key_down()
        self.app._key_down()
        self.app._mouse_down()
        self.assertEqual(self.recorder.starts, 1)
        self.assertEqual(self.op.state, State.RECORDING)
        self.app._key_up()
        self.speech.complete.set()
        await self.op.wait_until_idle()

    async def test_ignored_busy_press_cannot_start_on_key_repeat_after_ready(self):
        self.op._state(State.PROCESSING)
        self.app._key_down()
        self.op._state(State.READY)
        self.app._key_down()
        self.assertEqual(self.recorder.starts, 0)
        self.app._key_up()
        self.app._key_down()
        self.assertEqual(self.recorder.starts, 1)
        await self.op.interrupt()

    async def test_session_reset_clears_conversation_and_is_blocked_while_busy(self):
        self.app.on_event("response", response=response())
        self.op._state(State.PROCESSING)
        self.app.new_session()
        self.assertIn("Check this connector", self.app.conversation.get("1.0", "end"))
        self.op._state(State.READY)
        self.app.new_session()
        self.assertNotIn("Check this connector", self.app.conversation.get("1.0", "end"))
        self.assertIn("New diagnostic session", self.app.conversation.get("1.0", "end"))
