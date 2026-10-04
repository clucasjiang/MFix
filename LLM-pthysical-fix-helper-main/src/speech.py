"""Space-hold PCM recording and the existing ElevenLabs STT/TTS provider."""
import asyncio
import threading
import wave
from pathlib import Path
from uuid import uuid4

import httpx
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.core.api_error import ApiError

DEFAULT_VOICE_ID = "NOpBlnGInO9m6vDvFkFC"


class SpeechError(RuntimeError):
    """A readable speech service failure, without response headers."""


def speech_error(error: ApiError, action: str) -> SpeechError:
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("detail", body)
    if not isinstance(detail, dict):
        detail = {}
    code = detail.get("code", detail.get("status", ""))
    message = detail.get("message", "Speech service request failed")
    if code == "paid_plan_required":
        message = "This voice requires a paid ElevenLabs plan. Choose another voice or check your subscription."
    elif code == "unauthorized" and detail.get("status") == "missing_permissions":
        message = "The key needs voices_read permission to list voices. You can enter a voice ID directly."
    return SpeechError(f"ElevenLabs {action} ({error.status_code}): {str(message)[:800]}")


class Recorder:
    def __init__(self, artifacts: Path, device=None, sample_rate: int = 16000):
        self.artifacts, self.device, self.sample_rate = artifacts, device, sample_rate
        self._stream = None
        self._chunks: list[bytes] = []
        self._error: str | None = None
        self._bytes = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd
        if self._stream is not None:
            raise RuntimeError("Microphone is already recording")
        self._chunks, self._error, self._bytes = [], None, 0

        def callback(indata, frames, time, status):
            with self._lock:
                if status:
                    self._error = str(status)
                # Bound a forgotten hold to two minutes of memory; never truncate silently.
                if self._bytes + len(indata) > self.sample_rate * 2 * 120:
                    self._error = "Recording exceeded 120 seconds; release SPACE and retry"
                else:
                    self._chunks.append(bytes(indata))
                    self._bytes += len(indata)

        stream = sd.RawInputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                                   device=self.device, callback=callback)
        self._stream = stream
        stream.start()  # Keep the handle available for abort if startup fails.

    def abort(self) -> None:
        if self._stream is not None:
            stream = self._stream
            try:
                stream.abort()
            finally:
                stream.close()
                self._stream = None
        self._chunks.clear()

    def stop(self) -> Path:
        if self._stream is None:
            raise RuntimeError("Microphone is not recording")
        stream = self._stream
        try:
            stream.stop()
        finally:
            stream.close()
            self._stream = None
        with self._lock:
            pcm = b"".join(self._chunks)
            self._chunks.clear()
            error = self._error
        if error:
            raise RuntimeError(error)
        if not pcm:
            raise ValueError("No audio was recorded. Hold SPACE or the blue button while speaking, then release.")
        directory = self.artifacts / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"input_{uuid4().hex}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm)
        return path


class Speech:
    def __init__(self, client: httpx.AsyncClient, api_key: str,
                 voice_id: str = DEFAULT_VOICE_ID, device=None):
        self.client, self.api_key, self.voice_id, self.device = client, api_key, voice_id, device
        self._sdk = None

    def _elevenlabs(self) -> AsyncElevenLabs:
        if not self.api_key:
            raise ValueError("Set ELEVENLABS_API_KEY")
        if self._sdk is None:
            self._sdk = AsyncElevenLabs(api_key=self.api_key, httpx_client=self.client, timeout=45)
        return self._sdk

    async def transcribe(self, audio: Path) -> str:
        try:
            with audio.open("rb") as handle:
                transcription = await self._elevenlabs().speech_to_text.convert(
                    file=handle, model_id="scribe_v2", tag_audio_events=True,
                    language_code=None, diarize=True,
                    request_options={"timeout_in_seconds": 45, "max_retries": 0},
                )
        except ApiError as error:
            raise speech_error(error, "transcription") from error
        text = transcription.text.strip()
        if not text:
            raise ValueError("No speech was transcribed; hold SPACE and try again")
        return text

    async def list_voices(self) -> list[tuple[str, str]]:
        voices, token = [], None
        try:
            while True:
                page = await self._elevenlabs().voices.search(
                    page_size=100, next_page_token=token,
                    request_options={"timeout_in_seconds": 15, "max_retries": 0})
                voices.extend((voice.voice_id, voice.name or voice.voice_id) for voice in page.voices)
                token = page.next_page_token
                if not page.has_more or not token:
                    return voices
        except ApiError as error:
            raise speech_error(error, "voice list") from error

    async def speak(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd
        if not self.voice_id:
            raise ValueError("Set ELEVENLABS_VOICE_ID")
        chunks = []
        try:
            async for chunk in self._elevenlabs().text_to_speech.convert(
                voice_id=self.voice_id, text=text, model_id="eleven_v4", language_code="en",
                output_format="pcm_16000",
                request_options={"timeout_in_seconds": 45, "max_retries": 0},
            ):
                chunks.append(chunk)
        except ApiError as error:
            raise speech_error(error, "speech playback") from error
        pcm = b"".join(chunks)
        if not pcm or len(pcm) % 2:
            raise ValueError("TTS returned empty or malformed PCM audio")
        samples = np.frombuffer(pcm, dtype="<i2")
        try:
            sd.play(samples, samplerate=16000, device=self.device)
            while sd.get_stream().active:
                await asyncio.sleep(0.02)
        finally:
            # Cancellation stops actual playback before the next turn is accepted.
            sd.stop()
