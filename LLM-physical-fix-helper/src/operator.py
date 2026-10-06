"""Serialize diagnostic turns; only image/STT and TTS/pointer overlap."""
import asyncio
import json
import time
from pathlib import Path

from .camera import save_debug
from .models import DiagnosticResponse, Session, State
from .speech import SpeechError


class CameraError(RuntimeError):
    """No photo, new or previous; the message is fit for the conversation panel."""


class Operator:
    def __init__(self, recorder, camera, speech, grok, pointer, *, stage=8,
                 min_confidence=0.7, pointer_timeout=30.0, processing_timeout=60.0,
                 grok_timeout=95.0, output_timeout=90.0, notify=print, on_event=None):
        self.recorder, self.camera, self.speech = recorder, camera, speech
        self.grok, self.pointer = grok, pointer
        self.stage, self.min_confidence = stage, min_confidence
        self.pointer_timeout, self.processing_timeout = pointer_timeout, processing_timeout
        self.grok_timeout, self.output_timeout = grok_timeout, output_timeout
        self.notify = notify
        self.on_event = on_event
        self.state = State.READY
        self.session = Session()
        self.last_error: str | None = None
        self.last_response = None
        self.last_frame = None
        self.debug_path: Path | None = None
        self._previous_frame = None  # stands in if the camera fails
        self._reused_photo = False
        self._image_task = None
        self._turn_task = None
        self._pointer_active = False
        self._microphone_fault = False
        self._recovering = False
        self._cleanup_lock = asyncio.Lock()
        self.last_timings = {}

    def _state(self, state):
        self.state = state
        self.notify(f"State: {state.value}")
        self._emit("state", state=state)

    def _emit(self, kind, **data):
        if self.on_event is not None:
            self.on_event(kind, **data)

    async def _capture_image(self):
        started = time.perf_counter()
        try:
            frame = await self.camera.capture_and_process()
        except Exception as error:
            # Camera and OpenCV details stay in the console; carry on with the
            # previous photo when there is one.
            print(f"Camera failed: {type(error).__name__}: {error}", flush=True)
            if self._previous_frame is None:
                raise CameraError("I couldn't take a photo. Check the camera and try again.") from error
            frame, self._reused_photo = self._previous_frame, True
        finally:
            self.last_timings["capture_seconds"] = round(time.perf_counter() - started, 3)
        self.last_frame = self._previous_frame = frame
        self._emit("image", frame=frame)
        return frame

    def space_down(self) -> bool:
        # Synchronous admission eliminates races from key repeat/queued presses.
        if self.state != State.READY or self._recovering:
            return False
        self._state(State.RECORDING)
        self.last_error, self.last_response, self.debug_path = None, None, None
        self.last_frame = None
        self._reused_photo = False
        self.last_timings = {}
        self._emit("turn_started")
        try:
            self.recorder.start()
            if self.stage >= 2:
                self._image_task = asyncio.create_task(self._capture_image())
        except Exception as error:
            self.last_error = str(error)
            self.notify(f"Input failure: {error}")
            self._emit("error", message=self.last_error)
            try:
                self.recorder.abort()
            except Exception as cleanup_error:
                self._microphone_fault = True
                self.last_error = f"Microphone cleanup failed: {cleanup_error}"
                self._state(State.ERROR)
            else:
                self._state(State.READY)
            return False
        return True

    def space_up(self) -> bool:
        if self.state != State.RECORDING:
            return False
        self._state(State.PROCESSING)
        self._turn_task = asyncio.create_task(self._run_turn())
        return True

    async def wait_until_idle(self):
        if self._turn_task:
            await asyncio.shield(self._turn_task)

    def reset_session(self) -> bool:
        if self.state != State.READY:
            return False
        self.session.reset()
        self._previous_frame = None  # a new session may be a different device
        self.notify("New diagnostic session")
        self._emit("session_reset")
        return True

    async def _wait_pointer(self, response, context):
        # Set this BEFORE sending: a failed/timed-out send may have been accepted.
        self._pointer_active = True
        box = response.bbox
        async def operation():
            await self.pointer.send_bbox(box.x1, box.y1, box.x2, box.y2)
            self._state(State.WAITING_FOR_POINTER)
            await self.pointer.wait_until_complete()
        try:
            await asyncio.wait_for(operation(), self.pointer_timeout)
        except Exception as error:
            # Any pointer failure (unreachable, missed, timed out, disconnected):
            # tell the model and the console, never the user, and carry on.
            # A new aim or photo interrupts anything still running downstream.
            message = f"The pointer could not mark {response.target_label}: {str(error) or type(error).__name__}"
            context.pending_feedback.append(message + ". The laser is not marking it.")
            print(message, flush=True)
        self._pointer_active = False

    async def _outputs(self, response, context):
        jobs = []
        if self.stage >= 5:
            jobs.append(asyncio.create_task(self.speech.speak(response.speech)))
        if self.stage >= 6 and response.target_present:
            jobs.append(asyncio.create_task(self._wait_pointer(response, context)))
        completion = asyncio.gather(*jobs)
        try:
            await asyncio.wait_for(completion, self.output_timeout)
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            # Python 3.10 can leave the outer gather exception unobserved when
            # wait_for is interrupted; consume it after device jobs are drained.
            await asyncio.gather(completion, return_exceptions=True)

    async def _run_turn(self):
        context = self.session if self.stage >= 8 else Session()
        started = time.perf_counter()
        try:
            audio = self.recorder.stop()
            async def inputs():
                async def transcribe():
                    began = time.perf_counter()
                    try:
                        return await self.speech.transcribe(audio)
                    finally:
                        self.last_timings["stt_seconds"] = round(time.perf_counter() - began, 3)
                transcript_task = asyncio.create_task(transcribe())
                try:
                    if self._image_task is None:
                        return await transcript_task, None
                    # Shield the camera worker: its thread must finish before READY.
                    transcript, frame = await asyncio.gather(transcript_task, asyncio.shield(self._image_task))
                    return transcript, frame
                finally:
                    if not transcript_task.done():
                        transcript_task.cancel()
                    await asyncio.gather(transcript_task, return_exceptions=True)
            transcript, frame = await asyncio.wait_for(inputs(), self.processing_timeout)
            if not transcript.strip():
                raise ValueError("Empty transcript")
            if self._reused_photo:
                context.pending_feedback.append("The camera failed, so the attached photo is the previous one "
                                                "again; the scene may have changed since.")
            self.notify(f"Transcript: {transcript}")
            self._emit("transcript", text=transcript)
            self.last_frame = frame
            if self.stage < 3:
                return
            self._state(State.WAITING_FOR_GROK)
            began = time.perf_counter()
            try:
                response = await asyncio.wait_for(self.grok.diagnose(transcript, frame, context), self.grok_timeout)
            finally:
                self.last_timings["model_seconds"] = round(time.perf_counter() - began, 3)
                if getattr(self.grok, "last_metrics", None):
                    self.notify("Model request: " + json.dumps(self.grok.last_metrics))
            # Revalidate even injected adapters; never trust coerced/mutated coordinates.
            response = DiagnosticResponse.model_validate(response.model_dump())
            self.notify("Model response: " + response.model_dump_json(indent=2))
            debug_box, debug_label = None, response.target_label
            try:
                if response.target_present:
                    response.bbox.validate_image(frame.width, frame.height)
                    debug_box = response.bbox  # Visualize even a rejected low-confidence target.
                    if response.confidence < self.min_confidence:
                        raise ValueError("Target confidence is below the pointing threshold")
            except ValueError as error:
                context.pending_feedback.append(f"Target rejected: {error}; no pointing or original instruction occurred.")
                self.notify(f"Target rejected: {error}")
                response = DiagnosticResponse(speech="I couldn't identify that component reliably. Please describe it or give me another view.",
                    target_present=False, target_label=None, bbox=None, confidence=None, status="clarification")
            self.last_response = response
            self._emit("response", response=response)
            self.notify(response.model_dump_json(indent=2))
            if self.stage >= 4:
                self.debug_path = await asyncio.to_thread(save_debug, frame, debug_box, debug_label)
                self.notify(f"Debug image: {self.debug_path}")
                self._emit("debug_image", path=self.debug_path)
            context.turns.append({"transcript": transcript, "response": response.model_dump(), "output_complete": False})
            self._state(State.OUTPUTTING)
            began = time.perf_counter()
            try:
                await self._outputs(response, context)
            finally:
                self.last_timings["output_seconds"] = round(time.perf_counter() - began, 3)
            context.turns[-1]["output_complete"] = True
        except asyncio.CancelledError:
            self.last_error = "Turn interrupted"
            context.pending_feedback.append("Previous turn interrupted; speech/pointing may not have completed.")
            self.notify(self.last_error)
            self._emit("error", message=self.last_error)
        except Exception as error:
            self.last_error = str(error) if isinstance(error, (SpeechError, CameraError)) else f"{type(error).__name__}: {error}"
            context.pending_feedback.append(f"Previous turn failed: {type(error).__name__}; output may not have completed.")
            self.notify(f"Turn failed: {self.last_error}")
            self._emit("error", message=self.last_error)
            # Never speak unvalidated model output after a malformed response.
            if self.stage >= 5 and not self._pointer_active and not isinstance(error, SpeechError):
                try:
                    await asyncio.wait_for(self.speech.speak("I couldn't complete that step. Please try again."), self.output_timeout)
                except Exception:
                    pass
        finally:
            await self._cleanup()
            self.last_timings["total_after_release_seconds"] = round(time.perf_counter() - started, 3)
            self.notify("Turn timings: " + json.dumps(self.last_timings))
            self._emit("timings", timings=dict(self.last_timings))

    async def _cleanup(self):
        async with self._cleanup_lock:
            try:
                self.recorder.abort()
                self._microphone_fault = False
            except Exception as error:
                self._microphone_fault = True
                self.last_error = f"Microphone cleanup failed: {error}"
                self._emit("error", message=self.last_error)
            if self._image_task is not None:
                # asyncio.to_thread cannot kill capture; drain it to prevent overlap.
                await asyncio.gather(asyncio.shield(self._image_task), return_exceptions=True)
                self._image_task = None
            if self._pointer_active or self._microphone_fault:
                self._state(State.ERROR)
                self.notify("Device cleanup unconfirmed. New turns locked; press Escape to retry acknowledged stop.")
            elif self.state != State.STOPPED:
                self._state(State.READY)

    async def interrupt(self):
        if self._recovering:
            return
        self._recovering = True
        try:
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
                await asyncio.gather(self._turn_task, return_exceptions=True)
                # Cancellation before the coroutine's first instruction skips its
                # finally block; always finish device cleanup here as well.
                await self._cleanup()
            elif self.state == State.RECORDING:
                self._state(State.PROCESSING)
                await self._cleanup()
            elif self.state == State.ERROR and not self._pointer_active:
                await self._cleanup()
            if self._pointer_active:
                try:
                    acknowledged = await asyncio.wait_for(self.pointer.cancel(), self.pointer_timeout)
                    if acknowledged is not True:
                        raise RuntimeError("Downstream did not acknowledge stopped/idle")
                    self._pointer_active = False
                    await self._cleanup()
                except Exception as error:
                    self.last_error = f"Pointer stop unconfirmed: {error}"
                    self._state(State.ERROR)
                    self.notify(self.last_error)
                    self._emit("error", message=self.last_error)
        finally:
            self._recovering = False

    async def close(self):
        await self.interrupt()
        self._state(State.STOPPED)
