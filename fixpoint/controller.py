"""Background worker that owns the gimbal. The UI sends it commands."""

import csv
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .aim import AimResult, AimStatus, Aimer, Rect
from .calibration import CalibrationError, calibrate
from .camera import Frame
from .detector import Detection
from .model import AngleModel
from .rig import Rig

# Commands that interrupt whatever long operation is running.
INTERRUPTS_ANYTHING = {"clear", "jog", "quit"}
# Commands that only interrupt an aim; a running calibration finishes first.
INTERRUPTS_AIM = {"aim", "calibrate", "snapshot"}
# When the dot stops at a target it blinks this many times: FLASH_INTERVAL
# seconds off, then FLASH_INTERVAL seconds on.
FLASH_BLINKS = 3
FLASH_INTERVAL = 0.5
# Frames to wait after an exposure change before the image is trustworthy.
EXPOSURE_SETTLE_FRAMES = 8
LOG_FIELDS = ["time", "x1", "y1", "x2", "y2", "status", "elapsed_s", "corrections", "blinks",
              "dot_x", "dot_y", "error_px", "yaw", "pitch"]


@dataclass
class View:
    """Everything the UI draws, copied out under a lock."""

    status: str = "starting"
    message: str = ""
    rect: Rect | None = None
    aim_status: AimStatus | None = None
    detection: Detection | None = None
    detection_t: float = 0.0
    predicted: np.ndarray | None = None
    angles: tuple[float, float] = (90.0, 90.0)
    samples: np.ndarray | None = None
    model_rms: float | None = None
    px_per_degree: float | None = None
    latency: float = 0.0
    last_result: AimResult | None = None
    noise: float = 0.0


class Controller(threading.Thread):
    def __init__(
        self,
        rig: Rig,
        size: tuple[int, int],
        calibration_path: Path,
        start_angles: tuple[float, float],
        hold: bool = True,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(name="controller", daemon=True)
        self.rig = rig
        self.size = size
        self.calibration_path = Path(calibration_path)
        self.start_angles = start_angles
        self.hold = hold
        self.log_path = log_path
        self.model: AngleModel | None = None
        self.aimer: Aimer | None = None
        self.rect: Rect | None = None
        self._commands: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._view = View()
        self._running = True
        self._seq = 0
        self._last_refresh = 0.0
        self._last_stale_check = 0.0
        self._outside_since: float | None = None
        # Only re-aim on drift after a VALID result; failed targets would loop.
        self._holding = False
        self._busy: str | None = None  # command in progress
        # Photos for the assistant are taken this many exposure steps brighter
        # than tracking (each step doubles the exposure time); 0 disables.
        self.photo_exposure_step = 2.0
        # Resolution for those photos, if the camera can take stills (Camera.still);
        # None uses tracking frames. Map photo pixels back with camera.view_in_photo.
        self.photo_size: tuple[int, int] | None = None
        rig.on_frame = self._on_frame

    # UI side

    def submit(self, name: str, *args) -> None:
        self._enqueue(name, args, None)

    def request(self, name: str, *args) -> Future:
        """Like submit(), but returns a Future with the command's result.

        "aim" resolves to an AimResult once the dot has stopped and blinked, and
        "snapshot" to a BGR image. Cancelling the future before the command
        starts skips it.
        """
        future: Future = Future()
        self._enqueue(name, args, future)
        return future

    def _enqueue(self, name: str, args: tuple, future: Future | None) -> None:
        if future is not None and self.ident is not None and not self.is_alive():
            future.set_exception(RuntimeError("The pointer controller has stopped."))
            return
        if name in INTERRUPTS_ANYTHING or (name in INTERRUPTS_AIM and self._busy == "aim"):
            self._cancel.set()
        self._commands.put((name, args, future))

    @property
    def idle(self) -> bool:
        """No command running or queued."""
        return self._busy is None and self._commands.empty()

    def view(self) -> View:
        with self._lock:
            return replace(self._view)

    def shutdown(self) -> None:
        self.submit("quit")
        self.join(timeout=3.0)

    # Worker side

    def _update(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._view, key, value)

    def _say(self, message: str) -> None:
        self._update(message=message)
        print(message, flush=True)

    def _on_frame(self, frame: Frame, detection: Detection | None) -> None:
        gimbal = self.rig.gimbal
        predicted = None
        if self.aimer is not None:
            predicted = self.aimer.predict(*gimbal.position(frame.t - self.rig.latency))
        self._update(detection=detection, detection_t=frame.t, predicted=predicted,
                     angles=gimbal.position(), noise=self.rig.detector.last_noise)

    def run(self) -> None:
        try:
            self._startup()
            while self._running:
                try:
                    name, args, future = self._commands.get_nowait()
                except queue.Empty:
                    self._idle()
                    continue
                if future is not None and not future.set_running_or_notify_cancel():
                    continue
                self._cancel.clear()
                self._busy = name
                try:
                    result = self._handle(name, args)
                except Exception as error:
                    if future is None:
                        if not isinstance(error, RuntimeError):  # expected failures carry a message
                            traceback.print_exc()
                        self._say(str(error))
                    else:
                        future.set_exception(error)
                else:
                    if future is not None:
                        future.set_result(result)
                finally:
                    self._busy = None
        except Exception as error:  # keep the UI alive and show what happened
            traceback.print_exc()
            self._update(status="error")
            self._say(f"Controller error: {error}")
        finally:
            self._running = False
            self._fail_pending()

    def _fail_pending(self) -> None:
        """Release anyone still waiting on a queued request."""
        while True:
            try:
                _, _, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if future is not None and future.set_running_or_notify_cancel():
                future.set_exception(RuntimeError("The pointer controller has stopped."))

    def _startup(self) -> None:
        rig, gimbal = self.rig, self.rig.gimbal
        if gimbal.fine:
            self._say("Gimbal firmware supports fine moves (0.01 degree commands).")
        else:
            self._say("Old gimbal firmware: whole-degree moves only. Flash firmware/ for finer aiming.")
        start = self.start_angles
        if self.calibration_path.exists():
            try:
                self._set_model(AngleModel.load(self.calibration_path))
                samples = self.model.samples
                center = np.array(self.size) / 2
                start = tuple(samples[np.argmin(np.hypot(*(samples[:, 2:4] - center).T)), 0:2])
            except (OSError, ValueError, KeyError) as error:
                self._say(f"Ignoring unreadable calibration file: {error}")
        gimbal.move_to(*start, use_limits=False)
        time.sleep(max(0.0, gimbal.arrival_time() - time.perf_counter()) + 0.4)
        gimbal.set_laser(True)
        self._update(status="locating")
        latency = rig.measure_latency()
        if latency is None:
            self._update(status="no dot")
            self._say("Laser dot not found. Jog it into view (arrows/WASD), then press B or C.")
            return
        self._update(latency=latency)
        self._say(f"Found the dot. Camera latency {latency * 1000:.0f} ms.")
        if self.aimer is not None:
            self._verify_calibration()
        else:
            self._update(status="uncalibrated")
            self._say("No calibration yet: press C (takes about 15 s).")

    def _verify_calibration(self) -> None:
        detection = self.rig.blink_locate()
        if detection is None:
            return
        residual = detection.xy - self.model.to_pixel(*self.rig.gimbal.target())
        error = float(np.hypot(*residual))
        if error > 30.0:
            self._update(status="check calibration")
            self._say(f"Saved calibration is off by {error:.0f} px here. Press C to recalibrate.")
        else:
            self.aimer.offset = residual
            self._update(status="ready")
            self._say(f"Loaded calibration (fit {self.model.rms:.1f} px, now off by {error:.1f} px). Drag a target.")

    def _set_model(self, model: AngleModel) -> None:
        self.model = model
        self.aimer = Aimer(self.rig, model)
        self.rig.gimbal.limits = model.angle_bounds(margin=3.0)
        yaw, pitch = model.samples[len(model.samples) // 2, 0:2]
        self._update(samples=model.samples, model_rms=model.rms,
                     px_per_degree=model.pixels_per_degree(yaw, pitch))

    def _handle(self, name: str, args: tuple):
        rig, gimbal = self.rig, self.rig.gimbal
        if name == "quit":
            self._running = False
        elif name == "calibrate":
            self._calibrate()
        elif name == "aim":
            if self.aimer is None:
                raise RuntimeError("The pointer is not calibrated yet: press C in the tracker window.")
            self.rect = args[0]
            return self._aim()
        elif name == "snapshot":
            return self._snapshot()
        elif name == "clear":
            self.rect = None
            self._update(rect=None, aim_status=None, status="ready" if self.aimer else "uncalibrated")
        elif name == "blink":
            detection = rig.blink_locate(pairs=2)
            self._say("Dot found." if detection else "No dot visible.")
        elif name == "jog":
            self.rect = None
            self._update(rect=None, aim_status=None)
            yaw, pitch = gimbal.target()
            gimbal.move_to(yaw + args[0], pitch + args[1], use_limits=False)
        elif name == "laser":
            gimbal.set_laser(not gimbal.laser_on)
        elif name == "exposure":
            exposure = rig.camera.set_exposure(rig.camera.exposure + args[0])
            rig.camera.frames(5, time.perf_counter())  # let the sensor adapt
            self._refresh_reference()
            self._say(f"Exposure {exposure:g}.")
        elif name == "tune":
            self._update(status="tuning exposure")
            best = rig.tune_exposure()
            self._update(status="ready" if self.aimer else "uncalibrated")
            self._say(f"Exposure set to {best:g}." if best is not None else "Could not see the dot at any exposure.")

    def _calibrate(self) -> None:
        self.rect = None
        self._update(status="calibrating", rect=None, aim_status=None, samples=None)

        def progress(message: str, samples: np.ndarray) -> None:
            self._update(message=message, samples=samples if len(samples) else None)

        try:
            model = calibrate(self.rig, self.size, progress, self._cancel)
        except CalibrationError as error:
            self._update(status="uncalibrated" if self.model is None else "ready")
            if self.model is not None:
                self._set_model(self.model)
            self._say(str(error))
            return
        model.save(self.calibration_path, extra={"saved": datetime.now().isoformat(timespec="seconds"),
                                                 "image_size": list(self.size)})
        self._set_model(model)
        self._update(status="ready")
        self._say(f"Calibrated from {len(model.samples)} points, fit error {model.rms:.1f} px. "
                  "Drag a target rectangle.")

    def _aim(self) -> AimResult:
        rect = self.rect
        gimbal = self.rig.gimbal
        self._update(rect=rect, aim_status=AimStatus.MOVING, status="aiming")
        if not gimbal.laser_on:
            # Off since the last photo; let the camera see it before tracking.
            gimbal.set_laser(True)
            time.sleep(self.rig.latency + 0.05)
        previous, self._busy = self._busy, "aim"
        try:
            result = self.aimer.aim(rect, self._cancel)
            self._update(aim_status=result.status, last_result=result, status=result.status.value)
            error = f", {result.error_px:.1f} px from center" if result.error_px is not None else ""
            print(f"{result.status.value.upper()}: {result.message} "
                  f"({result.elapsed:.2f} s, {result.corrections} corrections{error})", flush=True)
            self._update(message=result.message)
            self._log(result)
            self._outside_since = None
            self._holding = result.status == AimStatus.VALID
            if result.status not in (AimStatus.CANCELLED, AimStatus.UNREACHABLE):
                # The dot has stopped at or near the target. Still busy aiming,
                # so a new target or photo cuts the blinking short.
                self._flash()
        finally:
            self._busy = previous
        return result

    def _flash(self) -> None:
        """Blink the dot in place to show it has stopped; a moving dot never blinks.

        The laser is left on, even when a clear or a new aim cuts the blinking short.
        """
        gimbal = self.rig.gimbal
        try:
            for _ in range(FLASH_BLINKS):
                gimbal.set_laser(False)
                if self._cancel.wait(FLASH_INTERVAL):
                    return
                gimbal.set_laser(True)
                if self._cancel.wait(FLASH_INTERVAL):
                    return
        finally:
            if not gimbal.laser_on:
                gimbal.set_laser(True)

    def _snapshot(self) -> np.ndarray:
        """A clean photo for the assistant: target cleared, laser off, brighter exposure.

        The laser stays off until the next aim, so the dot only ever marks the
        current instruction. Tracking runs dark; the photo is taken a couple of
        exposure steps brighter, at `photo_size` if set, then tracking resumes
        with its own exposure and a fresh reference.
        """
        rig, gimbal, camera = self.rig, self.rig.gimbal, self.rig.camera
        self.rect = None
        self._holding = False
        self._update(rect=None, aim_status=None, status="taking photo")
        gimbal.set_laser(False)
        tracking = getattr(camera, "exposure", None)
        brighter = tracking is not None and hasattr(camera, "set_exposure") and self.photo_exposure_step > 0
        # Longer than about 1/32 s would slow the camera below 30 fps.
        exposure = min(tracking + self.photo_exposure_step, -5.0) if brighter else tracking
        try:
            image = None
            if self.photo_size is not None and tuple(self.photo_size) != tuple(camera.size) and hasattr(camera, "still"):
                try:
                    # Reopening the camera starts after the laser-off command, so no wait is needed.
                    image = camera.still(tuple(self.photo_size), exposure)
                    camera.frames(3, time.perf_counter())  # tracking stream is back
                except Exception as error:  # OpenCV or the driver: a smaller photo beats none
                    print(f"Full-size photo failed ({error}); using a tracking frame.", flush=True)
            if image is None:
                image = self._tracking_photo(exposure if brighter else None)
            self._refresh_reference()
        finally:
            self._update(status="ready" if self.aimer else "uncalibrated")
        return image

    def _tracking_photo(self, exposure: float | None) -> np.ndarray:
        """The photo from tracking frames, optionally at another exposure for the moment."""
        rig, camera = self.rig, self.rig.camera
        tracking = getattr(camera, "exposure", None)
        if exposure is not None:
            camera.set_exposure(exposure)
        frames = camera.frames(EXPOSURE_SETTLE_FRAMES + 2, time.perf_counter() + rig.latency)
        if exposure is not None:
            camera.set_exposure(tracking)
            camera.frames(EXPOSURE_SETTLE_FRAMES, time.perf_counter())
        if len(frames) < 2:
            raise RuntimeError("The camera stopped delivering frames.")
        # Averaging two frames halves the sensor noise the model has to look through.
        return np.mean([f.image.astype(np.float32) for f in frames[-2:]], axis=0).round().astype(np.uint8)

    def _log(self, result: AimResult) -> None:
        if self.log_path is None or result.status == AimStatus.CANCELLED:
            return
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(LOG_FIELDS)
            dot = result.detection
            rect = result.rect
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"), rect.x1, rect.y1, rect.x2, rect.y2,
                result.status.value, f"{result.elapsed:.3f}", result.corrections, result.blinks,
                f"{dot.x:.1f}" if dot else "", f"{dot.y:.1f}" if dot else "",
                f"{result.error_px:.1f}" if dot else "", *result.angles,
            ])

    def _refresh_reference(self) -> None:
        """Re-take the laser-off reference: blink if the laser is on, else just grab frames."""
        rig = self.rig
        if rig.gimbal.laser_on:
            rig.blink_locate(pairs=1)
        else:
            frames = rig.camera.frames(2, time.perf_counter())
            if frames:
                rig.detector.set_reference([f.image for f in frames])
        self._last_refresh = time.perf_counter()

    def _idle(self) -> None:
        """Track the dot between commands; keep it in the target when holding."""
        rig = self.rig
        predict = self.aimer.predictor() if self.aimer is not None else None
        frame, detection = rig.track(self._seq, predict, 40.0, timeout=0.2, search_all=True)
        if frame is None:
            return
        self._seq = frame.seq
        now = time.perf_counter()
        moving = rig.gimbal.arrival_time() + rig.latency > now

        # Lighting or the scene changed: the reference no longer cancels the background.
        if now - self._last_stale_check > 1.0 and not moving:
            self._last_stale_check = now
            if rig.detector.changed_fraction(frame.image) > 0.03 and now - self._last_refresh > 3.0:
                self._refresh_reference()

        rect = self.rect
        if not (self.hold and self._holding and rect is not None and not moving):
            return
        if detection is not None and not rect.contains(detection.xy):
            self._outside_since = self._outside_since or frame.t
            if frame.t - self._outside_since > 0.5:
                self._say("Dot left the target; re-aiming.")
                self._aim()
        elif detection is not None:
            self._outside_since = None
