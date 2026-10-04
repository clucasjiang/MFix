"""Camera, gimbal and detector together: timing, blinking, settling."""

import time
from typing import Callable

import numpy as np

from .camera import Frame, FrameSource
from .detector import Detection, LaserDetector, laser_signal, mean_image
from .gimbal import GimbalBase

# Used until measure_latency() runs; deliberately on the slow side.
DEFAULT_LATENCY = 0.2

Predictor = Callable[[float], np.ndarray | None]


def median_detection(detections: list[Detection]) -> Detection:
    xy = np.median([d.xy for d in detections], axis=0)
    return Detection(
        float(xy[0]),
        float(xy[1]),
        float(np.median([d.snr for d in detections])),
        float(np.median([d.peak for d in detections])),
        max(d.rivals for d in detections),
    )


class Rig:
    def __init__(self, camera: FrameSource, gimbal: GimbalBase, detector: LaserDetector) -> None:
        self.camera = camera
        self.gimbal = gimbal
        self.detector = detector
        # Time from a command to the first frame that shows its effect.
        self.latency = DEFAULT_LATENCY
        # Mechanical lag after the PWM ramp reaches its target.
        self.servo_settle = 0.08
        # Called with every processed frame so the UI can draw it.
        self.on_frame: Callable[[Frame, Detection | None], None] | None = None
        self.blinks = 0

    def _report(self, frame: Frame, detection: Detection | None) -> None:
        if self.on_frame is not None:
            self.on_frame(frame, detection)

    def frames_after(self, t_event: float, count: int, timeout: float = 1.5) -> list[Frame]:
        """Frames guaranteed to have been captured after t_event."""
        return self.camera.frames(count, t_event + self.latency, timeout)

    def blink_locate(
        self,
        pairs: int = 2,
        predicted: np.ndarray | None = None,
        radius: float | None = None,
    ) -> Detection | None:
        """Find the dot by toggling the laser; also refreshes the reference.

        With several pairs, a pixel counts only as much as its weakest pair:
        the laser brightens it every time, while a blinking LED or a moving
        hand almost never lines up with every toggle.
        The gimbal must be still. The laser is left on.
        """
        config = self.detector.config
        sigma = config.blur_sigma
        off_frames: list[Frame] = []
        signals: list[np.ndarray] = []
        last_on: Frame | None = None
        for _ in range(pairs):
            self.gimbal.set_laser(False)
            off = self.frames_after(time.perf_counter(), 2)
            self.gimbal.set_laser(True)
            on = self.frames_after(time.perf_counter(), 2)
            if not off or not on:
                break
            off_frames += off
            last_on = on[-1]
            signals.append(laser_signal(mean_image([f.image for f in on]),
                                        mean_image([f.image for f in off]), sigma, config.border))
        self.blinks += 1
        if not signals:
            return None
        self.detector.set_reference([f.image for f in off_frames])
        signal = np.minimum.reduce(signals)
        self.detector.last_signal = signal
        detection = None
        if predicted is not None and radius is not None:
            detection = self.detector.find(signal, predicted, radius)
        if detection is None:
            detection = self.detector.find(signal)
        self._report(last_on, detection)
        return detection

    def track(self, after_seq: int, predict: Predictor | None = None, radius: float | None = None,
              timeout: float = 0.5, search_all: bool = False) -> tuple[Frame | None, Detection | None]:
        """Detect the dot in the next frame, near the prediction if one is given.

        With `search_all`, a miss near the prediction falls back to the whole frame.
        """
        frame = self.camera.wait_frame(after_seq, timeout=timeout)
        if frame is None:
            return None, None
        predicted = predict(frame.t) if predict is not None else None
        detection = self.detector.detect(frame.image, predicted, radius if predicted is not None else None)
        if detection is None and predicted is not None and search_all and self.detector.last_signal is not None:
            detection = self.detector.find(self.detector.last_signal)
        self._report(frame, detection)
        return frame, detection

    def settle(self, predict: Predictor | None, radius: float | None, timeout: float = 1.5,
               stable_px: float = 1.5, max_misses: int = 4) -> Detection | None:
        """Wait for the current move to finish and the dot to stop; return where it stopped.

        Frames during the move are still tracked (for display) but only frames
        after the move, servo lag and camera latency count toward settling.
        Without a prediction the whole frame is searched. If another bright
        spot competes with the dot, a blink decides which one is the laser.
        """
        detection = self._settle(predict, radius, timeout, stable_px, max_misses)
        if detection is not None and detection.rivals:
            predicted = predict(time.perf_counter()) if predict is not None else None
            return self.blink_locate(pairs=2, predicted=predicted, radius=radius)
        return detection

    def _settle(self, predict: Predictor | None, radius: float | None, timeout: float,
                stable_px: float, max_misses: int) -> Detection | None:
        settled_from = self.gimbal.arrival_time() + self.servo_settle + self.latency
        deadline = max(settled_from, time.perf_counter()) + timeout
        recent: list[Detection] = []
        misses = 0
        seq = 0
        while time.perf_counter() < deadline:
            frame, detection = self.track(seq, predict, radius, timeout=max(0.05, deadline - time.perf_counter()))
            if frame is None:
                break
            seq = frame.seq
            if frame.t < settled_from:
                continue
            if detection is None:
                misses += 1
                if misses >= max_misses:
                    return None
                continue
            recent = (recent + [detection])[-3:]
            if len(recent) == 3:
                xy = np.array([d.xy for d in recent])
                if np.abs(xy - np.median(xy, axis=0)).max() <= stable_px:
                    return median_detection(recent)
        return median_detection(recent) if recent else None

    def _level_at(self, image: np.ndarray, x: int, y: int) -> float:
        """Laser signal around one pixel, against the current reference."""
        height, width = image.shape[:2]
        x0, x1 = max(0, x - 7), min(width, x + 8)
        y0, y1 = max(0, y - 7), min(height, y + 8)
        patch = laser_signal(image[y0:y1, x0:x1], self.detector.reference[y0:y1, x0:x1],
                             self.detector.config.blur_sigma)
        return float(patch.max())

    def measure_latency(self, trials: int = 3, attempts: int = 3) -> float | None:
        """Time laser toggles until the camera shows them. Leaves the laser on.

        Every toggle must show up, with consistent timing; otherwise the spot
        being watched is not the laser (say, a blinking LED) and we retry.
        """
        for _ in range(attempts):
            detection = self.blink_locate(pairs=3)
            if detection is None:
                return None
            samples = self._toggle_delays(detection, trials)
            if samples is not None and max(samples) - min(samples) <= 0.12:
                self.latency = max(samples) + 0.01
                return self.latency
        self.gimbal.set_laser(True)
        return None

    def _toggle_delays(self, detection: Detection, trials: int) -> list[float] | None:
        x, y = int(round(detection.x)), int(round(detection.y))
        on_level = detection.peak
        samples = []
        for _ in range(trials):
            for enabled in (False, True):
                latest = self.camera.latest()
                seq = latest.seq if latest else 0
                t0 = time.perf_counter()
                self.gimbal.set_laser(enabled)
                delay = None
                while time.perf_counter() - t0 < 1.0:
                    frame = self.camera.wait_frame(seq, timeout=0.5)
                    if frame is None:
                        break
                    seq = frame.seq
                    level = self._level_at(frame.image, x, y)
                    if (level > 0.7 * on_level) if enabled else (level < 0.3 * on_level):
                        delay = frame.t - t0
                        break
                if delay is None:
                    self.gimbal.set_laser(True)
                    return None
                samples.append(delay)
        return samples

    def tune_exposure(self, candidates: tuple[float, ...] = (-10, -9, -8, -7, -6, -5)) -> float | None:
        """Pick the exposure where the dot stands out most, avoiding blown-out backgrounds.

        Run with the dot on the kind of surface you will point at.
        """
        if not hasattr(self.camera, "set_exposure"):
            return None
        results = []
        for exposure in candidates:
            self.camera.set_exposure(exposure)
            self.camera.frames(6, time.perf_counter())  # let the sensor adapt
            detection = self.blink_locate()
            if detection is None:
                continue
            saturated = float((self.detector.reference.max(axis=2) >= 250).mean())
            score = detection.snr * (0.5 if saturated > 0.01 else 1.0)
            results.append((score, -exposure, exposure))
        if not results:
            return None
        best = max(results)[2]
        self.camera.set_exposure(best)
        self.camera.frames(6, time.perf_counter())
        self.blink_locate()
        return best
