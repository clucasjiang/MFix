"""Steer the laser dot into a target rectangle."""

import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .detector import Detection
from .model import AngleModel
from .rig import Predictor, Rig, median_detection


class AimStatus(str, Enum):
    MOVING = "moving"
    VALID = "valid"  # dot seen inside the rectangle
    PREDICTED = "predicted"  # dot not visible, but the model puts it inside
    LOST = "lost"  # dot not visible and not predicted inside
    FAILED = "failed"  # dot visible but could not be brought inside
    UNREACHABLE = "unreachable"  # rectangle outside the calibrated area
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_corners(cls, a, b) -> "Rect":
        """Accepts the two corners in any order."""
        return cls(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array(((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2))

    def contains(self, point) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class AimResult:
    status: AimStatus
    rect: Rect
    detection: Detection | None
    angles: tuple[float, float]
    corrections: int
    elapsed: float
    blinks: int
    message: str = ""

    @property
    def error_px(self) -> float | None:
        """Distance from the dot to the rectangle center."""
        if self.detection is None:
            return None
        return float(np.hypot(*(self.detection.xy - self.rect.center)))


class Aimer:
    def __init__(self, rig: Rig, model: AngleModel, radius: float = 40.0,
                 confirm_frames: int = 5, max_corrections: int = 10) -> None:
        self.rig = rig
        self.model = model
        self.radius = radius
        self.confirm_frames = confirm_frames
        self.max_corrections = max_corrections
        # Where the dot really lands minus where the model says: absorbs a
        # bumped tripod or a surface at a slightly different height.
        self.offset = np.zeros(2)

    def predict(self, yaw: float, pitch: float) -> np.ndarray:
        return self.model.to_pixel(yaw, pitch) + self.offset

    def predictor(self) -> Predictor:
        """Predicted dot position for a frame time, following the firmware ramp."""
        rig = self.rig

        def predict(t: float) -> np.ndarray:
            return self.predict(*rig.gimbal.position(t - rig.latency))

        return predict

    def learn(self, angles: np.ndarray, detection: Detection) -> None:
        residual = detection.xy - self.model.to_pixel(*angles)
        self.offset += 0.5 * (residual - self.offset)

    def confirm(self, rect: Rect) -> Detection | None:
        """Require the dot inside the rectangle on the next few frames (one miss allowed)."""
        inside: list[Detection] = []
        misses = 0
        seq = 0
        predict = self.predictor()
        while len(inside) < self.confirm_frames:
            frame, detection = self.rig.track(seq, predict, self.radius)
            if frame is None:
                return None
            seq = frame.seq
            if detection is None or not rect.contains(detection.xy):
                misses += 1
                if misses > 1:
                    return None
                continue
            inside.append(detection)
        return median_detection(inside)

    def aim(self, rect: Rect, cancel: threading.Event | None = None) -> AimResult:
        rig, gimbal, model = self.rig, self.rig.gimbal, self.model
        started = time.perf_counter()
        blinks_before = rig.blinks
        goal = rect.center
        angles = np.array(gimbal.target())
        corrections = 0

        def result(status: AimStatus, detection: Detection | None, message: str = "") -> AimResult:
            return AimResult(status, rect, detection, (float(angles[0]), float(angles[1])), corrections,
                             time.perf_counter() - started, rig.blinks - blinks_before, message)

        if not model.covers(*goal, margin=15.0):
            return result(AimStatus.UNREACHABLE, None, "Target is outside the calibrated area.")

        angles = np.array(gimbal.move_to(*model.to_angles(*(goal - self.offset))))
        gain = 1.0
        last_error = np.inf
        last_detection: Detection | None = None
        for corrections in range(self.max_corrections + 1):
            if cancel is not None and cancel.is_set():
                return result(AimStatus.CANCELLED, last_detection)
            detection = rig.settle(self.predictor(), self.radius)
            if detection is None:
                detection = rig.blink_locate(pairs=2, predicted=self.predict(*angles), radius=2 * self.radius)
            if detection is None:
                if rect.contains(self.predict(*angles)):
                    return result(AimStatus.PREDICTED, None, "Dot not visible here; the model places it inside.")
                return result(AimStatus.LOST, None, "Lost the dot.")
            self.learn(angles, detection)

            if rect.contains(detection.xy):
                confirmed = self.confirm(rect)
                if confirmed is not None:
                    return result(AimStatus.VALID, confirmed, "Dot is inside the target.")
                last_detection = detection
                continue

            error = goal - detection.xy
            distance = float(np.hypot(*error))
            moved = last_detection is not None and np.hypot(*(detection.xy - last_detection.xy)) > 1.0
            # Shrink the step if the dot moved but got no closer (overshooting).
            # If it did not move at all, keep the gain: the command is still
            # inside the servo deadband and needs to keep accumulating.
            if moved and distance > 0.8 * last_error:
                gain = max(0.3, gain * 0.6)
            last_error = distance
            last_detection = detection

            try:
                step = gain * np.linalg.solve(model.jacobian(*angles), error)
            except np.linalg.LinAlgError:
                return result(AimStatus.FAILED, detection, "Calibration is degenerate here.")
            largest = float(np.abs(step).max())
            if 0 < largest < gimbal.step:
                step *= gimbal.step / largest  # smallest move the protocol can express
            new_angles = np.array(gimbal.move_to(*(angles + step)))
            if np.allclose(new_angles, angles):
                return result(AimStatus.FAILED, detection, "Gimbal is at its angle limit.")
            angles = new_angles

        return result(
            AimStatus.FAILED,
            last_detection,
            f"Could not settle inside; the target may be smaller than one gimbal step "
            f"(about {model.pixels_per_degree(*angles) * max(gimbal.step, 0.1):.1f} px).",
        )
