"""Automatic calibration: steer the dot across the image and record where it lands."""

import threading
from typing import Callable

import numpy as np

from .model import AngleModel
from .rig import Rig

ProgressCallback = Callable[[str, np.ndarray], None]


class CalibrationError(RuntimeError):
    pass


def image_grid(size: tuple[int, int], columns: int, rows: int, margin: float) -> np.ndarray:
    width, height = size
    xs = np.linspace(margin * width, (1 - margin) * width, columns)
    ys = np.linspace(margin * height, (1 - margin) * height, rows)
    return np.array([(x, y) for y in ys for x in xs])


def nearest_first(points: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Greedy tour so each move is short and the next point is near known ones."""
    remaining = list(range(len(points)))
    order = []
    here = start
    while remaining:
        index = min(remaining, key=lambda i: np.hypot(*(points[i] - here)))
        remaining.remove(index)
        order.append(index)
        here = points[index]
    return points[order]


def local_affine(source: np.ndarray, values: np.ndarray, point: np.ndarray, k: int = 6) -> np.ndarray | None:
    """Fit an affine map on the k nearest samples and evaluate it at `point`."""
    nearest = np.argsort(np.hypot(*(source - point).T))[:k]
    design = np.column_stack((source[nearest], np.ones(len(nearest))))
    if np.linalg.matrix_rank(design) < 3:
        return None
    coefficients, *_ = np.linalg.lstsq(design, values[nearest], rcond=None)
    return np.append(point, 1.0) @ coefficients


def calibrate(
    rig: Rig,
    size: tuple[int, int],
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    columns: int = 7,
    rows: int = 5,
    margin: float = 0.07,
    probe_step: float = 4.0,
    max_jump: float = 20.0,
    speed: float = 120.0,
) -> AngleModel:
    gimbal = rig.gimbal
    samples: list[list[float]] = []  # yaw, pitch, x, y

    def report(message: str) -> None:
        if progress is not None:
            progress(message, np.array(samples))

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise CalibrationError("Calibration cancelled.")

    def measure(predicted: np.ndarray | None, radius: float | None):
        detection = rig.settle((lambda t: predicted) if predicted is not None else None, radius)
        if detection is None:
            detection = rig.blink_locate(pairs=2, predicted=predicted, radius=radius)
        return detection

    previous_speed = gimbal.speed
    gimbal.set_speed(speed)
    try:
        start = np.array(gimbal.target())
        report("Looking for the laser dot...")
        detection = rig.blink_locate(pairs=2)
        if detection is None:
            raise CalibrationError("No laser dot visible. Jog it into view with the arrow keys, then press C.")
        samples.append([*start, detection.x, detection.y])
        origin = detection.xy

        # Probe each axis both ways to learn which way and how far the dot moves.
        for axis, name in ((0, "yaw"), (1, "pitch")):
            moved = 0.0
            for direction in (1, -1):
                check_cancel()
                angles = start.copy()
                angles[axis] += direction * probe_step
                commanded = gimbal.move_to(*angles, use_limits=False)
                report(f"Probing {name}...")
                detection = measure(None, None)
                if detection is not None:
                    samples.append([*commanded, detection.x, detection.y])
                    moved = max(moved, float(np.hypot(*(detection.xy - origin))))
            if moved < 3.0:
                raise CalibrationError(
                    f"Moving {name} by {probe_step:g} degrees did not visibly move the dot. "
                    "Check servo power and that the dot is on a visible surface."
                )

        points = np.array(samples)
        targets = nearest_first(image_grid(size, columns, rows, margin), origin)
        for index, target in enumerate(targets, start=1):
            check_cancel()
            points = np.array(samples)
            angles = local_affine(points[:, 2:4], points[:, 0:2], target)
            if angles is None:
                continue
            here = np.array(gimbal.target())
            angles = here + np.clip(angles - here, -max_jump, max_jump)
            commanded = gimbal.move_to(*angles, use_limits=False)
            report(f"Calibrating point {index}/{len(targets)}...")
            detection = measure(target, 80.0)
            if detection is None:
                report(f"Point {index}/{len(targets)}: dot not visible here, skipping.")
                continue
            samples.append([*commanded, detection.x, detection.y])

        if len(samples) < 10:
            raise CalibrationError(
                f"Only found the dot at {len(samples)} points. Make sure the surface fills the view."
            )
        model = AngleModel.fit_robust(np.array(samples))
        report(f"Calibrated from {len(model.samples)} points, fit error {model.rms:.1f} px.")
        return model
    finally:
        gimbal.set_speed(previous_speed)
