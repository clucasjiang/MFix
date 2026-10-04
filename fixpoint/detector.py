"""Find the laser dot by how much it brightens a laser-off reference image.

Comparing against a laser-off frame cancels everything that is already in the
scene: red parts, steady LEDs, glare from room lights. Thresholds are set in
multiples of the frame's own noise level, so dim or dark surfaces still work
as long as the dot raises them above the noise.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    snr: float  # peak signal divided by the frame's noise level
    peak: float  # peak brightening, in 8-bit levels
    rivals: int = 0  # other bright spots in the search area (LEDs, reflections)

    @property
    def xy(self) -> np.ndarray:
        return np.array((self.x, self.y))


@dataclass
class DetectorConfig:
    blur_sigma: float = 1.5
    min_snr: float = 9.0  # searching the whole frame
    min_snr_near: float = 6.0  # searching a window around a prediction
    min_peak: float = 6.0  # 8-bit levels; ignores sensor noise on very clean frames
    noise_floor: float = 0.7
    max_area: int = 900  # larger changed regions are scene changes, not the dot
    changed_level: float = 30.0  # per-pixel change that counts toward staleness
    border: int = 3  # pixels ignored at the image edges


def laser_signal(image: np.ndarray, reference: np.ndarray, sigma: float, border: int = 0) -> np.ndarray:
    diff = image.astype(np.float32) - reference
    # A red laser raises the red channel most. On bright surfaces its center
    # also raises green and blue, so those get some weight too.
    signal = cv2.GaussianBlur(diff[..., 2] + 0.25 * (diff[..., 0] + diff[..., 1]), (0, 0), sigma)
    if border:
        # Webcams often have junk in their outermost rows and columns.
        signal[:border] = signal[-border:] = 0
        signal[:, :border] = signal[:, -border:] = 0
    return signal


def noise_level(values: np.ndarray) -> float:
    """Noise sigma, taking heavy tails into account.

    The median-based estimate fits Gaussian sensor noise, but real scenes
    also flicker (mains-powered lights on glossy surfaces), which produces
    rare large spikes. The 99.9th percentile of |signal| sits at 3.29 sigma
    for Gaussian noise; using it as well raises the threshold when spikes
    are common. The dot covers too few pixels to move that percentile.
    """
    median = np.median(values)
    deviations = np.abs(values - median)
    gaussian = 1.4826 * np.median(deviations)
    tail = np.percentile(deviations, 99.9) / 3.29
    return float(max(gaussian, tail))


def mean_image(images: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(images).astype(np.float32), axis=0)


class LaserDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.reference: np.ndarray | None = None
        self.last_signal: np.ndarray | None = None
        self.last_noise = 0.0

    def set_reference(self, images: list[np.ndarray]) -> None:
        """Store the average of one or more laser-off frames."""
        self.reference = mean_image(images)

    def detect(
        self,
        image: np.ndarray,
        predicted: np.ndarray | None = None,
        radius: float | None = None,
    ) -> Detection | None:
        if self.reference is None:
            return None
        signal = laser_signal(image, self.reference, self.config.blur_sigma, self.config.border)
        self.last_signal = signal
        return self.find(signal, predicted, radius)

    def find(
        self,
        signal: np.ndarray,
        predicted: np.ndarray | None = None,
        radius: float | None = None,
    ) -> Detection | None:
        """Pick the most dot-like peak, favoring peaks near a prediction."""
        config = self.config
        noise = max(noise_level(signal[::4, ::4]), config.noise_floor)
        self.last_noise = noise
        height, width = signal.shape
        gated = predicted is not None and radius is not None
        if gated:
            px, py = float(predicted[0]), float(predicted[1])
            x0, x1 = int(max(0, px - radius)), int(min(width, px + radius + 1))
            y0, y1 = int(max(0, py - radius)), int(min(height, py + radius + 1))
            if x1 - x0 < 3 or y1 - y0 < 3:
                return None
            min_snr = config.min_snr_near
        else:
            x0, x1, y0, y1 = 0, width, 0, height
            min_snr = config.min_snr
        region = signal[y0:y1, x0:x1]
        threshold = max(config.min_peak, min_snr * noise)
        if float(region.max()) < threshold:
            return None

        mask = (region >= threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[tuple[float, Detection]] = []
        for label in range(1, count):
            bx, by, bw, bh, area = stats[label]
            if area > config.max_area:
                continue
            patch = region[by : by + bh, bx : bx + bw]
            values = np.where(labels[by : by + bh, bx : bx + bw] == label, patch, 0.0)
            peak = float(values.max())
            # Centroid of the top half of the peak: robust to a lopsided halo.
            weights = np.where(values >= 0.5 * peak, values, 0.0)
            ys, xs = np.mgrid[0:bh, 0:bw]
            total = float(weights.sum())
            x = x0 + bx + float((weights * xs).sum()) / total
            y = y0 + by + float((weights * ys).sum()) / total
            snr = peak / noise
            score = snr
            if gated:
                distance = np.hypot(x - px, y - py)
                score = snr / (1.0 + (distance / (0.5 * radius)) ** 2)
            candidates.append((score, Detection(x, y, snr, peak)))
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate[0])[1]
        # Fragments of the same halo sit right next to it; anything further is a rival.
        rivals = sum(1 for _, other in candidates if np.hypot(other.x - best.x, other.y - best.y) > 8.0)
        return Detection(best.x, best.y, best.snr, best.peak, rivals)

    def changed_fraction(self, image: np.ndarray) -> float:
        """Share of the scene that differs from the reference (hands, lighting)."""
        if self.reference is None:
            return 1.0
        diff = np.abs(image[::4, ::4].astype(np.float32) - self.reference[::4, ::4])
        return float((diff.max(axis=2) > self.config.changed_level).mean())
