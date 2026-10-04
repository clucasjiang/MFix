"""Smooth mapping between gimbal angles and where the dot lands in the image.

The true mapping depends on the gimbal geometry, the camera lens, and the
offset between camera and laser. A cubic polynomial fitted to measured
samples captures all of it without measuring any of them.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def monomials(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    columns = [u**i * v ** (total - i) for total in range(degree + 1) for i in range(total, -1, -1)]
    return np.stack(columns, axis=-1)


def monomial_gradients(u: float, v: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    du, dv = [], []
    for total in range(degree + 1):
        for i in range(total, -1, -1):
            j = total - i
            du.append(i * u ** (i - 1) * v**j if i else 0.0)
            dv.append(j * u**i * v ** (j - 1) if j else 0.0)
    return np.array(du), np.array(dv)


def degree_for(count: int) -> int:
    """Highest polynomial degree the sample count supports with some slack."""
    if count >= 16:
        return 3
    if count >= 8:
        return 2
    return 1


class PolyMap:
    """Least-squares 2D -> 2D polynomial on normalized inputs."""

    def __init__(self, degree: int, center: np.ndarray, scale: np.ndarray, coefficients: np.ndarray) -> None:
        self.degree = degree
        self.center = np.asarray(center, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, degree: int, ridge: float = 1e-6) -> "PolyMap":
        source = np.asarray(source, dtype=float)
        center = source.mean(axis=0)
        scale = np.maximum(source.std(axis=0), 1e-6)
        normalized = (source - center) / scale
        design = monomials(normalized[:, 0], normalized[:, 1], degree)
        regularized = design.T @ design + ridge * np.eye(design.shape[1])
        coefficients = np.linalg.solve(regularized, design.T @ np.asarray(target, dtype=float))
        return cls(degree, center, scale, coefficients)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        normalized = (points - self.center) / self.scale
        return monomials(normalized[:, 0], normalized[:, 1], self.degree) @ self.coefficients

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        """2x2 matrix of d(output)/d(input) at one point."""
        u, v = (np.asarray(point, dtype=float) - self.center) / self.scale
        du, dv = monomial_gradients(u, v, self.degree)
        return np.column_stack((du @ self.coefficients / self.scale[0], dv @ self.coefficients / self.scale[1]))

    def to_dict(self) -> dict:
        return {
            "degree": self.degree,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolyMap":
        return cls(data["degree"], data["center"], data["scale"], data["coefficients"])


class AngleModel:
    """Forward (angles -> pixel) and inverse (pixel -> angles) mapping.

    `samples` rows are (yaw, pitch, x, y) measured during calibration.
    """

    def __init__(self, forward: PolyMap, inverse: PolyMap, samples: np.ndarray) -> None:
        self.forward = forward
        self.inverse = inverse
        self.samples = np.asarray(samples, dtype=float)
        self.hull = cv2.convexHull(self.samples[:, 2:4].astype(np.float32))

    @classmethod
    def fit(cls, samples: np.ndarray, degree: int | None = None) -> "AngleModel":
        samples = np.asarray(samples, dtype=float)
        degree = degree_for(len(samples)) if degree is None else degree
        angles, pixels = samples[:, 0:2], samples[:, 2:4]
        forward = PolyMap.fit(angles, pixels, degree)
        inverse = PolyMap.fit(pixels, angles, degree)
        return cls(forward, inverse, samples)

    @classmethod
    def fit_robust(cls, samples: np.ndarray) -> "AngleModel":
        """Fit, drop samples far off the fit (e.g. a detected reflection), refit."""
        model = cls.fit(samples)
        errors = model.residuals()
        keep = errors <= 3.0 * np.median(errors) + 3.0
        if keep.sum() >= 6 and not keep.all():
            model = cls.fit(np.asarray(samples)[keep])
        return model

    def to_pixel(self, yaw: float, pitch: float) -> np.ndarray:
        return self.forward(np.array((yaw, pitch)))[0]

    def jacobian(self, yaw: float, pitch: float) -> np.ndarray:
        """Pixels moved per degree: columns are yaw and pitch."""
        return self.forward.jacobian(np.array((yaw, pitch)))

    def to_angles(self, x: float, y: float, iterations: int = 5) -> np.ndarray:
        """Invert the forward map, refining the inverse fit with Newton steps."""
        target = np.array((x, y), dtype=float)
        angles = self.inverse(target)[0]
        for _ in range(iterations):
            error = target - self.to_pixel(*angles)
            if np.hypot(*error) < 0.05:
                break
            try:
                step = np.linalg.solve(self.jacobian(*angles), error)
            except np.linalg.LinAlgError:
                break
            angles = angles + np.clip(step, -5.0, 5.0)
        return angles

    def residuals(self) -> np.ndarray:
        predicted = self.forward(self.samples[:, 0:2])
        return np.hypot(*(predicted - self.samples[:, 2:4]).T)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals() ** 2)))

    def angle_bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        yaw, pitch = self.samples[:, 0], self.samples[:, 1]
        return (
            float(yaw.min() - margin),
            float(yaw.max() + margin),
            float(pitch.min() - margin),
            float(pitch.max() + margin),
        )

    def covers(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) lies inside the calibrated area, or within `margin` px of it."""
        distance = cv2.pointPolygonTest(self.hull, (float(x), float(y)), True)
        return distance >= -margin

    def pixels_per_degree(self, yaw: float, pitch: float) -> float:
        return float(np.linalg.svd(self.jacobian(yaw, pitch), compute_uv=False).mean())

    def save(self, path: Path, extra: dict | None = None) -> None:
        data = {
            "forward": self.forward.to_dict(),
            "inverse": self.inverse.to_dict(),
            "samples": self.samples.tolist(),
            "rms_px": self.rms,
            **(extra or {}),
        }
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: Path) -> "AngleModel":
        data = json.loads(Path(path).read_text())
        return cls(PolyMap.from_dict(data["forward"]), PolyMap.from_dict(data["inverse"]), np.array(data["samples"]))
