"""Simulated camera, gimbal firmware and scene for testing without hardware.

The scene includes the cases that broke earlier attempts: a reddish table,
a dark chip, a saturated glare stripe and a hole where the dot vanishes,
static and blinking red LEDs. The gimbal side emulates the firmware's byte
protocol, its speed ramp, PWM tick quantization and servo deadband, and the
camera adds latency and sensor noise.
"""

import math
import threading
import time

import cv2
import numpy as np

from .camera import FrameSource
from .gimbal import (
    FIRMWARE_DEFAULT_SPEED,
    FIRMWARE_STARTUP_ANGLE,
    LASER_DEVICE,
    PITCH_AXIS,
    PITCH_FINE_AXIS,
    SPEED_DEVICE,
    STOP_DEVICE,
    YAW_AXIS,
    YAW_FINE_AXIS,
    AxisMotion,
    GimbalBase,
)


def rotation(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    vector = axis * math.radians(degrees)
    matrix, _ = cv2.Rodrigues(vector)
    return matrix


def align(a, b) -> np.ndarray:
    """Rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3)
    return rotation(axis, math.degrees(math.acos(np.clip(a @ b, -1, 1))))


def pwm_angle(angle: float, fine: bool) -> float:
    """Angle the servo actually receives after the firmware's PWM quantization."""
    if fine:
        pulse = round(500 + angle * 2000 / 180)
        tick = 20000 / 2**14
    else:
        pulse = 500 + int(angle + 0.5) * 2000 // 180
        tick = 20000 / 2**10
    return (int(pulse / tick) * tick - 500) * 180 / 2000


class SimWorld:
    def __init__(
        self,
        size: tuple[int, int] = (640, 480),
        fine: bool = True,
        latency: float = 0.09,
        noise: float = 2.0,
        deadband: float = 0.5,
        exposure: float = -8.0,
        seed: int = 1,
    ) -> None:
        self.size = size
        self.fine = fine
        self.latency = latency
        self.deadband = deadband
        self.exposure = exposure
        width, height = size
        # Camera 200 mm above the board, A4 width filling the image.
        self.focal = 0.5 * width / (148.5 / 200.0)
        self.camera_center = np.array((0.0, 0.0, 200.0))
        self.camera_rotation = rotation((1, 0, 0), 4) @ rotation((0, 1, 0), -3)
        self.k1 = -0.06
        # Laser beside the camera, on a slightly rolled mount; pitch 144 hits the center.
        self.laser_origin = np.array((45.0, -70.0, 185.0))
        aim = -self.laser_origin / np.linalg.norm(self.laser_origin)
        self.base = rotation(aim, 6) @ align((0, 0, -1), aim)
        self.yaw_gain, self.pitch_gain = -1.04, 0.97  # inverted yaw, servo scale error
        self.yaw_zero, self.pitch_zero = 90.0, 144.0

        self._lock = threading.Lock()
        self._yaw = AxisMotion(FIRMWARE_STARTUP_ANGLE)
        self._pitch = AxisMotion(FIRMWARE_STARTUP_ANGLE)
        self._speed = FIRMWARE_DEFAULT_SPEED
        self._servo = np.array((FIRMWARE_STARTUP_ANGLE, FIRMWARE_STARTUP_ANGLE))
        self._servo_goal = self._servo.copy()
        self._servo_t = time.perf_counter()
        self._laser_events: list[tuple[float, bool]] = [(0.0, False)]

        rng = np.random.default_rng(seed)
        self.background, self.reflectance, self.blinking_led = self._make_scene(rng)
        self._noise_bank = [rng.normal(0, noise, (height, width, 3)).astype(np.float32) for _ in range(6)]
        self._rng = rng

    # Scene

    def _make_scene(self, rng: np.random.Generator):
        width, height = self.size
        low = cv2.resize(rng.normal(0, 1, (height // 32, width // 32)).astype(np.float32), (width, height))
        streaks = cv2.GaussianBlur(rng.normal(0, 1, (height, 1)).astype(np.float32), (0, 0), 3)
        wood = np.zeros((height, width, 3), np.float32)
        wood[:] = (22, 38, 80)  # reddish-brown table, BGR at exposure -8
        wood += (6 * low + 8 * np.repeat(streaks, width, axis=1))[..., None]
        reflectance = np.full((height, width), 0.45, np.float32)

        paper = np.array([(70, 45), (575, 30), (585, 415), (60, 430)], np.int32)
        cv2.fillPoly(wood, [paper], (118, 118, 122))
        cv2.fillPoly(reflectance, [paper], 1.0)
        cv2.rectangle(wood, (150, 280), (250, 360), (14, 14, 16), -1)  # dark chip
        cv2.rectangle(reflectance, (150, 280), (250, 360), 0.18, -1)
        cv2.rectangle(wood, (380, 250), (520, 380), (35, 75, 25), -1)  # green PCB
        cv2.rectangle(reflectance, (380, 250), (520, 380), 0.55, -1)
        for x, y in ((400, 270), (430, 360), (505, 300)):
            cv2.circle(wood, (x, y), 3, (30, 30, 210), -1)  # red LEDs
        cv2.ellipse(wood, (300, 120), (60, 14), 15, 0, 360, (252, 252, 252), -1)  # glare
        cv2.circle(wood, (420, 150), 14, (5, 5, 5), -1)  # hole: the dot vanishes
        cv2.circle(reflectance, (420, 150), 14, 0.0, -1)
        return wood, reflectance, (480, 350)

    # Geometry

    def dot_pixel(self, yaw: float, pitch: float) -> np.ndarray | None:
        """Where the dot appears for physical servo angles, or None if off-image."""
        psi = math.radians(self.yaw_gain * (yaw - self.yaw_zero))
        phi = math.radians(self.pitch_gain * (pitch - self.pitch_zero))
        local = rotation((0, 1, 0), math.degrees(psi)) @ rotation((1, 0, 0), math.degrees(phi)) @ (0, 0, -1)
        direction = self.base @ local
        if direction[2] > -1e-6:
            return None
        point = self.laser_origin - self.laser_origin[2] / direction[2] * direction
        flip = np.diag((1.0, 1.0, -1.0))  # camera looks down the -z axis
        camera = flip @ self.camera_rotation @ (point - self.camera_center)
        x, y = camera[0] / camera[2], camera[1] / camera[2]
        distortion = 1 + self.k1 * (x * x + y * y)
        width, height = self.size
        pixel = np.array((self.focal * x * distortion + width / 2, self.focal * y * distortion + height / 2))
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            return None
        return pixel

    # Firmware

    def receive(self, data: bytes) -> None:
        """Apply one command frame, as the ESP32 firmware would."""
        now = time.perf_counter()
        device = data[1]
        with self._lock:
            if device in (YAW_AXIS, PITCH_AXIS) and data[2] <= 180:
                axis = self._yaw if device == YAW_AXIS else self._pitch
                axis.retarget(float(data[2]), now, self._speed)
            elif device in (YAW_FINE_AXIS, PITCH_FINE_AXIS) and self.fine:
                axis = self._yaw if device == YAW_FINE_AXIS else self._pitch
                axis.retarget(((data[2] << 8) | data[3]) / 100.0, now, self._speed)
            elif device == LASER_DEVICE:
                self._laser_events.append((now, bool(data[2])))
            elif device == STOP_DEVICE:
                for axis in (self._yaw, self._pitch):
                    axis.retarget(axis.position(now, self._speed), now, self._speed)
            elif device == SPEED_DEVICE and self.fine and data[2] > 0:
                for axis in (self._yaw, self._pitch):
                    axis.retarget(axis.target, now, self._speed)
                self._speed = float(data[2])

    def _advance_servos(self, t: float) -> None:
        """Integrate servo motion up to time t: PWM ticks, deadband, finite speed."""
        while self._servo_t < t:
            dt = min(0.005, t - self._servo_t)
            self._servo_t += dt
            commanded = np.array([
                pwm_angle(axis.position(self._servo_t, self._speed), self.fine) for axis in (self._yaw, self._pitch)
            ])
            error = commanded - self._servo
            # A servo ignores errors inside its deadband and stops just short
            # of the command on the side it approached from.
            restart = np.abs(commanded - self._servo_goal) > self.deadband / 2
            self._servo_goal = np.where(restart, commanded - np.sign(error) * self.deadband * 0.2, self._servo_goal)
            self._servo += np.clip(self._servo_goal - self._servo, -400 * dt, 400 * dt)

    def _laser_on_at(self, t: float) -> bool:
        state = False
        for when, on in self._laser_events:
            if when > t:
                break
            state = on
        return state

    def servo_angles(self) -> np.ndarray:
        with self._lock:
            return self._servo.copy()

    # Rendering

    def render(self, t: float) -> np.ndarray:
        """Camera image of the scene as it was at time t (call with increasing t)."""
        with self._lock:
            self._advance_servos(t)
            servo = self._servo.copy()
            laser = self._laser_on_at(t)
        gain = 2.0 ** (self.exposure + 8)
        image = self.background * gain
        if int(t / 0.65) % 2:
            cv2.circle(image, self.blinking_led, 3, (30 * gain, 30 * gain, 220 * gain), -1)
        if laser:
            pixel = self.dot_pixel(*servo)
            if pixel is not None:
                self._draw_dot(image, pixel, gain)
        image += self._noise_bank[self._rng.integers(len(self._noise_bank))]
        return np.clip(image, 0, 255).astype(np.uint8)

    def _draw_dot(self, image: np.ndarray, pixel: np.ndarray, gain: float) -> None:
        width, height = self.size
        cx, cy = pixel
        x0, x1 = max(0, int(cx) - 8), min(width, int(cx) + 9)
        y0, y1 = max(0, int(cy) - 8), min(height, int(cy) + 9)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        profile = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 1.8**2))
        amplitude = 80.0 * gain * self.reflectance[int(cy), int(cx)]
        image[y0:y1, x0:x1] += (profile * amplitude)[..., None] * np.array((0.35, 0.4, 1.0), np.float32)


class SimGimbal(GimbalBase):
    def __init__(self, world: SimWorld) -> None:
        super().__init__(fine=world.fine)
        self.world = world

    def _write(self, data: bytes) -> None:
        self.world.receive(data)


class SimCamera(FrameSource):
    def __init__(self, world: SimWorld, fps: float = 30.0) -> None:
        super().__init__()
        self.world = world
        self.size = world.size
        self._period = 1.0 / fps
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def exposure(self) -> float:
        return self.world.exposure

    def set_exposure(self, value: float) -> float:
        self.world.exposure = float(value)
        return self.world.exposure

    def show_settings(self) -> None:
        pass

    def start(self) -> "SimCamera":
        self._running = True
        self._thread = threading.Thread(target=self._run, name="sim-camera", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        next_time = time.perf_counter()
        while self._running:
            next_time += self._period
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.perf_counter()
            now = time.perf_counter()
            self._publish(self.world.render(now - self.world.latency), now)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
