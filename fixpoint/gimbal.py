"""Serial link to the ESP32 gimbal, plus a model of where the servos are."""

import math
import time

BAUD_RATE = 115200
FRAME_HEADER = 0x15
YAW_AXIS = 0x01
PITCH_AXIS = 0x02
LASER_DEVICE = 0x03
STOP_DEVICE = 0x04
YAW_FINE_AXIS = 0x05
PITCH_FINE_AXIS = 0x06
SPEED_DEVICE = 0x07
PING_DEVICE = 0x08
FIRMWARE_ID = b"FIXPOINT"

MINIMUM_ANGLE = 0.0
MAXIMUM_ANGLE = 180.0
FIRMWARE_STARTUP_ANGLE = 90.0
FIRMWARE_DEFAULT_SPEED = 60.0
FINE_STEP = 0.01
LEGACY_STEP = 1.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class AxisMotion:
    """Mirrors the firmware's constant-speed ramp for one axis."""

    def __init__(self, angle: float) -> None:
        self.start = angle
        self.target = angle
        self.t0 = 0.0

    def position(self, t: float, speed: float) -> float:
        travel = speed * max(0.0, t - self.t0)
        remaining = self.target - self.start
        if abs(remaining) <= travel:
            return self.target
        return self.start + math.copysign(travel, remaining)

    def retarget(self, target: float, t: float, speed: float) -> None:
        self.start = self.position(t, speed)
        self.target = target
        self.t0 = t

    def arrival(self, speed: float) -> float:
        return self.t0 + abs(self.target - self.start) / speed


class GimbalBase:
    """Command logic shared by the serial gimbal and the simulator.

    The firmware gives no position feedback, so `position()` replays its
    ramp to estimate where each servo is being driven at a given time.
    """

    def __init__(self, fine: bool) -> None:
        self.fine = fine
        self.speed = FIRMWARE_DEFAULT_SPEED
        self.laser_on = False
        self._yaw = AxisMotion(FIRMWARE_STARTUP_ANGLE)
        self._pitch = AxisMotion(FIRMWARE_STARTUP_ANGLE)
        # Soft limits (yaw_min, yaw_max, pitch_min, pitch_max) for aiming.
        self.limits = (MINIMUM_ANGLE, MAXIMUM_ANGLE, MINIMUM_ANGLE, MAXIMUM_ANGLE)

    @property
    def step(self) -> float:
        """Smallest angle change the protocol can express."""
        return FINE_STEP if self.fine else LEGACY_STEP

    def quantize(self, angle: float, low: float, high: float) -> float:
        angle = clamp(angle, max(low, MINIMUM_ANGLE), min(high, MAXIMUM_ANGLE))
        return round(round(angle / self.step) * self.step, 2)

    def move_to(self, yaw: float, pitch: float, use_limits: bool = True) -> tuple[float, float]:
        """Send a move and return the angles actually commanded."""
        yaw_min, yaw_max, pitch_min, pitch_max = (
            self.limits if use_limits else (MINIMUM_ANGLE, MAXIMUM_ANGLE) * 2
        )
        yaw = self.quantize(yaw, yaw_min, yaw_max)
        pitch = self.quantize(pitch, pitch_min, pitch_max)
        now = time.perf_counter()
        if yaw != self._yaw.target:
            self._send_angle(YAW_AXIS, yaw)
            self._yaw.retarget(yaw, now, self.speed)
        if pitch != self._pitch.target:
            self._send_angle(PITCH_AXIS, pitch)
            self._pitch.retarget(pitch, now, self.speed)
        return yaw, pitch

    def target(self) -> tuple[float, float]:
        return self._yaw.target, self._pitch.target

    def position(self, t: float | None = None) -> tuple[float, float]:
        """Estimated commanded servo angles at time t (perf_counter seconds)."""
        t = time.perf_counter() if t is None else t
        return self._yaw.position(t, self.speed), self._pitch.position(t, self.speed)

    def arrival_time(self) -> float:
        """perf_counter time when the firmware ramp reaches the current target."""
        return max(self._yaw.arrival(self.speed), self._pitch.arrival(self.speed))

    def set_speed(self, degrees_per_second: float) -> None:
        """Change the ramp speed. Only fine-protocol firmware supports this."""
        if not self.fine:
            return
        speed = clamp(round(degrees_per_second), 1, 255)
        now = time.perf_counter()
        for axis in (self._yaw, self._pitch):
            axis.retarget(axis.target, now, self.speed)
        self.speed = float(speed)
        self._write(bytes((FRAME_HEADER, SPEED_DEVICE, int(speed))))

    def set_laser(self, enabled: bool) -> None:
        self.laser_on = enabled
        self._write(bytes((FRAME_HEADER, LASER_DEVICE, int(enabled))))

    def stop(self) -> None:
        now = time.perf_counter()
        self._write(bytes((FRAME_HEADER, STOP_DEVICE, 0)))
        for axis in (self._yaw, self._pitch):
            here = axis.position(now, self.speed)
            axis.start = axis.target = here
            axis.t0 = now

    def close(self) -> None:
        self.set_laser(False)

    def _send_angle(self, axis: int, angle: float) -> None:
        if self.fine:
            device = YAW_FINE_AXIS if axis == YAW_AXIS else PITCH_FINE_AXIS
            centidegrees = int(round(angle * 100))
            self._write(bytes((FRAME_HEADER, device, centidegrees >> 8, centidegrees & 0xFF)))
        else:
            self._write(bytes((FRAME_HEADER, axis, int(round(angle)))))

    def _write(self, data: bytes) -> None:
        raise NotImplementedError


class SerialGimbal(GimbalBase):
    """The real ESP32 over USB serial."""

    def __init__(self, port_name: str, force_legacy: bool = False) -> None:
        import serial

        self._port = serial.Serial(port_name, BAUD_RATE, timeout=0.05)
        # Opening the port resets the ESP32: servos return to 90 degrees and
        # the laser is off. Wait for the firmware to boot before talking.
        time.sleep(2.0)
        self._port.reset_input_buffer()
        super().__init__(fine=False if force_legacy else self._ping())
        now = time.perf_counter()
        for axis in (self._yaw, self._pitch):
            axis.t0 = now

    def _ping(self) -> bool:
        self._port.write(bytes((FRAME_HEADER, PING_DEVICE, 0)))
        self._port.flush()
        deadline = time.perf_counter() + 0.5
        reply = b""
        while time.perf_counter() < deadline:
            reply += self._port.read(64)
            if FIRMWARE_ID in reply:
                return True
        return False

    def _write(self, data: bytes) -> None:
        self._port.write(data)
        self._port.flush()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._port.close()
