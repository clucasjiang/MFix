"""Interactively control the ESP32 gimbal from a Windows keyboard."""

import argparse
import time

try:
    import msvcrt
except ModuleNotFoundError as error:
    raise SystemExit("This controller currently requires Windows.") from error

try:
    import serial
except ModuleNotFoundError as error:
    raise SystemExit(
        "PySerial is required. Install it with: python -m pip install pyserial"
    ) from error


BAUD_RATE = 115200
FRAME_HEADER = 0x15
YAW_AXIS = 0x01
PITCH_AXIS = 0x02
LASER_DEVICE = 0x03
MINIMUM_ANGLE = 0
MAXIMUM_ANGLE = 180
CENTER_ANGLE = 90


def clamp_angle(angle: int) -> int:
    return max(MINIMUM_ANGLE, min(MAXIMUM_ANGLE, angle))


def send_position(port: serial.Serial, axis: int, angle: int) -> None:
    port.write(bytes((FRAME_HEADER, axis, clamp_angle(angle))))
    port.flush()


def set_laser(port: serial.Serial, enabled: bool) -> None:
    port.write(bytes((FRAME_HEADER, LASER_DEVICE, int(enabled))))
    port.flush()


def read_key() -> str:
    """Read one key and translate Windows arrow-key sequences."""
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        return {
            "H": "up",
            "P": "down",
            "K": "left",
            "M": "right",
        }.get(msvcrt.getwch(), "unknown")
    return key.lower()


def print_controls(step: int) -> None:
    print()
    print("Controls")
    print("  Left / A   : yaw left")
    print("  Right / D  : yaw right")
    print("  Up / W     : pitch up")
    print("  Down / S   : pitch down")
    print("  Space      : toggle laser on/off")
    print("  C          : center both axes")
    print("  Q          : quit")
    print(f"  Movement step: {step} degrees")
    print()


def print_status(yaw: int, pitch: int, laser_on: bool) -> None:
    laser_status = "ON" if laser_on else "OFF"
    print(
        f"Yaw: {yaw:3d} degrees | Pitch: {pitch:3d} degrees | "
        f"Laser: {laser_status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control the ESP32 gimbal with arrow keys or WASD."
    )
    parser.add_argument(
        "port",
        nargs="?",
        default="COM6",
        help="ESP32 serial port (default: COM6)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="degrees moved by each key press (default: 5)",
    )
    parser.add_argument(
        "--invert-yaw",
        action="store_true",
        help="reverse the yaw keyboard direction",
    )
    parser.add_argument(
        "--invert-pitch",
        action="store_true",
        help="reverse the pitch keyboard direction",
    )
    args = parser.parse_args()

    if not 1 <= args.step <= 180:
        parser.error("--step must be between 1 and 180")

    yaw = CENTER_ANGLE
    pitch = CENTER_ANGLE
    laser_on = False
    yaw_direction = 1 if args.invert_yaw else -1
    pitch_direction = 1 if args.invert_pitch else -1

    print(f"Opening {args.port} at {BAUD_RATE} baud...")

    try:
        with serial.Serial(args.port, BAUD_RATE, timeout=1) as port:
            # Opening the port can reset the ESP32. Wait before sending data.
            time.sleep(2.0)

            # Establish a known starting position for keyboard increments.
            send_position(port, YAW_AXIS, yaw)
            send_position(port, PITCH_AXIS, pitch)
            set_laser(port, False)

            print_controls(args.step)
            print_status(yaw, pitch, laser_on)

            try:
                while True:
                    key = read_key()

                    if key in ("q", "\x03"):
                        break

                    old_yaw = yaw
                    old_pitch = pitch

                    if key in ("a", "left"):
                        yaw = clamp_angle(yaw - args.step * yaw_direction)
                    elif key in ("d", "right"):
                        yaw = clamp_angle(yaw + args.step * yaw_direction)
                    elif key in ("w", "up"):
                        pitch = clamp_angle(pitch + args.step * pitch_direction)
                    elif key in ("s", "down"):
                        pitch = clamp_angle(pitch - args.step * pitch_direction)
                    elif key == "c":
                        yaw = CENTER_ANGLE
                        pitch = CENTER_ANGLE
                    elif key == " ":
                        laser_on = not laser_on
                        set_laser(port, laser_on)
                    else:
                        continue

                    if yaw != old_yaw:
                        send_position(port, YAW_AXIS, yaw)
                    if pitch != old_pitch:
                        send_position(port, PITCH_AXIS, pitch)

                    print_status(yaw, pitch, laser_on)
            finally:
                # Fail safe on a normal exit or keyboard interruption.
                try:
                    set_laser(port, False)
                except serial.SerialException:
                    pass

    except serial.SerialException as error:
        raise SystemExit(
            f"Could not use {args.port}: {error}\n"
            "Close PlatformIO Serial Monitor and verify the COM port."
        ) from error

    print("Controller closed. The gimbal will hold its last position.")


if __name__ == "__main__":
    main()
