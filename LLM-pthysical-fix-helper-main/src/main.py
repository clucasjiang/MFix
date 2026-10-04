"""Desktop repair assistant with conversation, image, and device selection."""
import argparse
import asyncio
from pathlib import Path

import httpx

from .camera import Camera
from .camera_capture import CAMERA_NAME
from .credentials import CredentialError, load_settings
from .model_backend import create_model
from .models import State
from .operator import Operator
from .pointer import load_pointer
from .rig import RigError
from .speech import DEFAULT_VOICE_ID, Recorder, Speech

ROOT = Path(__file__).resolve().parent.parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--image", type=Path, help="Use a fixture instead of opening the camera")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "data")
    parser.add_argument("--pointer", help="Downstream adapter factory: package.module:factory (default: dry run)")
    parser.add_argument("--pointer-timeout", type=float, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--state-mode", choices=("stored", "client"), default="client")
    parser.add_argument("--provider", choices=("auto", "openai", "grok"), default=None)
    parser.add_argument("--model", default=None, help="Vision model; overrides OPENAI_MODEL or XAI_MODEL")
    parser.add_argument("--reasoning-effort", choices=("auto", "low", "medium", "high", "xhigh"), default=None)
    parser.add_argument("--search", choices=("auto", "off"), default=None, help="Bounded automatic research or disable research")
    parser.add_argument("--history-images", choices=("latest", "all"), default=None,
                        help="Client history: newest photo only, or replay all past photos")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--camera-name", default=CAMERA_NAME, help="Initial Windows camera name")
    parser.add_argument("--camera-resolution", choices=("auto", "1920x1080", "1280x720", "640x480"), default="1920x1080")
    parser.add_argument("--once-text", help="One fixture turn without microphone or keyboard; needs --image and stage >= 3")
    rig = parser.add_argument_group("Fixpoint laser rig (points with the laser and supplies the camera)")
    rig.add_argument("--rig", action="store_true", help="Use the laser rig as camera and pointer")
    rig.add_argument("--rig-port", default="COM6", help="ESP32 serial port (default: COM6)")
    rig.add_argument("--rig-camera", type=int, default=1, help="Rig camera index (default: 1)")
    rig.add_argument("--rig-calibration", type=Path, help="Calibration file (default: fixpoint's calibration.json)")
    rig.add_argument("--rig-photo-size", choices=("1920x1080", "1280x960", "1280x720", "640x480"), default="1920x1080",
                     help="Resolution of the photo sent to the model; tracking stays at 640x480 (default: 1920x1080)")
    rig.add_argument("--rig-no-window", action="store_true", help="Do not open the tracker debug window")
    rig.add_argument("--rig-sim", action="store_true", help="Simulated rig: no camera or gimbal needed")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1 or args.pointer_timeout <= 0:
        parser.error("Confidence must be in [0,1] and pointer timeout must be positive")
    args.rig = args.rig or args.rig_sim
    if args.rig and args.pointer:
        parser.error("--rig is the pointer; do not combine it with --pointer")
    if args.once_text and (args.image is None or args.stage < 3):
        parser.error("--once-text requires --image and --stage >= 3")
    return args


class TextRecorder:
    def start(self): pass
    def stop(self): return Path("unused.wav")
    def abort(self): pass


class TextSpeech:
    def __init__(self, text, speech):
        self.text, self.speech = text, speech
    async def transcribe(self, path): return self.text
    async def speak(self, text): await self.speech.speak(text)


async def run(args):
    values = load_settings(ROOT / ".env")
    rig = None
    if args.rig:
        from .rig import LaserRig
        print("Starting the laser rig...", flush=True)
        rig = await asyncio.to_thread(LaserRig, port=args.rig_port, camera_index=args.rig_camera,
                                      calibration=args.rig_calibration, sim=args.rig_sim,
                                      window=not args.rig_no_window,
                                      photo_size=tuple(map(int, args.rig_photo_size.split("x"))))
    try:
        return await run_app(args, values, rig)
    finally:
        if rig is not None:
            await asyncio.to_thread(rig.close)


async def run_app(args, values, rig):
    async with httpx.AsyncClient() as client:
        model = create_model(client, values, provider=args.provider, model=args.model,
                             reasoning_effort=args.reasoning_effort, search=args.search,
                             history_images=args.history_images, state_mode=args.state_mode)
        speech = Speech(client, values.get("ELEVENLABS_API_KEY") or "",
                        (values.get("ELEVENLABS_VOICE_ID") or "").strip() or DEFAULT_VOICE_ID,
                        args.output_device)
        if rig is not None and args.image is None:
            # The rig owns the camera: photos come from it, in the pixels it steers by.
            from .rig import RigCamera
            camera = RigCamera(rig, args.artifacts)
        else:
            camera = Camera(args.artifacts, args.image, camera_name=args.camera_name,
                            resolution=None if args.camera_resolution == "auto" else tuple(map(int, args.camera_resolution.split("x"))))
        if rig is not None:
            from .rig import RigPointer
            pointer, pointer_label = RigPointer(rig), rig.label
        else:
            pointer, pointer_label = load_pointer(args.pointer), args.pointer or "Dry run — simulated completion"
        operator = Operator(
            TextRecorder() if args.once_text else Recorder(args.artifacts, args.input_device),
            camera,
            TextSpeech(args.once_text, speech) if args.once_text else speech,
            model,
            pointer, stage=args.stage,
            min_confidence=args.min_confidence, pointer_timeout=args.pointer_timeout,
        )
        if args.stage >= 6 and args.pointer is None and rig is None:
            print("Pointer mode: DRY RUN (simulated completion)", flush=True)
        if args.once_text:
            operator.space_down()
            operator.space_up()
            await operator.wait_until_idle()
            failed = operator.last_error is not None
            await operator.close()
            return 1 if failed else 0
        from .ui import DesktopApp
        app = DesktopApp(operator, pointer_label=pointer_label,
                         keys_configured=(bool(model.api_key), bool(values.get("ELEVENLABS_API_KEY"))))
        await app.run()
        return 0


def main():
    try:
        return asyncio.run(run(arguments()))
    except (CredentialError, PermissionError, RigError) as error:
        # pythonw has no console; make storage errors visible without logging keys.
        from tkinter import Tk, messagebox
        root = Tk()
        root.withdraw()
        title = "Unable to start the laser rig" if isinstance(error, RigError) else "Unable to load secure settings"
        messagebox.showerror(title, str(error), parent=root)
        root.destroy()
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
