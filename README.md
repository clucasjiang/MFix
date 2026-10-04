# MFix: AI-guided electronics repair

Final MHacks 2026 project. Describe a device problem while holding SPACE; the
assistant captures a fresh camera image, uses a multimodal OpenAI or Grok API to
generate a repair step and identify a component, and speaks the instruction using
ElevenLabs while an ESP32-controlled laser gimbal points at the selected part.
Conversation context is retained across troubleshooting turns.

This repository contains the final integrated assistant, laser tracker, ESP32-S3
firmware, and offline tests. The assistant is in
`LLM-pthysical-fix-helper-main/`; the physical tracking system is in `fixpoint/`.

## What is intentionally excluded

Captured photos and audio, runtime logs, virtual environments, firmware build
output, editor settings, archives, and machine-specific calibration are not
included. API keys are entered through **Model settings** and stored in Windows
Credential Manager. `.env.example` contains nonsecret settings only.

Calibrate your own rig with **C** in the tracker window before aiming; calibration
is generated locally and depends on your camera and laser placement. For a
hardware-free demonstration, use the simulator commands below; the full voice
assistant still requires model and ElevenLabs API access.

## Laser tracking

Select a rectangle on the live camera view; the gimbal steers the laser dot
into it and reports **VALID** once the camera sees the dot inside.

## Setup

1. Flash the firmware in `firmware/` (PlatformIO **Upload**, or
   `pio run -t upload` from that folder). It is the original firmware plus:
   - 14-bit servo PWM, written directly with `ESP32PWM`: the `Servo` class
     only resolves 1.76 degrees, so 4 in 10 whole-degree commands never moved
     the servo. Its `setTimerWidth()` cannot fix that on the ESP32-S3: it
     re-attaches the pin and puts both servos on the same output, so yaw
     follows pitch;
   - sub-degree moves (`15 05 HI LO` yaw, `15 06 HI LO` pitch, in hundredths
     of a degree), a speed command (`15 07 dps`) and a ping (`15 08 00`).

   The old 3-byte commands are unchanged, so `control_gimbal.py` still works.
   The Python side detects the new firmware and falls back to whole degrees
   on the old one.
2. `python -m pip install -r requirements.txt`

## Run

```powershell
python laser_tracker.py            # COM6, camera 1
python laser_tracker.py --sim      # no hardware: simulated rig
```

On first use press **C** to calibrate (about 20 s, no user steps). The result
is saved to `calibration.json` and checked automatically on the next start;
recalibrate whenever the tripod or camera moves noticeably.

| Input | Action |
| --- | --- |
| Drag, or click two corners | Set the target rectangle (right-click cancels) |
| X | Clear the target / cancel calibration |
| C | Calibrate |
| B | Find the dot by blinking the laser |
| L or Space | Laser on/off |
| Arrows / WASD (Shift: x5) | Jog the gimbal 1 degree |
| `[` `]` | Exposure down/up |
| T | Auto-pick the exposure where the dot stands out most |
| M | Show the detector's signal map |
| G | Show calibration points |
| P | Save a snapshot of the debug view to `snapshots/` |
| O | Camera driver settings dialog |
| Q / Esc | Quit (laser off) |

| Status | Meaning |
| --- | --- |
| valid | Dot seen inside the rectangle for 5 frames |
| predicted | Dot not visible (hole, glare, edge), but the model puts it inside |
| lost | Dot not visible and not predicted inside |
| failed | Dot visible but could not be brought inside (target smaller than one step, or angle limit) |
| unreachable | Target outside the calibrated area |

Whenever the dot stops at a target, even just off it, it blinks three times
in place (0.5 s off, 0.5 s on), so you can tell it has stopped rather than
still moving. After a valid result the target is held: if the dot drifts out
for 0.5 s (bumped tripod), it re-aims and blinks again. Every result is appended to `aim_log.csv`.

## Voice assistant (the full loop)

The assistant in `LLM-pthysical-fix-helper-main/` drives this rig: you hold
SPACE and describe the problem, it takes a photo, the model answers with a spoken
step and a box, and the laser points at that box while the step is spoken.

One-time setup, from `LLM-pthysical-fix-helper-main/`:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then close `laser_tracker.py` (the assistant needs the camera and COM6) and run
`.\assistant.cmd` from this folder (PowerShell needs the `.\`), or double-click
it. Enter the OpenAI and ElevenLabs keys under **Model settings** the first time.
Keys are stored by Windows, not in files.

1. **Hold SPACE** in the assistant window and speak. The rig clears the old
   target, switches the laser off, and takes a brighter 1920x1080 photo (about
   2 s, while you talk). Tracking pauses for that time and stays at 640x480.
2. **Release.** ElevenLabs transcribes; the model gets the transcript and the
   photo, and returns the spoken step plus a box in that photo's pixels. The rig
   converts the box to its 640x480 view, which is the middle 1440x1080 of the
   photo; the extra width at the sides is outside the laser's reach.
3. **Speak and point.** The step is spoken while the rig steers the dot to the
   middle of the box. When it stops it blinks three times (0.5 s off, 0.5 s
   on), so a dot still travelling is never mistaken for one that has arrived.
   The turn ends when both are done; the dot stays on the part until your next
   question. Pointer and camera problems never interrupt the conversation:
   they are printed in the console, and the session carries on.

The tracker window opens next to the assistant for calibration (C) and
troubleshooting; Space there does not toggle the laser.
`.\assistant.cmd --rig-sim` runs everything against the simulated rig.

## How it works

- **Camera** (`fixpoint/camera.py`): exposure, gain, white balance and focus
  are locked, and frames are read on a thread with timestamps.
- **Detection** (`fixpoint/detector.py`): each frame is compared with a
  laser-off reference, so red parts, steady LEDs and glare cancel out. The
  threshold scales with the frame's measured noise, including flicker
  spikes. The search starts near the predicted position.
- **Blinking** (`fixpoint/rig.py`): toggling the laser and differencing
  frames finds the dot when tracking is unsure. With several toggles, a
  pixel counts only if it brightened every time, which rejects blinking
  LEDs and moving hands. The camera delay is measured at startup and used
  for all of this timing.
- **Model** (`fixpoint/model.py`, `fixpoint/calibration.py`): calibration
  steers the dot to a 7x5 grid of image points and fits a cubic map from
  angles to pixels and back. Lens distortion and the camera-to-laser offset
  are absorbed by the fit.
- **Aiming** (`fixpoint/aim.py`): jump to the model's angles for the
  rectangle center, wait for the dot to settle, measure, then correct with a
  Newton step using the model's Jacobian. Residual drift is learned as an
  offset.

## Troubleshooting

- **No dot at startup:** jog it onto the surface, then press B.
- **Dot missed on some surfaces:** press T with the dot on that kind of
  surface, or check M. The dot should be the only bright spot.
- **Saved calibration "off by N px":** press C.

## Tests

```powershell
python -m unittest tests.test_units        # about 1 s
python -m unittest tests.test_end_to_end   # about 40 s, real-time simulator
```

## From code

```python
from fixpoint.aim import Rect
controller.submit("aim", Rect(x1, y1, x2, y2))   # or Aimer.aim(rect) -> AimResult
```

## Team

| Member | GitHub |
|---|---|
| Rongxin Zhang | [@kuyono530rx-droid](https://github.com/kuyono530rx-droid) |
| Junqian Li | [@lijunqian0818-ai](https://github.com/lijunqian0818-ai) |
| Congyi Jiang | [@clucasjiang](https://github.com/clucasjiang) |
| Mingyuan Zhang | [@ericissleeping](https://github.com/ericissleeping) |
