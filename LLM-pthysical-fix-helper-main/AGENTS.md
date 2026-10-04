# Helper: initialization and development guide for AI agents

This repository is the Windows desktop repair helper. Read this file first, then
README.md for detailed behavior. The older `MHacks 2026 AI Guided Pointer Build Plan.md`
is historical; current source and README describe the implemented application.
Sibling Li/MHack projects have different camera and coordinate behavior.

## 1. Initialize the Python runtime

Run commands from this repository's root in PowerShell. Use Windows and Python
3.10+ with Tkinter. Reuse an existing working `.venv`; inspect failures before
recreating an environment or replacing configuration.

For a fresh checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.main --help
```

FFmpeg comes from `imageio-ffmpeg`; an additional system FFmpeg installation is
not required. The desktop app uses native Tkinter and requires no HTTP server,
Codex session, or agent framework.

## 2. Initialize API keys securely

The normal user workflow is: launch `start_app.cmd`, open **Model settings**, choose
OpenAI or Grok, enter that provider's key and the ElevenLabs key, then click
**Save & start new session**. Leave key fields blank to retain stored keys.
Users do not need to configure Windows Credential Manager manually.

`src/credentials.py` uses the native current-user Windows Credential Manager.
Credentials persist for this Windows account on this computer under these targets:

| Service | Credential target |
| --- | --- |
| OpenAI | `MHacks.Helper/OPENAI_API_KEY` |
| Grok / xAI | `MHacks.Helper/XAI_API_KEY` |
| ElevenLabs STT and TTS | `MHacks.Helper/ELEVENLABS_API_KEY` |

- `.env` and `.env.example` contain **nonsecret settings only**. Never put API keys
  in source, templates, documentation, Git, logs, or chat. Inspect presence rather
  than printing secret values; do not ask users to paste keys into an AI conversation.
- Use `load_settings`, `save_settings`, and `WindowsCredentials` from
  `src/credentials.py`; do not introduce plaintext credential writes.
- Startup migrates existing `.env` keys, including the `CHAT_GPT_KEY` alias,
  verifies retrieval, and atomically removes their file entries. If verification
  or storage fails, preserve the source and report the error. Conflicting saved
  and legacy keys require resolution; do not silently overwrite or discard them.
- Explicit process environment variables override saved values. They are an
  advanced integration option, not the normal onboarding workflow, and are not
  persisted. `CHAT_GPT_KEY` is accepted as an OpenAI alias.
- A copied checkout does not transfer credentials. Each user enters their own
  keys through Settings. Moving the folder under the same account retains access
  because credential targets do not depend on the checkout path.
- Voice listing needs ElevenLabs `voices_read`; missing that permission does not
  necessarily prevent transcription or playback. A voice ID can be entered directly.

Nonsecret defaults are defined in `.env.example` and `src/model_backend.py`:
`MODEL_PROVIDER=auto` selects OpenAI if its key is present, otherwise Grok.
OpenAI currently defaults to `gpt-6.1-sol` with low reasoning; Grok defaults to
`grok-4.7` with medium reasoning. Model availability depends on the user's account;
provider failures never trigger a silent switch. Search defaults to `auto`, bounded
to one tool call; `off` disables research. CLI flags override startup settings.

## 3. Initialize camera and audio devices

In the desktop sidebar, click **Refresh devices & voices**, select the intended
**Camera**, **Microphone**, **Capture resolution**, and voice. Apply changes while
idle. Do not silently replace a disconnected selection with another device.

- Camera discovery uses Windows DirectShow. Duplicate names are distinguished
  using their unique device paths; keep the visible label with the chosen path.
- The startup camera is `UGREEN Camera 1080P` at 1920x1080. That known mode uses
  MJPEG at 30 fps. A newly selected different camera uses **Auto** resolution and
  negotiates its input format. Choose only a fixed resolution that device supports.
- A 720p webcam cannot open a forced 1080p mode. Use Auto or 1280x720. On failure,
  check the selected device/mode, other apps using it, and Windows camera permissions.
  Do not assume Li's mode probing or padded canvas is implemented in helper.
- Camera/microphone/voice selections made in the sidebar apply to the current
  process. For reproducible startup, pass CLI device options; do not assume those
  selections persist in `.env`.
- Microphone and playback indices are sounddevice indices. List them using
  `.\.venv\Scripts\python.exe -m sounddevice`; do not hardcode another machine's indices.
- Refresh lists devices and ElevenLabs voices but does not record or capture.
  Holding **Space** or **Hold to talk** records and captures a fresh photo.
  Release starts transcription/diagnosis. Escape/Stop interrupts; focus loss during
  recording aborts it. Accept no overlapping turns.

Example startup for a different webcam (the camera name must match discovery):

```powershell
.\.venv\Scripts\python.exe -m src.main --camera-name "USB2.0 HD UVC WebCam" --camera-resolution auto
```

## 4. Initialize image coordinates and the pointer boundary

Each turn sends the **clean current image**, without a grid, to the selected model.
`src/camera.py` reads width and height from the actual saved image. `src/grok.py`
supplies those dimensions and the coordinate convention explicitly in the request.
Never infer resolution from configuration, resize/rotate images sent to the model,
or use UI preview coordinates as camera coordinates.

- Origin: top-left `(0, 0)`; X increases rightward, Y downward; units are pixels.
- Model output: `bbox = {x1, y1, x2, y2}`, two opposite corners with inclusive
  endpoints. Require integer `0 <= x1 < x2 < width` and
  `0 <= y1 < y2 < height`. Unknown targets have `target_present=false` and null
  label, box, and confidence.
- `src/operator.py` validates geometry and the confidence threshold (default 0.7).
  Rejected targets produce clarification without pointing. Never clamp or repair
  invalid coordinates. Model confidence is not a measured accuracy guarantee.
- Valid output is forwarded unchanged as `send_bbox(x1, y1, x2, y2)`.
  Helper does not calculate a target center, motor angles, homography, or calibration.
- Physical homing, zero positions, calibration, and pixel-to-hardware conversion
  belong to the separate downstream pointer implementation. An image coordinate
  is not a calibrated laser coordinate or a meter probe contact.
- Default pointer mode is an explicit **dry run**. A real adapter is loaded using
  `--pointer package.module:factory`. Its zero-argument factory must return async
  `send_bbox`, `wait_until_complete`, and `cancel` methods as defined in `src/pointer.py`.
- READY requires speech playback and pointer completion. Pointer failures
  (including timeouts) are logged to the console, reported to the model, and the
  turn carries on; keep them out of the conversation panel. After an Escape
  interrupt, do not bypass the ERROR lock: recovery requires an explicit
  stopped/idle acknowledgement from `cancel()` plus successful microphone cleanup.
  Do not retry physical dispatch automatically.

## 5. Validate initialization without spending API credits

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m pip check
```

Tests use dummy keys, fixture images, mock provider responses, and simulated
hardware; they do not call paid endpoints or move a physical pointer. Native
credential checks should use a separate temporary namespace and remove only the
test entry afterward. Never enumerate, delete, or overwrite unrelated credentials.

For an explicitly requested live image/model check without a microphone, speech
playback, or physical pointing:

```powershell
.\.venv\Scripts\python.exe -m src.main --stage 4 --image C:\path\to\fixture.jpg --once-text "Identify the visible connector." --search off
```

That command **calls the selected model API**. It avoids live camera access but
still uploads the fixture. Report live hardware/API results separately from
offline tests; do not claim real-board accuracy from synthetic examples.

## 6. Preserve runtime invariants during changes

`src/operator.py` owns turn admission, recording/capture overlap, cleanup,
completion, and session state. `src/ui.py` owns Tk events; do not block the UI with
network or device work. Session memory lasts for the current process; a new
session clears dialogue state, not physical calibration. Model/provider changes
start a new session so encrypted output and response IDs cannot be mixed.
Client state defaults to the latest image plus prior text/full provider outputs;
historical coordinates are not evidence of the current scene.

Keep `.env`, `.venv`, captured images/audio in `data/`, and temporary
`tests/test-artifacts-*` out of Git. Preserve existing user files and configuration.
Explain repairs plainly, distinguish observations from hypotheses, and never
infer a hidden electrical fault from appearance alone.
