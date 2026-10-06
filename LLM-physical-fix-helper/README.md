# Multimodal repair operator

Small Python operator for the MHacks push-to-talk diagnostic loop. Select OpenAI
or Grok through the same Responses workflow: clean image, JSON schema, bounded
provider-side `web_search`, and multi-turn state. No agent framework.

AI agents initializing or modifying this repository should read [AGENTS.md](AGENTS.md).

## Run on Windows

Use Python 3.10+ with Tkinter (included in ordinary Python Windows installers).
From this repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m src.main
```

Open **Model settings**, choose OpenAI or Grok, paste its API key and your
ElevenLabs key, and click **Save & start new session**. The app saves keys securely
using Windows Credential Manager. No extra package, Windows setup, or manual file
editing is needed. Saved keys work when you reopen the app under the same Windows
account on this computer. Leave a key field blank to retain its saved value.

Keep the operator window focused. Hold **SPACE** to record; release it to diagnose.
The microphone opens on key-down and the camera captures/processes one image in
the background during the hold. After release, STT overlaps any remaining image
processing. Key repeat and presses during busy states cannot start another turn.
The desktop app displays the current camera capture and returned target box next
to a conversation containing your transcript and the selected model's spoken guidance. The
target card shows the selected component, confidence, and original pixel box.
An optional Activity tab contains detailed results and errors.

## Model choice and response speed

Open **Model settings** in the main window to choose **OpenAI** or **Grok**, set
the model, select reasoning effort, and enable/disable online search. You can
enter a provider key and ElevenLabs key there, or leave a masked field blank to keep its saved key.
Saving starts a new repair session; provider response IDs and encrypted reasoning
cannot be mixed between models. Only nonsecret settings save to `.env`; keys go
directly to Windows Credential Manager. Actual process environment variables
override saved values for advanced use and are never persisted by the app.

Existing `.env` keys (including `CHAT_GPT_KEY`) migrate automatically on startup.
The app reads back each stored key to verify it before atomically removing key
entries from the file. On failure, it preserves the source file and reports an
error; it never falls back to writing plaintext keys. If a legacy file key differs
from a securely saved key, resolve that conflict before continuing.
This does not remove copies in backups, Git history, or other project folders.

- `MODEL_PROVIDER=auto` (default): use OpenAI when its securely saved key or an
  explicit `OPENAI_API_KEY`/`CHAT_GPT_KEY` process override is present; otherwise select Grok.
  There is no automatic fallback after a provider fails.
- OpenAI defaults to **gpt-6.1-sol**, **low** reasoning. Set `OPENAI_MODEL` and
  `OPENAI_REASONING_EFFORT` to change them. Use medium/high when a difficult
  diagnosis needs more reasoning, then evaluate the tradeoff on your board.
- Grok defaults to **grok-4.7**, **medium** reasoning rather than the API's high
  default. Set `XAI_MODEL` and `XAI_REASONING_EFFORT`. `auto` omits effort and uses
  the provider default. Known non-reasoning model families omit the effort field.
- `{OPENAI,XAI}_SEARCH=auto` offers web search, limited to one tool call per request;
  the prompt asks the model to skip lookup for obvious visual questions. `off`
  removes research for a quicker visual-only demonstration. This limits tool calls,
  not the provider's total wall-clock time.
- `{OPENAI,XAI}_HISTORY_IMAGES=latest` (default) sends only the current image in
  client-managed requests. Earlier text, replies, research and encrypted outputs
  remain in history; historical coordinates are not current evidence. `all` replays
  past photographs too when visual comparison is essential. Stored mode still
  lets the provider retain its earlier images. No image is resized or rotated.
- Each session supplies a stable prompt-cache key. Editing earlier image messages
  can reduce cache reuse; this does not guarantee a cache hit or fixed latency.
- The bottom timing line reports transcription, model and output time after release.
  The Activity tab adds capture time, request size, reasoning tokens, search calls
  and cache usage when the provider returns them. Capture and STT overlap, as do
  speech and pointer output, so stage times should not simply be added together.

Example: a quick OpenAI visual-only run.

```powershell
.\.venv\Scripts\python.exe -m src.main --provider openai --model gpt-6.1-sol --reasoning-effort low --search off
```

For comparison: `--provider grok --model grok-4.7 --reasoning-effort medium`.
CLI flags override saved settings for startup. The window's settings can then
change the active provider/model for a new session.

Select a **Microphone**, **Camera**, and **Capture resolution** in the left sidebar.
**Refresh devices & voices** discovers newly connected devices and lists available
ElevenLabs voices without recording. Device
changes apply to the next turn and are disabled while busy. Choose **Auto** for a
camera's default resolution, or select a fixed resolution it supports. Changing
to a different camera selects Auto; the known UGREEN camera defaults to 1080p.
Disconnected selections are marked unavailable rather than silently replaced.
An image fixture disables the camera controls.

You can also press and hold the blue **Hold to talk** button with the mouse.
Press **New session** or **Ctrl+R** while READY for a new session. **Escape** or
**Stop** interrupts the current turn
or requests a stopped/idle acknowledgement after a pointer failure. Losing window
focus during recording aborts that recording. Close the window to shut down.

After dependency setup, double-click `start_app.cmd` to open the desktop app
without a persistent terminal window. The same launcher accepts the CLI options
below when invoked from PowerShell.

The default pointer is an explicitly labeled **dry run** with simulated completion.
Supply `--pointer package.module:factory` to use your separate physical subsystem.
The module must be importable in this Python environment; the factory takes no
arguments. Environment variables may configure the downstream adapter.

## Incremental development

```powershell
python -m src.main --stage 1
python -m src.main --stage 2
python -m src.main --stage 4 --image C:\path\to\board.jpg
python -m src.main --stage 8 --pointer my_pointer.adapter:create_pointer
```

| Stage | Enabled behavior |
| --- | --- |
| 1 | SPACE hold/release, microphone, ElevenLabs STT, print transcript |
| 2 | Add clean camera capture and concurrent image processing |
| 3 | Add selected model's image + transcript request, validate and print structured response |
| 4 | Save a separate `debug_bbox.png` next to each clean capture |
| 5 | Add ElevenLabs TTS and wait for playback to finish |
| 6 | Forward four unchanged pixel coordinates to the pointer adapter |
| 7 | Exercise downstream completion handling (also mandatory at stage 6) |
| 8 | Retain multi-turn diagnostic state within a session (default) |

Stages 1–2 need the ElevenLabs key. Stage 3–4 also need the selected model provider's key. Stage 5+
use voice `NOpBlnGInO9m6vDvFkFC` by default; `ELEVENLABS_VOICE_ID` overrides it.
An image fixture avoids the live camera at stages 2+.
For a single vision test without microphone, keyboard, or TTS:

```powershell
python -m src.main --stage 4 --image C:\path\to\board.jpg --once-text "Which cable should I check?"
```

`--input-device N` and `--output-device N` select sounddevice device indices;
list them with `python -m sounddevice`. These are also selectable in the UI.
`--camera-name "Camera name"` and `--camera-resolution auto` set the initial
camera configuration. `--artifacts PATH` selects the output folder.

Speech uses the ElevenLabs Python SDK's async client to keep the turn loop
responsive. STT calls `speech_to_text.convert` with `model_id="scribe_v2"`,
`tag_audio_events=True`, `language_code=None` (auto detection), and `diarize=True`.
The returned transcript text goes to the selected model. TTS calls `text_to_speech.convert`
with `model_id="eleven_v4"`, `language_code="en"`, and the configured voice ID.
`output_format="pcm_16000"` supplies raw audio for local sounddevice playback;
the turn waits for playback completion, and interruption stops playback.

## Camera and coordinates

`src/camera_capture.py` is adapted from the capture code of an earlier team
prototype: Windows DirectShow/FFmpeg capture, exposure settling, bounded capture,
and atomic publication. Camera name and resolution are configurable. The original
UGREEN 1080p/MJPEG settings remain available; other cameras negotiate their input
format.

The processed input remains a clean image. Image dimensions come from the saved
file that is sent to the model, including fixtures at arbitrary resolutions. Each
request includes the actual width/height and bottom-right pixel. Debug boxes are
drawn on a separate copy; neither the clean input nor pointer coordinates change.

Boxes use **inclusive pixel endpoints**:
`0 <= x1 < x2 < width`, `0 <= y1 < y2 < height`. Integers are strict: strings,
booleans, floats, missing fields, degenerate/outside boxes, and inconsistent target
fields are rejected. Invalid or low-confidence targets become spoken clarification
without a pointer call; coordinates are never clamped or repaired.
The default minimum confidence is 0.7 (`--min-confidence`). Actual grounding
accuracy still needs evaluation on your board using the saved debug images.

## Pointer adapter contract

The downstream implementation owns all hardware, geometry, calibration, laser
detection, and motion. Implement the async interface in `src/pointer.py`:

```python
async def send_bbox(self, x1: int, y1: int, x2: int, y2: int) -> None:
    # Pass these four pixels into your existing pointer program.
    ...

async def wait_until_complete(self) -> None:
    # Return only after COMPLETE / ALIGNED; raise on failure.
    ...

async def cancel(self) -> bool:
    # Request a stop and return True only after a stopped/idle ACK.
    ...
```

Adapter methods must be cancellable and must not block the asyncio loop. Wrap
blocking downstream APIs appropriately in the adapter; Python coroutine
cancellation alone is not proof that physical movement stopped.
The AI never calculates a center, changes coordinates, or issues hardware commands.
Only one box is dispatched per turn; there are no automatic dispatch retries.

TTS and pointing run concurrently. READY requires both playback and physical
completion. `--pointer-timeout` bounds sending plus waiting (default 30 seconds).
Any pointer failure (`PointingFailed`, a timeout, a send error) is printed in
the console and noted for the model ("the laser is not marking it"); nothing is
shown in the conversation and the turn ends READY. A new aim or photo
interrupts anything still running downstream. Interrupting with Escape after
dispatch is different: **ERROR** blocks further turns and session resets until
an explicit True stopped/idle acknowledgement and successful microphone cleanup.
STT, API, or schema failures produce no movement.

If the camera fails, the turn carries on with the previous photo and the model
is told it may be out of date; the error itself goes to the console. Only the
first photo of a session has nothing to fall back on.

## Fixpoint laser rig

`--rig` uses the Fixpoint rig next to this repository (`../fixpoint` and
`../laser_tracker.py`, or the folder in `FIXPOINT_ROOT`) as both camera and
pointer. `../assistant.cmd` runs this with the `.venv` below.

```powershell
python -m venv --system-site-packages .venv   # reuses the global OpenCV/numpy/pyserial
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main --rig        # COM6, camera 1
.\.venv\Scripts\python.exe -m src.main --rig-sim    # simulated rig, no hardware
```

- **One camera owner.** The rig keeps the camera open for laser tracking, so
  each turn's photo is taken through it (`src/rig.py`, `RigCamera`). The
  current target is cleared and the laser is switched off. The camera is then
  briefly reopened at 1920x1080 (MJPEG) with the exposure two steps brighter,
  two frames are averaged into a JPEG (quality 95), and tracking resumes at
  640x480. This takes about 2 s, while the user is still talking.
  `--rig-photo-size` picks another size; 640x480 skips the reopening.
- **Coordinates.** The tracker's 4:3 view is the middle 1440x1080 of the 16:9
  photo, scaled by 4/9 (`fixpoint.camera.view_in_photo`, measured to within
  about 1 px). `RigPointer` converts each box with that mapping; boxes in the
  side strips are outside the laser's reach and come back as unreachable.
- **Pointing.** `RigPointer` steers the dot to the middle half of the box and
  completes when the camera sees it there, or anywhere within 25 tracker pixels
  of the box if it cannot settle (tiny parts). Where the camera cannot see the
  dot (dark or glossy plastic), completion relies on the calibration
  ("predicted"). Whenever the dot stops, the controller blinks it three times
  in place (0.5 s off, 0.5 s on), including targets dragged in the tracker
  window and re-aims after a bump, so a dot still moving between parts never
  blinks. The turn completes after the blink. It then holds on the part,
  re-aiming if the rig is bumped, until the next photo turns it off. Every aim
  is logged to `../aim_log.csv`. If a full-size photo fails, the rig falls back
  to a 640x480 tracking frame.
- **Failures.** Not calibrated, outside the calibrated area, or stopping far
  from the box raise `PointingFailed`, which stays out of the conversation as
  described above.
  Escape/Stop cancels aiming and acknowledges once the servos have stopped.
- **Tracker window.** The debug view opens alongside (`--rig-no-window` hides
  it). Calibrate there with C, or beforehand with `python laser_tracker.py`;
  `--rig-calibration` selects another file. Space does not toggle the laser in
  that window, and `laser_tracker.py` must be closed while the app runs.
- `--rig-port` and `--rig-camera` change the ESP32 port and camera index.

## Session memory

Default `--state-mode client` uses `store=false` and replays conversation text
and full provider output items, including research and encrypted reasoning.
The default latest-image setting omits previous image blocks from requests.
Instructions and a fresh image accompany every turn. Text/output history still
grows with the session; reset when beginning a separate repair.

Optional `--state-mode stored` uses `store=true` and `previous_response_id`.
For Grok, instructions are supplied on the first request only: xAI rejects requests
that combine `instructions` with `previous_response_id`. OpenAI resends instructions
on every request because they do not carry over through the response ID. New session / Ctrl+R clears
local history and the response ID without deleting provider-retained records.
There is no automatic retry or silent switch of modes.

The voice selector accepts a listed voice or a voice ID (press Enter to apply).
Listing voices requires `voices_read` permission on the ElevenLabs API key;
transcription and playback can still work without that permission. Library voices
may require a paid subscription. Provider errors appear as readable messages,
without dumping response headers into the conversation.

Sessions persist across turns in the current process, not across app restarts.
Output failures/rejected targets become application feedback in the next turn so
the model does not assume a previous instruction or movement succeeded.

## Verification

Live camera/audio/model grounding and the physical adapter must be tested on your
actual devices. `--rig-sim` exercises the full turn against the simulated rig
without a camera or gimbal (model and ElevenLabs keys are still required).

Live OpenAI smoke checks on a generated
red/blue image confirmed authentication, structured replies and a two-turn client
conversation (about 3.8s and 2.6s for those simple requests, search off). Those
timings are not a repair-board benchmark or a Grok comparison. A third request
with automatic search available returned the exact blue rectangle bounds in
about 3.3s without invoking search. No physical pointer was connected for these checks.

API references: [OpenAI model](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[Windows credential storage](https://learn.microsoft.com/en-us/windows/win32/secbp/handling-passwords),
[OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[xAI structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs),
[image understanding](https://docs.x.ai/developers/model-capabilities/images/understanding),
[Responses state](https://docs.x.ai/developers/model-capabilities/text/comparison),
[provider web search](https://docs.x.ai/developers/tools/web-search),
[ElevenLabs STT](https://elevenlabs.io/docs/api-reference/speech-to-text/convert),
[ElevenLabs TTS](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
