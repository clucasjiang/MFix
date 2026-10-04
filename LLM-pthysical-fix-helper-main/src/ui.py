"""Desktop conversation, capture preview, and idle-only device selection."""
import asyncio
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk


from PIL import Image, ImageTk

from . import camera_capture
from .devices import CameraDevice, list_cameras, list_microphones
from .credentials import CredentialError, load_settings, save_settings
from .models import State
from .model_backend import create_model

BACKGROUND = "#0b1220"
PANEL = "#111d30"
FIELD = "#1b2b42"
TEXT = "#edf3fc"
MUTED = "#9bacc4"
ACCENT = "#55d4c4"
BLUE = "#3378ed"
RESOLUTIONS = {"Auto (camera default)": None, "1920 × 1080": (1920, 1080),
               "1280 × 720": (1280, 720), "640 × 480": (640, 480)}
STATE_LABELS = {
    State.READY: "Ready to listen", State.RECORDING: "Listening — release to send",
    State.PROCESSING: "Preparing your voice and image", State.WAITING_FOR_GROK: "Grok is diagnosing…",
    State.OUTPUTTING: "Speaking the next step", State.WAITING_FOR_POINTER: "Waiting for pointing to finish",
    State.ERROR: "Device stop needs confirmation", State.STOPPED: "Stopped",
}


class DesktopApp:
    def __init__(self, operator, *, pointer_label="Dry run", keys_configured=(False, False), root=None):
        self.operator = operator
        self.root = root if root is not None else tk.Tk()
        self.root.title("Repair assistant")
        self.root.geometry("1240x800")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BACKGROUND)
        self.closing = False
        self._held = None
        self._admitted = False
        self._refreshing = False
        self._tasks = set()
        self._source_image = None
        self._photo = None
        self._mic_choices = {"System default microphone": None}
        self._camera_choices = {}
        self._voice_choices = {}
        self.state_text = tk.StringVar(value=STATE_LABELS[operator.state])
        self.mic_text = tk.StringVar(value="System default microphone")
        self.camera_text = tk.StringVar(value=operator.camera.camera_name)
        self.voice_text = tk.StringVar(value=getattr(operator.speech, "voice_id", ""))
        self.resolution_text = tk.StringVar(value=next(
            (label for label, size in RESOLUTIONS.items() if size == operator.camera.resolution),
            "Auto (camera default)"))
        self.device_note = tk.StringVar(value="Looking for connected devices…")
        self.image_note = tk.StringVar(value="A fresh image is captured for each turn.")
        self.target_text = tk.StringVar(value="No component selected yet")
        self.model_text = tk.StringVar()
        self.key_text = tk.StringVar()
        self.timing_text = tk.StringVar(value="")
        self._build(pointer_label, keys_configured)
        self._refresh_model_label(keys_configured[1])
        self.operator.notify = self.log
        self.operator.on_event = self.on_event
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.bind("<FocusOut>", self._focus_out)
        self.root.bind("<Escape>", lambda event: self.interrupt())
        self.root.bind("<Control-r>", lambda event: self.new_session())
        # Reserve SPACE before widget class bindings (including comboboxes).
        self.root.bind_class("RepairPushToTalk", "<KeyPress-space>", self._key_down)
        self.root.bind_class("RepairPushToTalk", "<KeyRelease-space>", self._key_up)
        self._reserve_space(self.root)
        self._sync_controls()

    def _build(self, pointer_label, keys_configured):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=FIELD, background=FIELD,
                        foreground=TEXT, arrowcolor=TEXT, padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
                  foreground=[("disabled", MUTED), ("readonly", TEXT)],
                  selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
        style.configure("Repair.TButton", background=FIELD, foreground=TEXT, padding=(12, 9), borderwidth=0)
        style.map("Repair.TButton", background=[("active", "#284265"), ("disabled", PANEL)],
                  foreground=[("disabled", "#63758e")])
        style.configure("Repair.TNotebook", background=PANEL, borderwidth=0)
        style.configure("Repair.TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(15, 10))
        style.map("Repair.TNotebook.Tab", background=[("selected", FIELD)], foreground=[("selected", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BLUE)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(self.root, bg=PANEL, width=270)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(12, weight=1)
        self._label(sidebar, "REPAIR / ASSISTANT", color=ACCENT, size=11, bold=True).grid(row=0, padx=22, pady=(30, 6), sticky="w")
        self._label(sidebar, "Your workbench\ncompanion", size=18, bold=True, wrap=225).grid(row=1, padx=22, pady=(0, 20), sticky="w")
        self._label(sidebar, "MICROPHONE", color=MUTED, size=10, bold=True).grid(row=2, padx=22, pady=(0, 8), sticky="w")
        self.mic_combo = ttk.Combobox(sidebar, textvariable=self.mic_text, state="readonly", width=23)
        self.mic_combo.grid(row=3, padx=22, sticky="ew")
        self.mic_combo.bind("<<ComboboxSelected>>", self._choose_mic)
        self._label(sidebar, "CAMERA", color=MUTED, size=10, bold=True).grid(row=4, padx=22, pady=(14, 8), sticky="w")
        self.camera_combo = ttk.Combobox(sidebar, textvariable=self.camera_text, state="readonly", width=23)
        self.camera_combo.grid(row=5, padx=22, sticky="ew")
        self.camera_combo.bind("<<ComboboxSelected>>", self._choose_camera)
        self._label(sidebar, "CAPTURE RESOLUTION", color=MUTED, size=10, bold=True).grid(row=6, padx=22, pady=(14, 8), sticky="w")
        self.resolution_combo = ttk.Combobox(sidebar, textvariable=self.resolution_text,
                                           values=list(RESOLUTIONS), state="readonly", width=23)
        self.resolution_combo.grid(row=7, padx=22, sticky="ew")
        self.resolution_combo.bind("<<ComboboxSelected>>", self._choose_resolution)
        self._label(sidebar, "VOICE / VOICE ID", color=MUTED, size=10, bold=True).grid(row=8, padx=22, pady=(14, 8), sticky="w")
        self.voice_combo = ttk.Combobox(sidebar, textvariable=self.voice_text, width=23)
        self.voice_combo.grid(row=9, padx=22, sticky="ew")
        self.voice_combo.bind("<<ComboboxSelected>>", self._choose_voice)
        self.voice_combo.bind("<Return>", self._choose_voice)
        self.voice_combo.bind("<FocusOut>", self._choose_voice)
        self.refresh_button = ttk.Button(sidebar, text="Refresh devices & voices", style="Repair.TButton", command=self.refresh_devices)
        self.refresh_button.grid(row=10, padx=22, pady=(14, 10), sticky="ew")
        self._label(sidebar, variable=self.device_note, color=MUTED, size=10, wrap=220).grid(row=11, padx=22, sticky="nw")
        instructions = tk.Frame(sidebar, bg=PANEL)
        instructions.grid(row=13, padx=22, pady=16, sticky="ew")
        self._label(instructions, "HOLD TO TALK", color=ACCENT, size=10, bold=True).pack(anchor="w", pady=(0, 8))
        self._label(instructions, "Hold SPACE or the blue button.\nRelease to send.\nEscape: stop · Ctrl+R: new session", color=MUTED, size=10, wrap=220).pack(anchor="w")

        content = tk.Frame(self.root, bg=BACKGROUND)
        content.grid(row=0, column=1, sticky="nsew", padx=24, pady=24)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)
        header = tk.Frame(content, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self._label(header, "Let's find the next step.", bg=BACKGROUND, size=25, bold=True).pack(anchor="w")
        self.status_label = self._label(header, variable=self.state_text, bg=BACKGROUND, color=ACCENT, size=12)
        self.status_label.pack(anchor="w", pady=(9, 0))
        self._label(header, f"Pointer: {pointer_label}", bg=BACKGROUND, color=MUTED, size=10).pack(anchor="w", pady=(5, 0))
        model_row = tk.Frame(header, bg=BACKGROUND)
        model_row.pack(fill="x", pady=(8, 0))
        self._label(model_row, variable=self.model_text, bg=BACKGROUND, color=MUTED, size=10).pack(side="left")
        self.model_button = ttk.Button(model_row, text="Model settings", style="Repair.TButton", command=self.show_model_settings)
        self.model_button.pack(side="right")

        body = tk.Frame(content, bg=BACKGROUND)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=3, uniform="body")
        body.grid_columnconfigure(1, weight=3, uniform="body")
        body.grid_rowconfigure(0, weight=1)
        scene = tk.Frame(body, bg=PANEL)
        scene.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        scene.grid_columnconfigure(0, weight=1)
        scene.grid_rowconfigure(1, weight=1)
        self._label(scene, "CURRENT SCENE", color=MUTED, size=10, bold=True).grid(row=0, column=0, padx=16, pady=16, sticky="w")
        self.preview = tk.Canvas(scene, bg="#080e18", highlightthickness=0, width=380, height=300)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16)
        self.preview.bind("<Configure>", lambda event: self._draw_preview())
        self._label(scene, variable=self.image_note, color=MUTED, size=10, wrap=360).grid(row=2, column=0, padx=16, pady=14, sticky="w")
        self._label(scene, "VISUAL TARGET", color=ACCENT, size=10, bold=True).grid(row=3, column=0, padx=16, pady=(16, 8), sticky="w")
        self._label(scene, variable=self.target_text, size=12, wrap=360).grid(row=4, column=0, padx=16, pady=(0, 24), sticky="w")

        notebook = ttk.Notebook(body, style="Repair.TNotebook")
        notebook.grid(row=0, column=1, sticky="nsew")
        conversation = tk.Frame(notebook, bg=PANEL)
        activity = tk.Frame(notebook, bg=PANEL)
        notebook.add(conversation, text="Conversation")
        notebook.add(activity, text="Activity")
        self.conversation = self._text_panel(conversation)
        self.conversation.tag_configure("speaker", foreground=ACCENT, font=("Segoe UI", 10, "bold"), spacing1=16, spacing3=5)
        self.conversation.tag_configure("speech", foreground=TEXT, font=("Segoe UI", 14), spacing3=16)
        self.conversation.tag_configure("user", foreground="#c0cee1", font=("Segoe UI", 12), spacing3=16)
        self.conversation.tag_configure("note", foreground=MUTED, font=("Segoe UI", 11), spacing3=12)
        self.conversation.tag_configure("error", foreground="#ffb5ac", font=("Segoe UI", 11), spacing1=12, spacing3=12)
        self.activity = self._text_panel(activity, font=("Consolas", 10))
        self._chat("Welcome. Select your devices, then hold SPACE to ask about the device in front of the camera.\n", "note")

        controls = tk.Frame(content, bg=BACKGROUND)
        controls.grid(row=2, column=0, sticky="ew", pady=(20, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.talk_button = tk.Label(controls, text="Hold to talk  ·  SPACE", bg=BLUE, fg="white",
                                    font=("Segoe UI", 14, "bold"), padx=20, pady=17, cursor="hand2")
        self.talk_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.talk_button.bind("<ButtonPress-1>", self._mouse_down)
        self.talk_button.bind("<ButtonRelease-1>", self._mouse_up)
        self.new_button = ttk.Button(controls, text="New session", style="Repair.TButton", command=self.new_session)
        self.new_button.grid(row=0, column=1, padx=(0, 10))
        self.stop_button = ttk.Button(controls, text="Stop", style="Repair.TButton", command=self.interrupt)
        self.stop_button.grid(row=0, column=2)
        self._label(content, variable=self.key_text, bg=BACKGROUND, color=MUTED, size=10).grid(row=3, column=0, pady=(12, 0), sticky="w")
        self._label(content, variable=self.timing_text, bg=BACKGROUND, color=MUTED, size=10).grid(row=4, column=0, pady=(4, 0), sticky="w")

    @staticmethod
    def _label(parent, text=None, *, variable=None, bg=PANEL, color=TEXT, size=12, bold=False, wrap=0):
        return tk.Label(parent, text=text, textvariable=variable, bg=bg, fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"),
                        justify="left", anchor="w", wraplength=wrap)

    @staticmethod
    def _text_panel(parent, font=("Segoe UI", 12)):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        text = tk.Text(parent, bg=PANEL, fg=TEXT, relief="flat", borderwidth=0,
                       font=font, wrap="word", padx=18, pady=14, state="disabled", width=24, height=12)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        return text

    def _reserve_space(self, widget):
        widget.bindtags(("RepairPushToTalk", *widget.bindtags()))
        for child in widget.winfo_children():
            self._reserve_space(child)

    def _chat(self, text, tag):
        self.conversation.configure(state="normal")
        self.conversation.insert("end", text, tag)
        self.conversation.see("end")
        self.conversation.configure(state="disabled")

    def log(self, message):
        print(message, flush=True)
        self.activity.configure(state="normal")
        self.activity.insert("end", f"{datetime.now():%H:%M:%S}  {message}\n")
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def on_event(self, kind, **data):
        if kind == "state":
            self.state_text.set("Model is inspecting the image…" if data["state"] == State.WAITING_FOR_GROK else STATE_LABELS[data["state"]])
            self.status_label.configure(fg="#ffb5ac" if data["state"] == State.ERROR else ACCENT)
            self._sync_controls()
        elif kind == "turn_started":
            self.target_text.set("Waiting for the next target")
            self.timing_text.set("")
            self.image_note.set("Capturing a fresh image…")
            self._source_image = None
            self._draw_preview()
        elif kind == "transcript":
            self._chat("YOU\n", "speaker")
            self._chat(data["text"] + "\n", "user")
        elif kind == "response":
            response = data["response"]
            self._chat(getattr(self.operator.grok, "name", "Grok").upper() + "\n", "speaker")
            self._chat(response.speech + "\n", "speech")
            if response.target_present:
                box = response.bbox
                self.target_text.set(f"{response.target_label}\nConfidence {response.confidence:.0%}\n"
                                     f"Pixels ({box.x1}, {box.y1}) → ({box.x2}, {box.y2})")
            else:
                self.target_text.set("No pointing target for this reply")
        elif kind == "image":
            frame = data["frame"]
            self.show_image(frame.path)
            self.image_note.set(f"Captured image · {frame.width} × {frame.height} pixels")
        elif kind == "debug_image":
            self.show_image(data["path"])
        elif kind == "error":
            self._chat(data["message"] + "\n", "error")
        elif kind == "timings":
            timing = data["timings"]
            self.timing_text.set(" · ".join(f"{label} {timing[key]:.1f}s" for key, label in
                (("stt_seconds", "Voice"), ("model_seconds", "Model"), ("output_seconds", "Output"),
                 ("total_after_release_seconds", "Total")) if key in timing))
        elif kind == "session_reset":
            self.conversation.configure(state="normal")
            self.conversation.delete("1.0", "end")
            self.conversation.configure(state="disabled")
            self._chat("New diagnostic session. Hold SPACE to begin.\n", "note")
            self.target_text.set("No component selected yet")
            self.image_note.set("A fresh image is captured for each turn.")
            self._source_image = None
            self._draw_preview()

    def show_image(self, path):
        try:
            with Image.open(path) as source:
                self._source_image = source.convert("RGB").copy()
            self._draw_preview()
        except (OSError, ValueError) as error:
            self.log(f"Could not display capture: {error}")

    def _draw_preview(self):
        self.preview.delete("all")
        width, height = max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height())
        if self._source_image is None:
            self._photo = None
            self.preview.create_text(width / 2, height / 2, text="Your next capture\nwill appear here", fill=MUTED,
                                     font=("Segoe UI", 14), justify="center")
            return
        image = self._source_image.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self.preview.create_image(width / 2, height / 2, image=self._photo)

    def _available(self):
        return self.operator.state == State.READY and not self._refreshing and not self.closing and not self.operator._recovering

    def _sync_controls(self):
        ready = self._available()
        self.voice_combo.configure(state="normal" if ready else "disabled")
        for widget in (self.mic_combo, self.camera_combo, self.resolution_combo):
            widget.configure(state="readonly" if ready else "disabled")
        # A fixture or the laser rig's own camera cannot be switched here.
        if self.operator.camera.fixture is not None or getattr(self.operator.camera, "external", False):
            self.camera_combo.configure(state="disabled")
            self.resolution_combo.configure(state="disabled")
        for widget in (self.refresh_button, self.new_button, self.model_button):
            widget.configure(state="normal" if ready else "disabled")
        busy = self.operator.state not in (State.READY, State.STOPPED)
        self.stop_button.configure(state="normal" if busy else "disabled", text="Recover" if self.operator.state == State.ERROR else "Stop")
        recording = self.operator.state == State.RECORDING
        self.talk_button.configure(text="Listening… release to send" if recording else "Hold to talk  ·  SPACE",
                                   bg=ACCENT if recording else BLUE if ready else FIELD,
                                   fg=BACKGROUND if recording else "white" if ready else MUTED,
                                   cursor="hand2" if ready or recording else "arrow")

    def _refresh_model_label(self, speech_configured=None):
        model = self.operator.grok
        name = getattr(model, "name", "Grok")
        model_name = getattr(model, "model", "grok-4.7")
        effort = getattr(model, "reasoning_effort", "auto")
        search = getattr(model, "search", "auto")
        self.model_text.set(f"{name} · {model_name} · Reasoning {effort} · Search {search}")
        if speech_configured is None:
            speech_configured = bool(getattr(self.operator.speech, "api_key", ""))
        self.key_text.set(f"{name}: {'key loaded' if getattr(model, 'api_key', '') else 'key missing'} · "
                          f"ElevenLabs: {'key loaded' if speech_configured else 'key missing'}")

    def _apply_model_settings(self, values, *, provider, model, effort, search):
        if not self._available():
            raise ValueError("Wait until the current turn finishes before changing models")
        candidate = create_model(self.operator.grok.client, values, provider=provider, model=model,
                                 reasoning_effort=effort, search=search,
                                 history_images=getattr(self.operator.grok, "history_images", "latest"),
                                 state_mode=getattr(self.operator.grok, "state_mode", "client"))
        if not candidate.api_key:
            raise ValueError("Add the selected provider's API key before switching")
        if not self.operator.reset_session():
            raise ValueError("The current turn has not finished")
        self.operator.grok = candidate
        self._refresh_model_label()

    def show_model_settings(self):
        if not self._available():
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Model settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        form = ttk.Frame(dialog, padding=20)
        form.pack(fill="both", expand=True)
        current = self.operator.grok
        provider = tk.StringVar(value=getattr(current, "provider", "grok"))
        model = tk.StringVar(value=getattr(current, "model", "grok-4.7"))
        effort = tk.StringVar(value=getattr(current, "reasoning_effort", "medium"))
        search = tk.StringVar(value=getattr(current, "search", "auto"))
        key = tk.StringVar()
        voice_key = tk.StringVar()
        fields = [("Provider", provider, ("openai", "grok")), ("Model", model, None),
                  ("Reasoning effort", effort, ("low", "medium", "high", "xhigh", "auto")),
                  ("Online search", search, ("auto", "off"))]
        for row, (label, variable, choices) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
            widget = ttk.Combobox(form, textvariable=variable, values=choices or (),
                                  state="readonly" if choices else "normal", width=30)
            widget.grid(row=row, column=1, sticky="ew")
            if row == 0:
                widget.bind("<<ComboboxSelected>>", lambda event: (
                    model.set("gpt-6.1-sol" if provider.get() == "openai" else "grok-4.7"),
                    effort.set("low" if provider.get() == "openai" else "medium")))
        ttk.Label(form, text="API key (optional)").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=key, show="•", width=32).grid(row=4, column=1, sticky="ew")
        ttk.Label(form, text="ElevenLabs key (optional)").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=voice_key, show="•", width=32).grid(row=5, column=1, sticky="ew")
        ttk.Label(form, text="Leave keys blank to keep them. Keys are saved securely by Windows.\n"
                  "Saving starts a new repair session. Low reasoning favors speed.\n"
                  "Enter keys and click Save; no Windows setup is needed.").grid(
                      row=6, column=0, columnspan=2, sticky="w", pady=12)

        def save():
            if not self._available():
                return
            path = Path(__file__).resolve().parent.parent / ".env"
            prefix = "OPENAI" if provider.get() == "openai" else "XAI"
            changes = {"MODEL_PROVIDER": provider.get(), f"{prefix}_MODEL": model.get().strip(),
                       f"{prefix}_REASONING_EFFORT": effort.get(), f"{prefix}_SEARCH": search.get()}
            if not changes[f"{prefix}_MODEL"]:
                messagebox.showerror("Model required", "Enter a model name", parent=dialog)
                return
            if key.get().strip():
                changes[f"{prefix}_API_KEY"] = key.get().strip()
            if voice_key.get().strip():
                changes["ELEVENLABS_API_KEY"] = voice_key.get().strip()
            try:
                # Validate before changing the saved configuration or current session.
                values = {**load_settings(path), **changes}
                candidate = create_model(current.client, values, provider=provider.get(), model=model.get().strip(),
                                         reasoning_effort=effort.get(), search=search.get())
                if not candidate.api_key:
                    raise ValueError("The selected provider's API key is missing")
                values = save_settings(path, changes)
                from .speech import Speech
                old_speech = self.operator.speech
                speech = Speech(current.client, values.get("ELEVENLABS_API_KEY") or "",
                                old_speech.voice_id, old_speech.device)
                self._apply_model_settings(values, provider=provider.get(), model=model.get().strip(),
                                           effort=effort.get(), search=search.get())
                self.operator.speech = speech
                self._refresh_model_label()
            except (ValueError, OSError, CredentialError) as error:
                messagebox.showerror("Settings not applied", str(error), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(form, text="Save & start new session", command=save).grid(row=7, column=1, sticky="e")

    def _choose_mic(self, event=None):
        if self._available():
            self.operator.recorder.device = self._mic_choices[self.mic_text.get()]
            self.device_note.set("Microphone selected for the next turn.")

    def _choose_camera(self, event=None):
        if self._available() and self.operator.camera.fixture is None:
            selected = self._camera_choices[self.camera_text.get()]
            self.operator.camera.camera_name = selected.input_name
            self.operator.camera.camera_label = selected.name
            self.resolution_text.set("1920 × 1080" if selected.name == camera_capture.CAMERA_NAME else "Auto (camera default)")
            self.operator.camera.resolution = RESOLUTIONS[self.resolution_text.get()]
            self.device_note.set("Camera selected for the next capture.")

    def _choose_resolution(self, event=None):
        if self._available():
            self.operator.camera.resolution = RESOLUTIONS[self.resolution_text.get()]

    def _choose_voice(self, event=None):
        if self._available():
            text = self.voice_text.get().strip()
            voice_id = self._voice_choices.get(text, text)
            if voice_id and not any(char.isspace() for char in voice_id):
                self.operator.speech.voice_id = voice_id
                self.device_note.set("Voice selected for the next reply.")

    def refresh_devices(self):
        if self._available():
            self.schedule(self._discover())

    async def _discover(self):
        if self._refreshing:
            return
        self._refreshing = True
        self.device_note.set("Looking for connected devices…")
        self._sync_controls()
        problems = []
        try:
            microphones, cameras = await asyncio.gather(asyncio.to_thread(list_microphones),
                                                        asyncio.to_thread(list_cameras), return_exceptions=True)
            if isinstance(microphones, Exception):
                problems.append("Microphone discovery failed")
                self.log(str(microphones))
            else:
                self._mic_choices = {"System default microphone": None, **{item.label: item.index for item in microphones}}
                chosen = next((name for name, index in self._mic_choices.items() if index == self.operator.recorder.device), None)
                if chosen is None:
                    chosen = f"Selected microphone [{self.operator.recorder.device}] (unavailable)"
                    self._mic_choices[chosen] = self.operator.recorder.device
                self.mic_text.set(chosen)
            if isinstance(cameras, Exception):
                problems.append("Camera discovery failed")
                self.log(str(cameras))
            else:
                counts = {}
                self._camera_choices = {}
                chosen = None
                current = self.operator.camera.camera_name
                for camera in cameras:
                    counts[camera.name] = counts.get(camera.name, 0) + 1
                    label = camera.name if counts[camera.name] == 1 else f"{camera.name} ({counts[camera.name]})"
                    self._camera_choices[label] = camera
                    if camera.input_name == current or (chosen is None and camera.name == current):
                        chosen = label
                if chosen is None:
                    chosen = getattr(self.operator.camera, "camera_label", current) + " (unavailable)"
                    self._camera_choices[chosen] = CameraDevice(getattr(self.operator.camera, "camera_label", current), current)
                self.camera_text.set(chosen)
                if not cameras:
                    problems.append("No connected cameras found")
            self.mic_combo.configure(values=list(self._mic_choices))
            self.camera_combo.configure(values=list(self._camera_choices))
            if self.operator.camera.fixture is not None:
                self.camera_text.set("Image fixture")
            elif getattr(self.operator.camera, "external", False):
                self.camera_text.set(self.operator.camera.camera_label)
            self.device_note.set(". ".join(problems) if problems else "Devices ready. Changes apply to the next turn.")
            if hasattr(self.operator.speech, "list_voices") and self.operator.speech.api_key:
                try:
                    voices = await self.operator.speech.list_voices()
                    self._voice_choices = {f"{name} [{voice_id}]": voice_id for voice_id, name in voices}
                    self.voice_combo.configure(values=list(self._voice_choices))
                    current = self.operator.speech.voice_id
                    self.voice_text.set(next((label for label, value in self._voice_choices.items() if value == current), current))
                except Exception as error:
                    self.log(str(error))
                    self.device_note.set("Devices ready. Voice list unavailable; enter a voice ID and press Enter.")
        finally:
            self._refreshing = False
            self._sync_controls()

    def _press(self, source):
        if self._held is None:
            self._choose_voice()
            self._held = source
            self._admitted = self._available() and self.operator.space_down()

    def _release(self, source):
        if self._held == source:
            admitted = self._admitted
            self._held, self._admitted = None, False
            if admitted:
                self.operator.space_up()

    def _key_down(self, event=None):
        self._press("keyboard")
        return "break"

    def _key_up(self, event=None):
        self._release("keyboard")
        return "break"

    def _mouse_down(self, event=None):
        self._press("mouse")
        if self._admitted and self._held == "mouse":
            self.talk_button.grab_set()
        return "break"

    def _mouse_up(self, event=None):
        self.talk_button.grab_release()
        self._release("mouse")
        return "break"

    def _focus_out(self, event):
        if self.operator.state == State.RECORDING:
            self.root.after_idle(self._check_recording_focus)

    def _check_recording_focus(self):
        # Native ttk dropdowns have Tcl widgets without Python counterparts.
        # Read the focus path directly; focus_displayof() raises KeyError there.
        if (not self.closing and self.operator.state == State.RECORDING
                and not self.root.tk.call("focus", "-displayof", self.root._w)):
            self._admitted = False
            self.talk_button.grab_release()
            self.interrupt()

    def new_session(self):
        if self._available():
            self.operator.reset_session()

    def interrupt(self):
        self.schedule(self.operator.interrupt())

    def schedule(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        def done(job):
            self._tasks.discard(job)
            if not job.cancelled() and job.exception():
                self.log(f"Control failed: {job.exception()}")
        task.add_done_callback(done)

    def request_close(self):
        self.closing = True
        self._sync_controls()

    async def run(self):
        self.root.update()
        self.refresh_devices()
        try:
            while not self.closing:
                self.root.update()
                await asyncio.sleep(0.01)
        finally:
            self.closing = True
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.operator.close()
            self.root.destroy()
