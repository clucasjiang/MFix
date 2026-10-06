from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class State(str, Enum):
    READY = "READY"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    WAITING_FOR_GROK = "WAITING_FOR_GROK"
    OUTPUTTING = "OUTPUTTING"
    WAITING_FOR_POINTER = "WAITING_FOR_POINTER"
    ERROR = "ERROR"  # Physical completion is unknown; new turns stay locked.
    STOPPED = "STOPPED"


class BoundingBox(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    x1: int
    y1: int
    x2: int
    y2: int

    def validate_image(self, width: int, height: int) -> None:
        if not (0 <= self.x1 < self.x2 < width and 0 <= self.y1 < self.y2 < height):
            raise ValueError("Bounding box is degenerate or outside the supplied image")


class DiagnosticResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    speech: str = Field(min_length=1)
    target_present: bool
    target_label: str | None
    bbox: BoundingBox | None
    confidence: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    status: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_target(self):
        if not self.speech.strip():
            raise ValueError("Speech is empty")
        if self.target_present:
            if self.bbox is None or not self.target_label or not self.target_label.strip() or self.confidence is None:
                raise ValueError("A target requires a box, label, and confidence")
        elif self.bbox is not None or self.target_label is not None or self.confidence is not None:
            raise ValueError("No target requires null box, label, and confidence")
        return self


@dataclass(frozen=True)
class ImageFrame:
    path: Path
    width: int
    height: int


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    previous_response_id: str | None = None
    history: list[dict] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    pending_feedback: list[str] = field(default_factory=list)
    model_identity: tuple[str, str] | None = None

    def reset(self) -> None:
        self.session_id = uuid4().hex
        self.previous_response_id = None
        self.history.clear()
        self.turns.clear()
        self.pending_feedback.clear()
        self.model_identity = None
