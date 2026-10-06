"""The AI boundary: four unchanged pixel coordinates and completion/cancel ACK."""
import importlib
from typing import Protocol


class PointingFailed(RuntimeError):
    """Downstream could not point at the target but confirms it is stopped/idle.

    Raise this from send_bbox or wait_until_complete for known outcomes such as
    an unreachable target. Like any pointer failure it is kept from the user:
    the model is told the target was not marked and the session carries on.
    """


class Pointer(Protocol):
    async def send_bbox(self, x1: int, y1: int, x2: int, y2: int) -> None: ...
    async def wait_until_complete(self) -> None:
        """Return only on COMPLETE / ALIGNED; raise on downstream failure.

        Raise PointingFailed when the target was not reached but the pointer is idle.
        """
        ...
    async def cancel(self) -> bool:
        """Return True only after downstream acknowledges stopped/idle."""
        ...


class DryRunPointer:
    """Explicit development simulation; never contacts physical hardware."""
    async def send_bbox(self, x1: int, y1: int, x2: int, y2: int) -> None:
        print(f"DRY RUN bbox: ({x1}, {y1}, {x2}, {y2})", flush=True)

    async def wait_until_complete(self) -> None:
        print("DRY RUN pointer completion", flush=True)

    async def cancel(self) -> bool:
        return True


def load_pointer(spec: str | None) -> Pointer:
    if spec is None:
        return DryRunPointer()
    module_name, factory_name = spec.split(":", 1)
    pointer = getattr(importlib.import_module(module_name), factory_name)()
    for name in ("send_bbox", "wait_until_complete", "cancel"):
        if not callable(getattr(pointer, name, None)):
            raise TypeError(f"Pointer adapter must implement async {name}()")
    return pointer
