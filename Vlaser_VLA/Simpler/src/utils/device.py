"""Device selection shared by the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

import torch

DeviceLike = str | torch.device | None
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None, index: int = 0) -> torch.device:
    """Resolve ``None`` to the detected backend at ``index``.

    An explicit device is honoured as given, except that a bare ``xpu`` or
    ``cuda`` gets ``index``.
    """
    if device is None or device == "":
        device_type = default_device_type()
        return (
            torch.device("cpu")
            if device_type == "cpu"
            else torch.device(device_type, index)
        )
    device = torch.device(device)
    if device.type in _ACCELERATORS and device.index is None:
        return torch.device(device.type, index)
    return device
