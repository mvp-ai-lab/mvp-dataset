"""Small PyTorch compatibility shims."""

from __future__ import annotations

try:
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from torch.utils.data import default_collate, get_worker_info
    from torch.utils.data._utils.pin_memory import pin_memory as pin_memory_item
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

    class TorchDataLoader:
        """Fallback DataLoader used when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the object."""
            msg = "[TorchUnavailable] install torch to use TorchLoader"
            raise RuntimeError(msg)

    class TorchIterableDataset:
        """Fallback base class used when PyTorch is not installed."""

    def default_collate(samples: list[object]) -> object:
        """Collate a batch with PyTorch when it is available."""
        msg = "[TorchUnavailable] install torch to use default_collate"
        raise RuntimeError(msg)

    def get_worker_info() -> object | None:
        """Return PyTorch worker information when running in a DataLoader worker."""
        return None

    def pin_memory_item(item: object) -> object:
        """Pin an item in memory when PyTorch pinning is available."""
        return item
else:
    TORCH_AVAILABLE = True


__all__ = [
    "TORCH_AVAILABLE",
    "TorchDataLoader",
    "TorchIterableDataset",
    "default_collate",
    "get_worker_info",
    "pin_memory_item",
]
