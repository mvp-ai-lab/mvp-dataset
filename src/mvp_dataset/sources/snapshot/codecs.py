"""Value codecs used by materialized snapshots."""

from __future__ import annotations

import io
import json
from functools import cache
from importlib import import_module

_CODEC_FIELD = "__mvp_snapshot_codec__"
_TORCH_TENSOR_CODEC = "torch-tensor-v2"
_TENSOR_FIELDS = frozenset({_CODEC_FIELD, "payload", "dtype", "shape", "device_type"})


def encode_snapshot_value(value: object) -> object:
    """Encode supported non-Arrow values into self-describing Arrow values."""
    torch = _load_torch(required=False)
    if torch is not None and isinstance(value, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return {
            _CODEC_FIELD: _TORCH_TENSOR_CODEC,
            "payload": buffer.getvalue(),
            "dtype": str(value.dtype),
            "shape": json.dumps(list(value.shape), separators=(",", ":")),
            "device_type": value.device.type,
        }
    if isinstance(value, dict):
        if _CODEC_FIELD in value:
            msg = f"[SnapshotCodecConflict] value contains reserved field {_CODEC_FIELD!r}"
            raise ValueError(msg)
        return {key: encode_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return [encode_snapshot_value(item) for item in value]
    return value


def decode_snapshot_value(value: object) -> object:
    """Decode self-describing snapshot values into their original runtime types."""
    if isinstance(value, dict):
        if value.get(_CODEC_FIELD) == _TORCH_TENSOR_CODEC:
            return _decode_torch_tensor(value)
        return {key: decode_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_snapshot_value(item) for item in value]
    return value


def _decode_torch_tensor(value: dict[str, object]) -> object:
    if set(value) != _TENSOR_FIELDS:
        msg = "[InvalidSnapshotTensor] tensor envelope fields are invalid"
        raise RuntimeError(msg)
    payload = value["payload"]
    dtype = value["dtype"]
    raw_shape = value["shape"]
    device_type = value["device_type"]
    if not isinstance(payload, bytes) or not all(isinstance(item, str) for item in (dtype, raw_shape, device_type)):
        msg = "[InvalidSnapshotTensor] tensor envelope values are invalid"
        raise RuntimeError(msg)

    try:
        shape = json.loads(raw_shape)
    except json.JSONDecodeError as error:
        msg = "[InvalidSnapshotTensor] tensor shape is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(shape, list) or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape):
        msg = "[InvalidSnapshotTensor] tensor shape is invalid"
        raise RuntimeError(msg)

    torch = _load_torch(required=True)
    try:
        tensor = torch.load(io.BytesIO(payload), map_location=device_type, weights_only=True)
    except RuntimeError as error:
        msg = f"[SnapshotTensorDeviceUnavailable] unable to restore tensor on device type {device_type!r}"
        raise RuntimeError(msg) from error
    if not isinstance(tensor, torch.Tensor):
        msg = "[InvalidSnapshotTensor] decoded value is not a torch.Tensor"
        raise RuntimeError(msg)
    if str(tensor.dtype) != dtype or list(tensor.shape) != shape or tensor.device.type != device_type:
        msg = "[InvalidSnapshotTensor] decoded tensor metadata does not match its envelope"
        raise RuntimeError(msg)
    return tensor


@cache
def _load_torch(*, required: bool) -> object | None:
    try:
        return import_module("torch")
    except ModuleNotFoundError as error:
        if not required:
            return None
        msg = "[TorchUnavailable] install torch to read tensor snapshot values"
        raise RuntimeError(msg) from error
