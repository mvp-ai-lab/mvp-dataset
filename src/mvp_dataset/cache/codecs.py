"""Value codecs for cache tar storage.

Codec tags
----------
``raw``        bytes / bytearray  — stored as-is
``utf8``       str                — UTF-8 encoded
``json``       JSON-compatible scalars and containers
``structured`` nested containers that include raw bytes
``npy``        numpy.ndarray      — stored via numpy.save
``npy_tensor`` torch.Tensor       — converted to CPU, stored via numpy.save
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

CODEC_RAW = "raw"
CODEC_UTF8 = "utf8"
CODEC_JSON = "json"
CODEC_STRUCTURED = "structured"
CODEC_NPY = "npy"
CODEC_NPY_TENSOR = "npy_tensor"

_STRUCTURED_BYTES_MARKER_KEY = "__mvp_dataset_codec__"
_STRUCTURED_BYTES_MARKER_VALUE = "bytes"
_STRUCTURED_BYTES_DATA_KEY = "base64"


def encode_value(value: Any) -> tuple[bytes, str]:
    """Encode *value* for storage in a cache tar.

    Args:
        value: The Python value to encode.  See the module docstring for
            the supported type matrix.

    Returns:
        A ``(data_bytes, codec_tag)`` tuple where *codec_tag* identifies
        the encoding used.

    Raises:
        TypeError: If *value* has an unsupported type.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), CODEC_RAW

    if isinstance(value, str):
        return value.encode("utf-8"), CODEC_UTF8

    if _is_json_compatible(value):
        return (
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
            CODEC_JSON,
        )

    if _is_structured_compatible(value):
        return (
            json.dumps(_encode_structured(value), ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
            CODEC_STRUCTURED,
        )

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            return buf.getvalue(), CODEC_NPY
    except ImportError:
        pass

    try:
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            arr = value.cpu().numpy()
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            return buf.getvalue(), CODEC_NPY_TENSOR
    except ImportError:
        pass

    msg = (
        f"[CacheCodecError] unsupported value type {type(value).__name__!r}; "
        f"supported: bytes, str, JSON-compatible scalars/containers, "
        f"nested containers containing bytes, "
        f"numpy.ndarray, torch.Tensor"
    )
    raise TypeError(msg)


def decode_value(data: bytes, codec_tag: str) -> Any:
    """Decode *data* using *codec_tag* back to the original Python value.

    Args:
        data: Raw bytes read from a cache tar member.
        codec_tag: The codec identifier stored alongside the data
            (e.g. ``"raw"``, ``"utf8"``, ``"json"``, ``"npy"``,
            ``"npy_tensor"``).

    Returns:
        The decoded Python value.

    Raises:
        ValueError: If *codec_tag* is not recognized.
    """
    if codec_tag == CODEC_RAW:
        return data

    if codec_tag == CODEC_UTF8:
        return data.decode("utf-8")

    if codec_tag == CODEC_JSON:
        return json.loads(data.decode("utf-8"))

    if codec_tag == CODEC_STRUCTURED:
        return _decode_structured(json.loads(data.decode("utf-8")))

    if codec_tag == CODEC_NPY:
        import numpy as np

        return np.load(io.BytesIO(data), allow_pickle=False)

    if codec_tag == CODEC_NPY_TENSOR:
        import numpy as np
        import torch

        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return torch.from_numpy(arr)

    msg = f"[CacheCodecError] unknown codec tag {codec_tag!r}"
    raise ValueError(msg)


def _is_json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_compatible(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_compatible(v) for k, v in value.items())
    return False


def _is_structured_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_structured_compatible(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_structured_compatible(v) for k, v in value.items())
    return False


def _encode_structured(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {
            _STRUCTURED_BYTES_MARKER_KEY: _STRUCTURED_BYTES_MARKER_VALUE,
            _STRUCTURED_BYTES_DATA_KEY: base64.b64encode(bytes(value)).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_encode_structured(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_structured(v) for k, v in value.items()}
    msg = f"[CacheCodecError] unsupported structured value type {type(value).__name__!r}"
    raise TypeError(msg)


def _decode_structured(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_structured(v) for v in value]
    if isinstance(value, dict):
        if value.get(_STRUCTURED_BYTES_MARKER_KEY) == _STRUCTURED_BYTES_MARKER_VALUE and set(value) == {
            _STRUCTURED_BYTES_MARKER_KEY,
            _STRUCTURED_BYTES_DATA_KEY,
        }:
            return base64.b64decode(value[_STRUCTURED_BYTES_DATA_KEY].encode("ascii"))
        return {k: _decode_structured(v) for k, v in value.items()}
    return value
