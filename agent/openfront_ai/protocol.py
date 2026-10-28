"""Trames du socket d'environnement (voir env/src/protocol.ts).

    u32 BE  longueur totale du reste de la trame
    u32 BE  longueur de l'en-tête JSON
    ...     en-tête JSON UTF-8
    ...     tableaux binaires little-endian, chacun aligné sur 8 octets

L'en-tête décrit les tableaux dans ``__buffers`` (nom, type, forme, position,
taille en octets).
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

_U32 = struct.Struct(">I")
_DTYPES = {"uint8": "<u1", "uint16": "<u2", "int32": "<i4", "float32": "<f4"}
_NAMES = {np.dtype(np.dtype(v).type): k for k, v in _DTYPES.items()}


def _pad8(n: int) -> int:
    return (n + 7) & ~7


def encode_frame(header: dict[str, Any], buffers: dict[str, np.ndarray] | None = None) -> bytes:
    specs = []
    blobs = []
    offset = 0
    for name, array in (buffers or {}).items():
        dtype = _NAMES[np.dtype(array.dtype.type)]
        data = np.ascontiguousarray(array, dtype=_DTYPES[dtype]).tobytes()
        specs.append({"name": name, "dtype": dtype, "shape": list(array.shape), "offset": offset, "nbytes": len(data)})
        blobs.append((offset, data))
        offset = _pad8(offset + len(data))
    if specs:
        header = {**header, "__buffers": specs}
    body = json.dumps(header).encode("utf-8")
    blob_start = _pad8(8 + len(body))
    out = bytearray(blob_start + offset)
    out[0:4] = _U32.pack(len(out) - 4)
    out[4:8] = _U32.pack(len(body))
    out[8 : 8 + len(body)] = body
    for start, data in blobs:
        out[blob_start + start : blob_start + start + len(data)] = data
    return bytes(out)


def decode_body(body: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Décode une trame sans ses 4 premiers octets de longueur."""
    (json_length,) = _U32.unpack_from(body, 0)
    header = json.loads(body[4 : 4 + json_length])
    specs = header.pop("__buffers", [])
    blob_start = _pad8(8 + json_length) - 4
    buffers = {}
    for spec in specs:
        start = blob_start + spec["offset"]
        array = np.frombuffer(body, dtype=_DTYPES[spec["dtype"]], count=spec["nbytes"] // np.dtype(_DTYPES[spec["dtype"]]).itemsize, offset=start)
        buffers[spec["name"]] = array.reshape(spec["shape"])
    return header, buffers
