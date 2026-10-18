"""Deterministic uncompressed deployable tensor payload and strict byte ledger.

Includes buffers, tensor descriptors and operator metadata. No pickle, padding,
optimizer, teacher or calibration cache. Whole-model exports must include every
uncompressed tensor; per-layer size alone is not a whole-model compression ratio.
"""
import json
import struct
from dataclasses import dataclass

import torch

MAGIC = b"QAITENS1"


def payload_parts(module):
    """Describe actual tensors without copying a whole model just to count bytes.

    Only exact storage aliases are shared. Equal-valued independent tensors and
    overlapping views remain distinct; inference must preserve these aliases.
    """
    tensors, records, aliases, seen = [], [], {}, {}
    for name, tensor in sorted(module.state_dict().items()):
        if tensor.device.type == "meta":
            raise ValueError("Cannot count an actual deployment from meta tensors")
        identity = (tensor.device, tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                    tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if tensor.numel() and identity in seen:
            aliases[name] = seen[identity]
            continue
        seen[identity] = name
        records.append({"name": name, "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape), "nbytes": tensor.numel() * tensor.element_size()})
        tensors.append(tensor)
    metadata = module.deployment_metadata() if hasattr(module, "deployment_metadata") else {"kind": "state_dict"}
    header = json.dumps({"format": 2, "operator": metadata, "tensors": records, "aliases": aliases},
                        ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return header, tensors


def deployment_size(module):
    header, tensors = payload_parts(module)
    return 16 + len(header) + sum(t.numel() * t.element_size() for t in tensors)


def deployment_payload(module):
    header, tensors = payload_parts(module)
    raw = [t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() for t in tensors]
    return MAGIC + struct.pack("<Q", len(header)) + header + b"".join(raw)


def decode_payload(payload):
    if len(payload) < 16 or payload[:8] != MAGIC:
        raise ValueError("Invalid tensor payload")
    size = struct.unpack("<Q", payload[8:16])[0]
    header = json.loads(payload[16:16 + size])
    cursor = 16 + size
    tensors = {}
    allowed = {str(t): t for t in (torch.float64, torch.float32, torch.float16,
                                   torch.bfloat16, torch.int64, torch.int32,
                                   torch.int16, torch.int8, torch.uint8, torch.bool)}
    for record in header["tensors"]:
        dtype = allowed.get(record["dtype"])
        if dtype is None or record["name"] in tensors:
            raise ValueError("Unsupported dtype or duplicate tensor")
        n = record["nbytes"]
        if n < 0 or cursor + n > len(payload):
            raise ValueError("Truncated or empty tensor")
        tensor = (torch.frombuffer(bytearray(payload[cursor:cursor + n]), dtype=dtype).clone()
                  if n else torch.empty(0, dtype=dtype))
        tensors[record["name"]] = tensor.reshape(record["shape"])
        cursor += n
    if cursor != len(payload):
        raise ValueError("Unexpected trailing bytes")
    for name, source in header.get("aliases", {}).items():
        if name in tensors or source not in tensors:
            raise ValueError("Invalid storage alias")
        tensors[name] = tensors[source]
    return header["operator"], tensors


@dataclass(frozen=True)
class WholeModelBudget:
    cap_bytes: int
    fixed_bytes: int

    def __post_init__(self):
        if type(self.cap_bytes) is not int or type(self.fixed_bytes) is not int or not 0 <= self.fixed_bytes <= self.cap_bytes:
            raise ValueError("Invalid whole-model byte budget")

    def check(self, variable_payloads):
        variable = sum(len(p) for p in variable_payloads)
        total = self.fixed_bytes + variable
        if total > self.cap_bytes:
            raise ValueError(f"Byte cap exceeded: {total} > {self.cap_bytes}")
        return {"cap_bytes": self.cap_bytes, "fixed_bytes": self.fixed_bytes,
                "variable_bytes": variable, "total_bytes": total,
                "unused_bytes": self.cap_bytes - total}
