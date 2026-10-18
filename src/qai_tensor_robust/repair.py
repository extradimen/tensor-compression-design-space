"""Protocol v1.1 pure task distillation, detached group-DRO and finite batches.

No acquisition, model loading, GPU allocation or evaluation-data access.
"""
from dataclasses import dataclass
import math
import random

import torch

DOMAINS = ("text", "math")


@dataclass
class ExampleKL:
    kl: torch.Tensor
    entropy: torch.Tensor
    target_counts: torch.Tensor


def example_kl(student, teacher, target_mask, *, chunk_positions=64):
    """Teacher||student, T=1, full vocabulary FP32; average each example first.

    Chunk positions, never truncate or renormalize the vocabulary. Target masks
    mark label positions; prediction at t scores target t+1. No CE is added.
    """
    if (student.ndim != 3 or student.shape != teacher.shape or student.shape[1] < 2
            or target_mask.shape != student.shape[:2] or target_mask.dtype != torch.bool
            or teacher.device != student.device or target_mask.device != student.device
            or type(chunk_positions) is not int or chunk_positions < 1):
        raise ValueError("Invalid task KL tensors or position chunk")
    counts = target_mask[:, 1:].sum(1)
    if (counts == 0).any():
        raise ValueError("Zero retained task targets; reject this sequence in every arm")
    losses, entropies = [], []
    for i, mask in enumerate(target_mask[:, 1:]):
        indices = mask.nonzero().flatten()
        chunks, hc = [], []
        for positions in indices.split(chunk_positions):
            raw_s, raw_t = student[i, positions].float(), teacher[i, positions].detach().float()
            if not torch.isfinite(raw_s).all() or not torch.isfinite(raw_t).all():
                raise ValueError("Nonfinite task logits")
            log_s, log_t = raw_s.log_softmax(-1), raw_t.log_softmax(-1)
            p_t = log_t.exp()
            chunks.append((p_t * (log_t - log_s)).sum())
            hc.append(-(p_t * log_t).sum())
        losses.append(torch.stack(chunks).sum() / counts[i])
        entropies.append(torch.stack(hc).sum() / counts[i])
    return ExampleKL(torch.stack(losses), torch.stack(entropies), counts)


def frozen_entropies(training_example_entropies):
    """Token mean on the training task mask, frozen and shared by all arms.

    Supply (per-example entropy mean, retained target count). This scalar
    normalizer differs from KL, whose per-example means are averaged equally.
    """
    if set(training_example_entropies) != set(DOMAINS):
        raise ValueError("Both training domains required")
    result = {}
    for domain, values in training_example_entropies.items():
        if (not values or any(not isinstance(v,(tuple,list)) or len(v)!=2 for v in values)
                or any(not math.isfinite(float(v)) or float(v)<0 or type(n) is not int or n<1 for v,n in values)):
            raise ValueError("Finite nonnegative training entropies required")
        result[domain] = max(0.1, sum(float(v)*n for v,n in values) / sum(n for _,n in values))
    return result


class GroupDRO:
    def __init__(self, *, eta=0.1, log_q=None):
        if eta != 0.1:
            raise ValueError("Protocol v1.1 DRO eta is fixed at 0.1")
        self.eta = eta
        self.log_q = torch.full((2,), -math.log(2), dtype=torch.float64)
        if log_q is not None:
            q = torch.as_tensor(log_q, dtype=torch.float64)
            if q.shape != (2,) or not torch.isfinite(q).all():
                raise ValueError("Invalid DRO checkpoint")
            if abs(float(q.logsumexp(0))) > 1e-10:
                raise ValueError("DRO checkpoint weights must be normalized")
            self.log_q = q.clone().detach()

    @property
    def weights(self):
        return dict(zip(DOMAINS, self.log_q.exp().tolist()))

    def update(self, losses):
        if set(losses) != set(DOMAINS):
            raise ValueError("Both group losses required")
        value = torch.tensor([float(losses[g].detach()) if isinstance(losses[g], torch.Tensor)
                              else float(losses[g]) for g in DOMAINS], dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite group loss")
        self.log_q = (self.log_q + self.eta * value).detach()
        self.log_q -= self.log_q.logsumexp(0)
        return self.weights

    def loss(self, losses):
        weights = self.update(losses)  # before the parameter step; q has no graph
        return sum(weights[g] * losses[g] for g in DOMAINS)

    def state_dict(self):
        return {"eta": self.eta, "domains": list(DOMAINS), "log_q": self.log_q.tolist()}


def macro_batches(examples, *, seed, input_cap=65536, sequence_length=512, max_steps=None):
    """Immutable 4:1 stream; reject incomplete/empty target records, no cycling.

    Conservatively count all 512 model-input positions, including prompts and
    padding. Nonpadding exposure and task targets are reported separately.
    """
    if input_cap != 65536 or sequence_length != 512 or type(seed) is not int:
        raise ValueError("Pilot exposure constants or seed differ from protocol")
    groups = {g: [] for g in DOMAINS}
    seen = set()
    for row in examples:
        key = row.get("sequence_id", row.get("unit_id"))
        if not key or key in seen or row.get("domain") not in groups:
            raise ValueError("Unique sequence identities and original domains required")
        seen.add(key)
        if any(len(row[k]) != 512 for k in ("input_ids", "attention_mask", "target_mask")):
            raise ValueError("Freeze padded/truncated length 512 before batching")
        if any(type(v) is not bool for k in ("attention_mask", "target_mask") for v in row[k]):
            raise ValueError("Explicit boolean token masks required")
        if any(t and not a for t, a in zip(row["target_mask"], row["attention_mask"])) or not any(row["target_mask"][1:]):
            raise ValueError("Invalid/zero task targets")
        groups[row["domain"]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        group.sort(key=lambda r: r.get("sequence_id", r["unit_id"]))
        rng.shuffle(group)
    available = min(len(groups["text"]) // 4, len(groups["math"]))
    if max_steps is not None:
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("Explicit nonnegative maximum steps required")
        available = min(available, max_steps)
    batches, used = [], 0
    for step in range(available):
        batch = groups["text"][step * 4:step * 4 + 4] + [groups["math"][step]]
        inputs = sum(len(r["input_ids"]) for r in batch)
        if used + inputs > input_cap:
            break  # previous complete macro-batch only
        batches.append(batch)
        used += inputs
    return batches, {"input_tokens": used, "padded_positions": len(batches) * 5 * 512,
                     "nonpadding_input_tokens": sum(sum(r["attention_mask"]) for b in batches for r in b),
                     "target_tokens": sum(sum(r["target_mask"][1:]) for b in batches for r in b),
                     "optimizer_steps": len(batches), "input_cap": input_cap,
                     "sequence_ids": [[r.get("sequence_id", r["unit_id"]) for r in b] for b in batches]}


def learning_rate(step, total_steps):
    """Step numbered 1..N; ceil(5% N) warmup, then cosine to zero at final step."""
    if not 1 <= step <= total_steps:
        raise ValueError("Optimizer step outside the frozen schedule")
    warmup = max(1, math.ceil(0.05 * total_steps))
    if step <= warmup:
        return 1e-4 * step / warmup
    return 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
