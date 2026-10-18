"""Task-level CE/KL primitives. No model loading, data access or job launcher.

The formal protocol must choose its supervised tokens and CE/KL coefficient.
Masks explicitly mark valid target tokens; prompt/padding inclusion is never
inferred from a tokenizer, task name or model chat template.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass
class TokenLoss:
    ce_sum: torch.Tensor
    kl_sum: torch.Tensor
    token_count: int
    kl_weight: float

    @property
    def mean(self):
        return (self.ce_sum + self.kl_weight * self.kl_sum) / self.token_count


def causal_task_loss(student_logits, labels, target_mask, *, teacher_logits=None,
                     kl_weight=0.0):
    """Student position t predicts label t+1. KL is teacher || student.

Teacher probabilities and the dense normalization denominator must be frozen.
For relative excess with CE+lambda*KL, the dense denominator is dense CE on
the same supervised tokens, not KL(teacher, teacher), which is zero.
"""
    if student_logits.ndim != 3 or student_logits.shape[1] < 2:
        raise ValueError("Expected [batch, length>=2, vocabulary] logits")
    if labels.shape != student_logits.shape[:2] or target_mask.shape != labels.shape:
        raise ValueError("Labels/mask must match batch and length")
    if labels.dtype != torch.long or target_mask.dtype != torch.bool:
        raise ValueError("Labels must be int64 and target mask boolean")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("KL weight must be finite and nonnegative")
    if labels.device != student_logits.device or target_mask.device != student_logits.device:
        raise ValueError("Logits, labels and mask must share a device")
    mask = target_mask[:, 1:]
    count = int(mask.sum())
    if not count:
        raise ValueError("No valid next-token supervision")
    targets = labels[:, 1:][mask]
    if (targets < 0).any() or (targets >= student_logits.shape[-1]).any():
        raise ValueError("Supervised labels outside vocabulary")
    raw = student_logits[:, :-1][mask]
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite supervised student logits")
    work_dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
    logp = F.log_softmax(raw.to(work_dtype), dim=-1)
    ce_sum = -logp.gather(1, targets[:, None]).sum()
    kl_sum = ce_sum.new_zeros(())
    if teacher_logits is not None:
        if teacher_logits.shape != student_logits.shape or teacher_logits.device != student_logits.device:
            raise ValueError("Teacher logits must match student logits")
        teacher_raw = teacher_logits[:, :-1][mask].detach().to(work_dtype)
        if not torch.isfinite(teacher_raw).all():
            raise ValueError("Nonfinite supervised teacher logits")
        teacher_logp = F.log_softmax(teacher_raw, dim=-1)
        kl_sum = (teacher_logp.exp() * (teacher_logp - logp)).sum()
    elif kl_weight:
        raise ValueError("Positive KL coefficient requires frozen teacher logits")
    return TokenLoss(ce_sum, kl_sum, count, kl_weight)


def aggregate_domain_losses(batches):
    """Token-weighted aggregation, not an unweighted mean of batch means."""
    if not batches:
        raise ValueError("At least one batch is required")
    weights = {item.kl_weight for item in batches}
    if len(weights) != 1:
        raise ValueError("CE/KL recipe must be identical across batches")
    total_tokens = sum(item.token_count for item in batches)
    if any(item.token_count <= 0 for item in batches):
        raise ValueError("Invalid token counts")
    return sum(item.ce_sum + item.kl_weight * item.kl_sum for item in batches) / total_tokens
