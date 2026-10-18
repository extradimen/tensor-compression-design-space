"""Metrics and representation-independent repair objectives."""
import torch


def second_moment(samples):
    if samples.ndim != 2 or not len(samples) or not torch.isfinite(samples).all():
        raise ValueError("Expected finite nonempty [samples, channels]")
    return samples.T @ samples / len(samples)


def functional_error(error, samples):
    return (samples @ error.T).square().sum(-1).mean()


def moment_error(error, moment):
    return ((error @ moment) * error).sum()


def domain_repair_objective(student_means, dense_means, mode, weights=None):
    """Use identical relative excess loss for both representations.

Dense means must be frozen, detached and computed on the same samples. A
held-out domain must not appear in these dictionaries. Max is exact (not LSE).
This function accepts task losses; it does not substitute layer MSE for CE/KL.
"""
    if not student_means or set(student_means) != set(dense_means):
        raise ValueError("Student and dense domains must match and be nonempty")
    names = sorted(student_means)
    values = []
    for name in names:
        s = student_means[name]
        d = torch.as_tensor(dense_means[name], dtype=s.dtype, device=s.device).detach()
        if s.numel() != 1 or d.numel() != 1 or not torch.isfinite(s).all() or not torch.isfinite(d).all() or d <= 0:
            raise ValueError("Expected scalar finite losses and positive dense means")
        values.append((s - d) / d)
    excess = torch.stack(values)
    if mode == "worst_domain":
        return excess.max()
    if mode != "mixture":
        raise ValueError("Unknown repair objective")
    if weights is None:
        coeff = torch.ones_like(excess)
    else:
        if set(weights) != set(names):
            raise ValueError("Weight domains differ")
        coeff = excess.new_tensor([weights[n] for n in names])
    if not torch.isfinite(coeff).all() or (coeff < 0).any() or coeff.sum() <= 0:
        raise ValueError("Invalid mixture weights")
    return (excess * coeff / coeff.sum()).sum()
