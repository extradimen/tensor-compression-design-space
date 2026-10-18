"""Independent reference implementation of MPO and matrix factorizations.

Rows are output channels; columns are input channels. Activation samples use
shape [samples, input]. Direct MPO forward never reconstructs the dense matrix.
This is a correctness implementation, not an optimized deployment kernel.
"""
from dataclasses import dataclass
from math import prod

import torch
from torch import nn


@dataclass(frozen=True)
class Layout:
    output_modes: tuple[int, ...]
    input_modes: tuple[int, ...]

    def __post_init__(self):
        if not self.output_modes or len(self.output_modes) != len(self.input_modes):
            raise ValueError("Input and output must have the same nonzero mode count")
        if len(self.input_modes) > 12:
            raise ValueError("Reference einsum supports at most twelve MPO cores")
        if any(type(n) is not int or n < 1 for n in self.output_modes + self.input_modes):
            raise ValueError("Modes must be positive integers")

    @property
    def shape(self):
        return prod(self.output_modes), prod(self.input_modes)

    def tensorize(self, weight):
        if tuple(weight.shape) != self.shape:
            raise ValueError("Weight shape does not match the tensorization")
        k = len(self.input_modes)
        axes = [j for i in range(k) for j in (i, i + k)]
        return weight.reshape(*self.output_modes, *self.input_modes).permute(axes).contiguous()


def _check_weight(weight):
    if weight.ndim != 2 or weight.dtype not in (torch.float32, torch.float64) or not torch.isfinite(weight).all():
        raise ValueError("Initialization needs a finite float32/float64 matrix; convert deployment dtype afterwards")


def _check_scale(scale, weight):
    if scale is None:
        return None
    scale = scale.to(device=weight.device, dtype=weight.dtype)
    if scale.shape != (weight.shape[1],) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Input scale must be finite, positive and match input channels")
    return scale.detach().clone()


def activation_scale(samples, statistic="rms", alpha=1.0, min_scale=1e-4):
    """RMS uses E[z^2], while mean_abs/std reproduce AWT-style statistics.

RMS(alpha=1) is a diagonal second-moment proxy. STD is deliberately centered;
it must not be called the exact diagonal second-moment objective.
"""
    if samples.ndim != 2 or not len(samples) or not torch.isfinite(samples).all():
        raise ValueError("Expected finite nonempty samples")
    if alpha < 0 or min_scale <= 0:
        raise ValueError("Invalid scaling hyperparameters")
    if statistic == "rms":
        importance = samples.square().mean(0).sqrt()
    elif statistic == "mean_abs":
        importance = samples.abs().mean(0)
    elif statistic == "std":
        importance = samples.var(0, unbiased=False).sqrt()
    else:
        raise ValueError("Unknown activation statistic")
    return (importance / importance.mean().clamp_min(1e-8)).pow(alpha).clamp_min(min_scale)


def _rescaled_input(x, scale):
    if scale is None:
        return x
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return (x.to(dtype) / scale.to(dtype)).to(x.dtype)


def unfolding_spectra(weight, layout):
    """True original tensor cut spectra, distinct from TT-SVD residual spectra."""
    tensor = layout.tensorize(weight)
    physical = [m * n for m, n in zip(layout.output_modes, layout.input_modes)]
    return [torch.linalg.svdvals(tensor.reshape(prod(physical[:cut]), -1))
            for cut in range(1, len(physical))]


def tt_svd(weight, layout, ranks):
    _check_weight(weight)
    k = len(layout.input_modes)
    if len(ranks) != k - 1 or any(type(r) is not int or r <= 0 for r in ranks):
        raise ValueError("Supply one positive rank cap per internal bond")
    residual = layout.tensorize(weight)
    left_rank, cores, tails = 1, [], []
    for i, (m, n) in enumerate(zip(layout.output_modes, layout.input_modes)):
        if i == k - 1:
            cores.append(residual.reshape(left_rank, m, n, 1))
            break
        matrix = residual.reshape(left_rank * m * n, -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(ranks[i], len(s))
        tails.append(float(s[rank:].square().sum()))
        cores.append(u[:, :rank].reshape(left_rank, m, n, rank))
        residual = s[:rank, None] * vh[:rank]
        left_rank = rank
    return cores, tails


class MPO(nn.Module):
    def __init__(self, cores, layout, scale=None, bias=None):
        super().__init__()
        if len(cores) != len(layout.input_modes):
            raise ValueError("Core count does not match layout")
        left = 1
        for i, core in enumerate(cores):
            if core.ndim != 4 or core.shape[:3] != (left, layout.output_modes[i], layout.input_modes[i]):
                raise ValueError("Core shape or neighboring bond mismatch")
            left = core.shape[-1]
        if left != 1:
            raise ValueError("Last bond must have dimension one")
        self.layout = layout
        self.cores = nn.ParameterList([nn.Parameter(c.detach().clone()) for c in cores])
        original_scale = scale
        _check_scale(scale, cores[0].new_empty(layout.shape))
        scale = None if original_scale is None else original_scale.detach().clone().to(cores[0].device)
        self.register_buffer("input_scale", scale)
        if bias is not None and bias.shape != (layout.shape[0],):
            raise ValueError("Invalid bias shape")
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_weight(cls, weight, layout, ranks, scale=None, bias=None):
        scale = _check_scale(scale, weight)
        scaled = weight if scale is None else weight * scale
        cores, tails = tt_svd(scaled, layout, ranks)
        module = cls(cores, layout, scale, bias)
        return module, {"tt_svd_discarded_energy": tails,
                        "tt_svd_frobenius_bound_squared": sum(tails),
                        "effective_ranks": [c.shape[-1] for c in cores[:-1]],
                        "metric": "weight_frobenius" if scale is None else "column_scaled_frobenius"}

    def forward(self, x):
        if x.shape[-1] != self.layout.shape[1]:
            raise ValueError("Input feature count mismatch")
        k = len(self.layout.input_modes)
        flat = x.reshape(-1, x.shape[-1])
        cores = [core.to(x.dtype) for core in self.cores]
        pieces = []
        # Exactly two operands at every step: no optional einsum optimizer
        # can combine the cores into a dense weight before applying x. Chunk
        # FP32 division too, bounding scale/contraction inference workspace.
        for chunk in flat.split(64):
            current = _rescaled_input(chunk, self.input_scale).reshape(-1, *self.layout.input_modes)
            labels = [0, *range(1, k + 1)]
            for i, core in enumerate(cores):
                core_labels = [2*k+1+i, k+1+i, 1+i, 2*k+2+i]
                output_labels = [0, *range(k+1, k+2+i)]
                if i < k-1:
                    output_labels += [2*k+2+i, *range(i+2, k+1)]
                current = torch.einsum(current, labels, core, core_labels, output_labels)
                labels = output_labels
            pieces.append(current)
        out = torch.cat(pieces).reshape(*x.shape[:-1], self.layout.shape[0])
        return out if self.bias is None else out + self.bias

    def dense_weight(self):
        """Diagnostic only: return the effective matrix after input rescaling."""
        tensor = self.cores[0].squeeze(0)
        for core in self.cores[1:]:
            tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
        tensor = tensor.squeeze(-1)
        k = len(self.layout.input_modes)
        axes = list(range(0, 2 * k, 2)) + list(range(1, 2 * k, 2))
        weight = tensor.permute(axes).reshape(self.layout.shape)
        return weight if self.input_scale is None else weight / self.input_scale

    def deployment_metadata(self):
        return {"kind": "mpo", "output_modes": self.layout.output_modes,
                "input_modes": self.layout.input_modes,
                "ranks": [c.shape[-1] for c in self.cores[:-1]]}


class MatrixFactors(nn.Module):
    def __init__(self, left, right, scale=None, bias=None):
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("Invalid matrix factor shapes")
        self.left = nn.Parameter(left.detach().clone())
        self.right = nn.Parameter(right.detach().clone())
        original_scale = scale
        _check_scale(scale, right)
        scale = None if original_scale is None else original_scale.detach().clone().to(right.device)
        self.register_buffer("input_scale", scale)
        if bias is not None and bias.shape != (left.shape[0],):
            raise ValueError("Invalid bias shape")
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_weight(cls, weight, rank, scale=None, bias=None):
        _check_weight(weight)
        if type(rank) is not int or not 1 <= rank <= min(weight.shape):
            raise ValueError("Invalid matrix rank")
        scale = _check_scale(scale, weight)
        scaled = weight if scale is None else weight * scale
        u, s, vh = torch.linalg.svd(scaled, full_matrices=False)
        return cls(u[:, :rank] * s[:rank].sqrt(), s[:rank, None].sqrt() * vh[:rank], scale, bias)

    @classmethod
    def coala_projection(cls, weight, samples, rank, ridge=0.0, bias=None):
        """COALA Proposition 1 and isotropic ridge specialization, not full paper reproduction.

Solve min_rank ||(W-W') X^T||_F^2/N + ridge ||W-W'||_F^2
using the left singular subspace of [W X^T/sqrt(N), sqrt(ridge) W].
No Gram matrix inverse; factors deploy as U (U^T W).
"""
        _check_weight(weight)
        if samples.ndim != 2 or samples.shape[1] != weight.shape[1] or not len(samples):
            raise ValueError("Calibration shape mismatch")
        if not torch.isfinite(samples).all() or ridge < 0 or type(rank) is not int or not 1 <= rank <= min(weight.shape):
            raise ValueError("Invalid projection arguments")
        # Streaming QR of the transposed target, then a square SVD. This
        # preserves its left singular subspace without allocating an entire
        # token-by-output SVD factor or squaring the condition number in a Gram
        # eigendecomposition. No matrix inverse and no new tuning parameter.
        triangular = None
        for chunk in samples.split(4096):
            rows = (weight @ chunk.T).T / len(samples) ** 0.5
            if triangular is not None:
                rows = torch.cat([triangular, rows])
            triangular = torch.linalg.qr(rows, mode='r').R
        if ridge:
            triangular = torch.linalg.qr(torch.cat([triangular, ridge ** 0.5 * weight.T]), mode='r').R
        _, _, vh = torch.linalg.svd(triangular, full_matrices=False)
        u = vh[:rank].T
        return cls(u, u.T @ weight, bias=bias)

    def forward(self, x):
        x = _rescaled_input(x, self.input_scale)
        out = (x @ self.right.to(x.dtype).T) @ self.left.to(x.dtype).T
        return out if self.bias is None else out + self.bias

    def dense_weight(self):
        weight = self.left @ self.right
        return weight if self.input_scale is None else weight / self.input_scale

    def deployment_metadata(self):
        return {"kind": "matrix_factors", "rank": self.left.shape[1]}
