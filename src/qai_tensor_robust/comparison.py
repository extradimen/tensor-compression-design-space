"""Validate representation-independent paired comparison contracts.

This does not schedule jobs, reserve GPUs, or replace the shared controller.
ComparisonContract is a historical synthetic strict-cap contract. The recovered
v1.1 uses ExposureContract: equal exposure and a separate preparation-time cap;
actual FLOPs are recorded and are never asserted equal from equal token counts.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ComparisonContract:
    model_revision: str
    data_fingerprint: str
    layer_names: tuple[str, ...]
    whole_model_cap_bytes: int
    objective_fingerprint: str
    teacher_targets_fingerprint: str
    calibration_tokens: int
    repair_tokens: int
    optimizer_steps: int
    optimizer_fingerprint: str
    repair_flops_cap: int
    repair_walltime_cap_seconds: int
    search_trials: int
    checkpoint_rule: str
    seed: int

    def validate(self):
        text_fields = {'model_revision', 'data_fingerprint', 'objective_fingerprint',
                       'teacher_targets_fingerprint', 'optimizer_fingerprint', 'checkpoint_rule'}
        for key, value in asdict(self).items():
            if key in text_fields and (not isinstance(value, str) or not value):
                raise ValueError(f"Unresolved comparison field: {key}")
            if key not in text_fields and key != 'layer_names':
                if type(value) is not int or (key != 'seed' and value < 0):
                    raise ValueError(f"Invalid integer comparison field: {key}")
        if not isinstance(self.layer_names, tuple) or not self.layer_names or any(not isinstance(n, str) or not n for n in self.layer_names) or len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("Layer set must be nonempty and unique")
        if self.whole_model_cap_bytes <= 0:
            raise ValueError("Whole-model byte cap must be positive")


def require_identical_contract(left, right):
    left.validate()
    right.validate()
    differences = [k for k, v in asdict(left).items() if v != asdict(right)[k]]
    if differences:
        raise ValueError("Unmatched comparison: " + ', '.join(differences))
    return True


@dataclass(frozen=True)
class ExposureContract:
    model_revision: str
    data_fingerprint: str
    layer_names: tuple[str, ...]
    whole_model_cap_bytes: int
    objective_fingerprint: str
    teacher_targets_fingerprint: str
    teacher_entropy_fingerprint: str
    batch_manifest_fingerprint: str
    calibration_tokens: int
    repair_tokens: int
    optimizer_steps: int
    optimizer_fingerprint: str
    preparation_gpu_seconds_ceiling: int
    repair_walltime_cap_seconds: int
    search_trials: int
    checkpoint_rule: str
    seed: int

    def validate(self):
        strings = {k for k, v in asdict(self).items() if isinstance(v, str)}
        required_strings = {"model_revision", "data_fingerprint", "objective_fingerprint",
                            "teacher_targets_fingerprint", "teacher_entropy_fingerprint",
                            "batch_manifest_fingerprint", "optimizer_fingerprint", "checkpoint_rule"}
        if not required_strings.issubset(strings) or any(not getattr(self, k) for k in required_strings):
            raise ValueError("Unresolved exposure/teacher/batch identity")
        for key, value in asdict(self).items():
            if key not in required_strings and key != "layer_names":
                if type(value) is not int or (key != "seed" and value < 0):
                    raise ValueError("Invalid exposure contract field: " + key)
        if self.whole_model_cap_bytes < 1 or not self.layer_names or len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("Whole-model byte cap and unique layer set required")
        if self.checkpoint_rule != "final":
            raise ValueError("Protocol primary checkpoint must be final")
        if self.repair_tokens > 65536:
            raise ValueError("Pilot repair input-token cap exceeded")
