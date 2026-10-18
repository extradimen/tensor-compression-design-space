"""Frozen Protocol86 source-cluster bootstrap over fully returned model results."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "research/inputs/sealed/Confirm86.jsonl"
RESULTS = ROOT / "research/results/gpu"
OUT = ROOT / "outputs/PROTOCOL86_INDEPENDENT_CONFIRMATION_20261018.json"
CELLS = {"D_sem": "step5-08", "D_alt": "step5-26",
         "C_sem": "step5-14", "C_alt": "step5-32"}
SEEDS = (1103, 1104, 1105)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(model_id, expected, input_sha, release):
    folder = RESULTS / ("confirm86-" + model_id)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for name, spec in manifest["files"].items():
        path = folder / name
        if not path.is_file() or sha(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("Return SHA or size mismatch: " + model_id + "/" + name)
    data = json.loads((folder / "complete.json").read_text())
    if data["status"] != "COMPLETE" or data["model_id"] != model_id or data["input_sha256"] != input_sha:
        raise ValueError("Wrong result identity: " + model_id)
    spec = release["models"][model_id]
    expected_model_sha = spec["payload"]["sha256"] if spec["kind"] != "native" else spec["files"][0]["sha256"]
    config_path = ROOT / "research/confirm86/configs" / (model_id + ".json")
    if data["model_sha256"] != expected_model_sha or data["config_sha256"] != sha(config_path):
        raise ValueError("Wrong model or config SHA: " + model_id)
    rows = data["rows"]
    if len(rows) != 160 or {r["unit_id"] for r in rows} != set(expected):
        raise ValueError("Wrong result denominator: " + model_id)
    if len(rows) != len({r["unit_id"] for r in rows}):
        raise ValueError("Duplicate result unit: " + model_id)
    for row in rows:
        source = expected[row["unit_id"]]
        if any(row[key] != source[key] for key in ("domain", "cluster_id", "content_sha256", "source_revision", "source_split")):
            raise ValueError("Wrong source identity: " + model_id)
    return {r["unit_id"]: r for r in rows}, {"model_id": model_id,
        "complete_sha256": sha(folder / "complete.json"),
        "manifest_sha256": sha(folder / "MANIFEST.json"),
        "elapsed_seconds": data["elapsed_seconds"]}


def mean_and_boot(values, indices):
    return float(values.mean()), values[indices].mean(axis=1)


def main():
    frozen = [json.loads(s) for s in INPUT.read_text().splitlines() if s.strip()]
    input_sha = sha(INPUT)
    if input_sha != "9780191cfdf5eb64ee80304f9bb882d1a5e614f64c828022b3ceea437baa5293":
        raise ValueError("Changed frozen input")
    if len(frozen) != 160 or len({x["unit_id"] for x in frozen}) != 160:
        raise ValueError("Invalid input denominator")
    expected_ids = {x["unit_id"]: x for x in frozen}
    release_path = ROOT / "research/confirm86/release.json"
    if sha(release_path) != "38123a67da3285758e076c02274bc7b2704729b37083382b9363a2e9d2cf777c":
        raise ValueError("Changed candidate release")
    release = json.loads(release_path.read_text())
    ids_by_domain = {domain: [x["unit_id"] for x in frozen if x["domain"] == domain]
                     for domain in ("text", "math")}
    if tuple(map(len, ids_by_domain.values())) != (32, 128):
        raise ValueError("Wrong domain denominator")
    measures = {}
    receipt = []
    for cell, prefix in CELLS.items():
        rows = []
        for seed in SEEDS:
            model_id = f"{prefix}-s{seed}"
            data, identity = load(model_id, expected_ids, input_sha, release)
            rows.append(data)
            receipt.append(identity)
        measures[cell] = rows
    native, identity = load("native", expected_ids, input_sha, release)
    receipt.append(identity)
    rng = np.random.default_rng(86001)
    domains = {}
    for domain, unit_ids in ids_by_domain.items():
        if len({(x["domain"], x["cluster_id"]) for x in frozen if x["domain"] == domain}) != len(unit_ids):
            raise ValueError("Nonindependent source clusters")
        score = {}
        for cell, reps in measures.items():
            score[cell] = np.array([np.mean([r[u]["nll_sum"] / r[u]["target_count"]
                                             for r in reps]) for u in unit_ids], dtype=np.float64)
        native_score = np.array([native[u]["nll_sum"] / native[u]["target_count"] for u in unit_ids])
        # The same source-cluster draws are used for every layout and objective.
        indices = rng.integers(0, len(unit_ids), size=(10000, len(unit_ids)))
        delta_d = score["D_alt"] - score["D_sem"]
        delta_c = score["C_alt"] - score["C_sem"]
        interaction = delta_c - delta_d
        dd, dd_boot = mean_and_boot(delta_d, indices)
        dc, dc_boot = mean_and_boot(delta_c, indices)
        inter, inter_boot = mean_and_boot(interaction, indices)
        upper_d = float(np.quantile(dd_boot, 0.9875))
        lower_c = float(np.quantile(dc_boot, 0.0125))
        domains[domain] = {
            "source_clusters": len(unit_ids),
            "cell_seed_mean_nll": {k: float(v.mean()) for k, v in score.items()},
            "native_context_nll": float(native_score.mean()),
            "delta_D_alt_minus_sem": dd,
            "delta_C_alt_minus_sem": dc,
            "interaction_C_minus_D": inter,
            "one_sided_98_75pct_bound_D_upper": upper_d,
            "one_sided_98_75pct_bound_C_lower": lower_c,
            "interaction_two_sided_95pct_percentile": [float(np.quantile(inter_boot, 0.025)),
                                                      float(np.quantile(inter_boot, 0.975))],
            "D_reproduced": upper_d < 0,
            "C_reproduced": lower_c > 0,
        }
    complete = all(domains[d][k] for d in ("text", "math") for k in ("D_reproduced", "C_reproduced"))
    report = {"schema_version": 1, "project_id": "bac14391-c670-45aa-891f-2a0aab49abc7",
              "protocol": "Protocol86", "input_sha256": input_sha,
              "unit": "new source clusters from heldout project train-source pool",
              "model_runs": receipt, "bootstrap": {"seed": 86001, "draws": 10000,
                  "resampling_unit": "source cluster", "family": "four directional claims",
                  "familywise_alpha": 0.05, "one_sided_bound": 0.9875},
              "domains": domains, "all_four_directions_reproduced": complete,
              "scope": "One model, one fixed repair/update/capacity setting; not an external benchmark test or cross-model confirmation",
              "pretraining_contamination_excluded": False}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"complete": complete, "output": str(OUT), "sha256": sha(OUT)}))


if __name__ == "__main__":
    main()
