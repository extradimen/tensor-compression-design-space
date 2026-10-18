"""Frozen Protocol87 source-cluster bootstrap for structure/objective/repair."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "research/inputs/sealed/Confirm87.jsonl"
RELEASE = ROOT / "research/confirm87/release.json"
RESULTS = ROOT / "research/results/gpu"
OUT = ROOT / "outputs/PROTOCOL87_STRUCTURE_OBJECTIVE_REPAIR_CONFIRMATION_20261018.json"
INPUT_SHA = "0b66f88671c0c9777efcd9640b6fbf289965033c4b8b54ab74c55ea7a52b06f2"
RELEASE_SHA = "c664a17331febbd6cb3bdf2b051457fc83125b63384b00e3e633343fcf9e85f5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(model_id, expected, release):
    folder = RESULTS / ("confirm87-" + model_id)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for name, spec in manifest["files"].items():
        path = folder / name
        if not path.is_file() or sha(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("Return SHA or size mismatch: " + model_id + "/" + name)
    data = json.loads((folder / "complete.json").read_text())
    config_path = ROOT / "research/confirm87/configs" / (model_id + ".json")
    if (data["status"] != "COMPLETE" or data["model_id"] != model_id or
        data["input_sha256"] != INPUT_SHA or data["config_sha256"] != sha(config_path) or
        data["model_sha256"] != release["models"][model_id]["payload"]["sha256"]):
        raise ValueError("Wrong result identity: " + model_id)
    rows = data["rows"]
    if len(rows) != 160 or {r["unit_id"] for r in rows} != set(expected):
        raise ValueError("Wrong result denominator: " + model_id)
    if len(rows) != len({r["unit_id"] for r in rows}):
        raise ValueError("Duplicate result unit: " + model_id)
    for row in rows:
        source = expected[row["unit_id"]]
        if any(row[key] != source[key] for key in ("domain", "cluster_id", "content_sha256", "source_revision", "source_split")):
            raise ValueError("Wrong source identity: " + model_id)
        if row["target_count"] <= 0 or row["nll_sum"] < 0:
            raise ValueError("Invalid per-unit NLL: " + model_id)
    return {r["unit_id"]: r for r in rows}, {"model_id": model_id,
        "complete_sha256": sha(folder / "complete.json"),
        "manifest_sha256": sha(folder / "MANIFEST.json"),
        "elapsed_seconds": data["elapsed_seconds"]}


def main():
    if sha(INPUT) != INPUT_SHA or sha(RELEASE) != RELEASE_SHA:
        raise ValueError("Changed frozen input or release")
    release = json.loads(RELEASE.read_text())
    frozen = [json.loads(s) for s in INPUT.read_text().splitlines() if s.strip()]
    expected = {x["unit_id"]: x for x in frozen}
    if len(frozen) != len(expected) or len(frozen) != 160:
        raise ValueError("Wrong input denominator")
    by_domain = {d: [x["unit_id"] for x in frozen if x["domain"] == d] for d in ("text", "math")}
    if [len(by_domain[d]) for d in ("text", "math")] != [32, 128]:
        raise ValueError("Wrong domain counts")
    models, receipts = {}, []
    for family in ("matrix", "mpo"):
        for objective in ("O-W", "O-D", "O-C"):
            for repair in ("none", "output"):
                ident = f"{family}__{objective}__exchange2__{repair}"
                models[ident], receipt = load(ident, expected, release)
                receipts.append(receipt)
    rng = np.random.default_rng(87001)
    domains = {}
    for domain, ids in by_domain.items():
        if len({expected[u]["cluster_id"] for u in ids}) != len(ids):
            raise ValueError("Nonindependent clusters")
        values = {ident: np.array([models[ident][u]["nll_sum"] / models[ident][u]["target_count"]
                                  for u in ids], dtype=np.float64) for ident in models}
        target_counts = {ident: [models[ident][u]["target_count"] for u in ids] for ident in models}
        if len({tuple(x) for x in target_counts.values()}) != 1:
            raise ValueError("Target denominator differs by model")
        paired = {}
        for obj in ("O-W", "O-D", "O-C"):
            for repair in ("none", "output"):
                paired[f"{obj}_{repair}"] = values[f"mpo__{obj}__exchange2__{repair}"] - values[f"matrix__{obj}__exchange2__{repair}"]
        effects = {"OW_none": paired["O-W_none"], "OD_none": paired["O-D_none"],
                   "OC_none": paired["O-C_none"],
                   "OD_repair_attenuation": paired["O-D_output"] - paired["O-D_none"],
                   "OC_repair_attenuation": paired["O-C_output"] - paired["O-C_none"]}
        index = rng.integers(0, len(ids), size=(10000, len(ids)))
        report = {"source_clusters": len(ids), "cell_mean_nll": {k: float(v.mean()) for k, v in values.items()},
                  "all_paired_mean_mpo_minus_matrix": {k: float(v.mean()) for k, v in paired.items()}}
        tests = {}
        for name, value in effects.items():
            draws = value[index].mean(axis=1)
            direction = "negative" if name in ("OW_none", "OD_repair_attenuation", "OC_repair_attenuation") else "positive"
            bound = float(np.quantile(draws, 0.995 if direction == "negative" else 0.005))
            tests[name] = {"mean": float(value.mean()), "direction": direction,
                "one_sided_99_5pct_bound": bound,
                "passes": bound < 0 if direction == "negative" else bound > 0,
                "two_sided_95pct_percentile": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}
        report["predefined_tests"] = tests
        domains[domain] = report
    primary = all(domains[d]["predefined_tests"][k]["passes"]
                  for d in ("text", "math") for k in ("OW_none", "OD_none", "OC_none"))
    secondary = all(domains[d]["predefined_tests"][k]["passes"]
                    for d in ("text", "math") for k in ("OD_repair_attenuation", "OC_repair_attenuation"))
    result = {"schema_version": 1, "project_id": "bac14391-c670-45aa-891f-2a0aab49abc7",
        "protocol": "Protocol87", "input_sha256": INPUT_SHA, "release_sha256": RELEASE_SHA,
        "source_cluster_bootstrap": {"draws": 10000, "seed": 87001, "familywise_alpha": 0.05,
            "predefined_one_sided_claims": 10, "each_one_sided_bound": 0.995},
        "model_runs": receipts, "domains": domains,
        "all_six_primary_directions_reproduced": primary,
        "all_four_secondary_repair_directions_reproduced": secondary,
        "scope": "One model and rank setting; matrix vs MPO bundles factorization and layout/execution; project-heldout train-source",
        "pretraining_contamination_excluded": False}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"primary": primary, "secondary": secondary, "output": str(OUT), "sha256": sha(OUT)}))


if __name__ == "__main__":
    main()
