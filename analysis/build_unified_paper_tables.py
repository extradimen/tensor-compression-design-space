"""Summarize existing development evidence; no model execution or new evaluation."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def read(name):
    return json.loads((OUT / name).read_text())

reg = read("COMPARISON_DATA_COLLECTION_REGISTER_20261017_v42.json")
quality = read("STEP5_P85_FULL36_THREE_SEED_QUALITY_MATRIX_20261018.json")
remote = read("STEP5_P85_FULL36_REMOTE_DEPLOYED_BYTES_AUDIT_20261018.json")
old = {}
for row in reg["rows"]:
    if row["grid"] == "step5_tucker_reference36":
        for seed in row.get("per_seed_summary_evidence") or []:
            old[(row["cell_id"], seed["seed"])] = seed
new = {(r["cell_id"], r["seed"]): r for r in remote["new_run_final_payloads"]}

def med(xs):
    return statistics.median(xs) if xs else None

def new_metrics(entry):
    full = json.loads((ROOT / entry["complete_json_path"]).read_text())
    matching = [b for b in full["blocks"] if isinstance(b.get("payload"), dict)
                and b["payload"].get("sha256") == entry["final_model_sha256"]]
    assert matching, entry
    block = matching[-1]
    records = block["runtime"]["fixed_decode"]["records"]
    assert len(records) == 5
    return {
        "deployed_bytes": entry["final_deployed_bytes"],
        "decode_seconds_median": med([r["task_seconds"] for r in records]),
        "decode_cuda_allocated_peak_bytes": max(r["memory_peak_bytes"]["cuda_allocated"] for r in records),
        "decode_board_joules_median": med([r.get("board_energy_joules") for r in records if r.get("board_energy_joules") is not None]),
        "preparation_wall_seconds": block["preparation_wall_seconds"],
        "quality_wall_seconds": block["quality_wall_seconds"],
        "preparation_peak_cuda_bytes": block["preparation_peak_cuda_bytes"],
        "source": entry["complete_json_path"],
        "payload_sha256": entry["final_model_sha256"],
        "payload_proof": "same_project_campus_independent_sha",
        "vintage": "protocol85_new",
    }

def old_metrics(entry):
    return {
        "deployed_bytes": entry["payload_bytes_reported"],
        "decode_seconds_median": entry.get("fixed_decode_median_seconds"),
        "decode_cuda_allocated_peak_bytes": entry.get("fixed_decode_peak_allocated_bytes"),
        "decode_board_joules_median": entry.get("fixed_decode_median_board_joules"),
        "preparation_wall_seconds": entry.get("preparation_wall_seconds"),
        "quality_wall_seconds": entry.get("quality_evaluation_seconds"),
        "preparation_peak_cuda_bytes": entry.get("preparation_peak_cuda_bytes"),
        "source": entry["source_path"],
        "payload_sha256": entry.get("payload_sha256_reported"),
        "payload_proof": "mac_local_independent_sha" if entry.get("payload_local_sha_verified") else "historical_reported",
        "vintage": "historical",
    }

rows = []
for cell in quality["cells"]:
    seeds = []
    for q in cell["seed_results"]:
        key = (cell["cell_id"], q["seed"])
        assert (key in old) != (key in new), key
        m = old_metrics(old[key]) if key in old else new_metrics(new[key])
        seeds.append({"seed": q["seed"], "nll_text": q["domain_nll"]["text"],
                      "nll_math": q["domain_nll"]["math"], **m})
    assert len(seeds) == 3
    axes = ["nll_text", "nll_math", "deployed_bytes", "decode_seconds_median",
            "decode_cuda_allocated_peak_bytes", "decode_board_joules_median",
            "preparation_wall_seconds", "quality_wall_seconds", "preparation_peak_cuda_bytes"]
    rows.append({"cell_id": cell["cell_id"], "coordinates": cell["coordinates"],
                 "seed_rows": seeds, "descriptive_median": {a: med([s[a] for s in seeds if s[a] is not None]) for a in axes}})

assert len(rows) == 36 and sum(len(r["seed_rows"]) for r in rows) == 108
assert sum(s["vintage"] == "historical" for r in rows for s in r["seed_rows"]) == 54
assert sum(s["vintage"] == "protocol85_new" for r in rows for s in r["seed_rows"]) == 54
doc = {"project_id": quality["project_id"], "scope": "fixed-development descriptive aggregation, not independent formal test",
       "denominator": {"cells": 36, "nominal_seed_bindings": 108, "historical": 54, "protocol85_new": 54},
       "measurement_warning": "Nominal seeds need not be independent. Runtime, memory, board energy and preparation times mix historical and Protocol85 runs; cross-vintage differences are observational. Campus SHA is not Mac-local SHA.",
       "rows": rows}
(OUT / "STEP5_P85_UNIFIED_QUALITY_COST_TABLE_20261018.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
initial = []
for r in reg["rows"]:
    if r["grid"] == "initial36":
        initial.append({"cell_id": r["cell_id"], "coordinates": r["coordinates"],
                        "nll": r["quality"]["domain_nll"], "cost": r["cost"], "source": r["source"]})
assert len(initial) == 36
(OUT / "INITIAL36_UNIFIED_QUALITY_COST_TABLE_20261018.json").write_text(json.dumps({
    "scope": "initial36 single-source development endpoints; do not pool with Step5 nominal seeds",
    "denominator": 36, "rows": initial}, ensure_ascii=False, indent=2) + "\n")
print("initial36=36; step5=36 cells, 108 bindings; historical=54 new=54")
