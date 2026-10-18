"""Non-evidential consistency checks for released aggregate tables."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / 'data'

def read(name):
    return json.loads((data / name).read_text())

with (data / 'PAPER_TABLE_STEP5_MATCHED_LAYOUT_54_BINDINGS_20261018.csv').open(newline='') as stream:
    layout = list(csv.DictReader(stream))
with (data / 'PAPER_TABLE_PREDEFINED_CONFIRMATION_TESTS_20261018.csv').open(newline='') as stream:
    confirm = list(csv.DictReader(stream))
initial = read('INITIAL36_UNIFIED_QUALITY_COST_TABLE_20261018.json')
step5 = read('STEP5_P85_UNIFIED_QUALITY_COST_TABLE_20261018.json')
p86 = read('PROTOCOL86_INDEPENDENT_CONFIRMATION_20261018.json')
p87 = read('PROTOCOL87_STRUCTURE_OBJECTIVE_REPAIR_CONFIRMATION_20261018.json')
seed = read('STEP5_P85_NOMINAL_SEED_IDENTIFIABILITY_AUDIT_20261018.json')
assert len(initial['rows']) == 36
assert len(step5['rows']) == 36
assert sum(len(row['seed_rows']) for row in step5['rows']) == 108
assert len(layout) == 54
assert len({(r['semantic_cell'],r['alternative_cell']) for r in layout}) == 18
assert len(confirm) == 14
assert {int(r['protocol']) for r in confirm} == {86,87}
assert all(int(r['alt_minus_sem_deployed_bytes']) > 0 for r in layout)
assert all(r['passes_predefined_direction'] == 'True' for r in confirm)
assert p86['all_four_directions_reproduced'] is True
assert p87['all_six_primary_directions_reproduced'] is True
assert p87['all_four_secondary_repair_directions_reproduced'] is True
print('Verified: initial 36; Step5 36 cells/108 labels; 18 layout pairs/54 bindings; 14 predefined confirmation rows.')
print('These are aggregate checks, not a re-analysis of withheld source-cluster inputs.')
