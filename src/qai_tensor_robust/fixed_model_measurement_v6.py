"""CPU-testable measurement boundaries; no scheduler or model acquisition.

Caller supplies an already-loaded frozen deployment model through a workload.
CUDA sync, memory counters, and power telemetry are injected by the remote adapter.
This module alone does not qualify a GPU measurement or authorize a launch.
"""
import math
import time


def _finite_nonnegative(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def integrate_board_energy(samples, start, end, max_gap_seconds=1.0):
    """Trapezoidal joules, interpolated window edges, no missing-as-zero."""
    if not all(_finite_nonnegative(x) for x in (start, end, max_gap_seconds)) or end <= start or max_gap_seconds <= 0:
        raise ValueError('Invalid measurement window')
    if len(samples) < 3:
        raise ValueError('Insufficient power samples')
    pairs = [(s['monotonic_seconds'], s['watts']) for s in samples]
    if any(not _finite_nonnegative(t) or not _finite_nonnegative(w) for t, w in pairs):
        raise ValueError('Invalid power samples')
    if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError('Power timestamps must increase')
    if pairs[0][0] > start or pairs[-1][0] < end:
        raise ValueError('Power samples do not cover task window')
    energy = 0.0
    gaps = []
    for (t0, w0), (t1, w1) in zip(pairs, pairs[1:]):
        left, right = max(start, t0), min(end, t1)
        if right <= left:
            continue
        gap = t1 - t0
        if gap > max_gap_seconds:
            raise ValueError('Power telemetry gap exceeds protocol')
        gaps.append(gap)
        wl = w0 + (w1 - w0) * (left - t0) / gap
        wr = w0 + (w1 - w0) * (right - t0) / gap
        energy += (wl + wr) * 0.5 * (right - left)
    return {'board_energy_joules': energy, 'board_energy_kWh': energy / 3600000,
            'sample_count': len(samples), 'maximum_sample_gap_seconds': max(gaps),
            'scope': 'assigned GPU board, not whole-system or exact process energy'}


def measure_fixed_workload(workload, synchronize, reset_memory_peak,
                           read_memory_peak, *, warmups=2, repetitions=5,
                           clock=time.perf_counter):
    """Return every timed repeat. Workload returns identical integer work counts.

    Construction, loading, warmup, reporting and energy reduction stay outside
    timed task windows. No memory-baseline subtraction is performed.
    """
    if type(warmups) is not int or warmups < 0 or type(repetitions) is not int or repetitions < 1:
        raise ValueError('Invalid repeat counts')
    for _ in range(warmups):
        workload()
        synchronize()
    records, expected_counts = [], None
    for repeat in range(repetitions):
        synchronize()
        reset_memory_peak()
        start = clock()
        counts = workload()
        synchronize()
        end = clock()
        memory = read_memory_peak()
        if not _finite_nonnegative(start) or not _finite_nonnegative(end) or end <= start:
            raise ValueError('Invalid synchronized task duration')
        if not isinstance(counts, dict) or not counts or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError('Workload must return explicit integer work counts')
        if expected_counts is None:
            expected_counts = dict(counts)
        if counts != expected_counts:
            raise ValueError('Work differs between repeated task windows')
        if not isinstance(memory, dict) or any(not _finite_nonnegative(v) for v in memory.values()):
            raise ValueError('Invalid memory counters')
        duration = end - start
        records.append({'repeat': repeat, 'start_monotonic_seconds': start,
                        'end_monotonic_seconds': end, 'task_seconds': duration,
                        'work_counts': dict(counts), 'work_units_per_second':
                        {k: v / duration for k, v in counts.items()},
                        'memory_peak_bytes': dict(memory),
                        'energy_status': 'MISSING_UNTIL_WINDOW_TELEMETRY_BOUND'})
    return {'timing_boundary': 'already loaded/frozen/warmed; GPU sync before and after task',
            'warmups_excluded': warmups, 'repetitions': repetitions, 'records': records,
            'preparation_cost_included': False, 'scientific_GPU_measurement_claimed': False}


def validate_complete_point(point):
    """Missing runtime metrics fail completeness instead of disappearing."""
    for key in ('point_id', 'model_sha256', 'protocol_manifest_sha256', 'coordinates',
                'actual_whole_bytes', 'quality_by_domain', 'fixed_runtime'):
        if key not in point:
            raise ValueError('Missing point identity/measurement: ' + key)
    if set(point['quality_by_domain']) != {'text', 'math'}:
        raise ValueError('Separate text/math quality required')
    for task in ('search_nll', 'fixed_decode'):
        report = point['fixed_runtime'].get(task, {})
        if report.get('scientific_GPU_measurement_claimed') is not True:
            raise ValueError('GPU production evidence absent: ' + task)
        if report.get('preparation_cost_included') is not False or len(report.get('records', [])) != 5:
            raise ValueError('Incorrect task boundary or repeat count')
        for row in report['records']:
            if (not _finite_nonnegative(row.get('task_seconds')) or row['task_seconds'] <= 0
                    or not row.get('memory_peak_bytes') or not row.get('work_counts')
                    or row.get('energy_status') != 'MEASURED_BOARD_ENERGY'
                    or not _finite_nonnegative(row.get('board_energy_joules'))):
                raise ValueError('Missing/invalid required runtime metric')
    return True
