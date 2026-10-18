import unittest

from qai_tensor_robust.fixed_model_measurement_v6 import (
    integrate_board_energy, measure_fixed_workload, validate_complete_point,
)


class FixedMeasurementTests(unittest.TestCase):
    def test_energy_linear_power_exact_window(self):
        samples = [{'monotonic_seconds': t, 'watts': 100 + 20*t} for t in (0., .5, 1.)]
        self.assertAlmostEqual(integrate_board_energy(samples, .25, .75)['board_energy_joules'], 55.)

    def test_missing_and_gapped_energy_rejected(self):
        for samples in ([], [{'monotonic_seconds': t, 'watts': 100.} for t in (0., 2., 3.)]):
            with self.assertRaises(ValueError):
                integrate_board_energy(samples, .1, .9)

    def test_window_requires_full_coverage(self):
        samples = [{'monotonic_seconds': t, 'watts': 100.} for t in (.2, .5, .8)]
        with self.assertRaises(ValueError):
            integrate_board_energy(samples, 0., 1.)

    def test_sync_boundary_and_no_preparation_time(self):
        events = []
        values = iter((10., 12., 20., 22.))
        def work():
            events.append('work')
            return {'generated_tokens': 32}
        def sync(): events.append('sync')
        def reset(): events.append('reset')
        def memory():
            events.append('memory')
            return {'allocated': 100, 'reserved': 200}
        def clock():
            events.append('clock')
            return next(values)
        report = measure_fixed_workload(work, sync, reset, memory, warmups=1,
                                        repetitions=2, clock=clock)
        self.assertEqual(events, ['work','sync'] + ['sync','reset','clock','work','sync','clock','memory']*2)
        self.assertEqual([r['task_seconds'] for r in report['records']], [2.,2.])
        self.assertEqual(report['records'][0]['work_units_per_second']['generated_tokens'], 16.)
        self.assertFalse(report['preparation_cost_included'])
        self.assertFalse(report['scientific_GPU_measurement_claimed'])

    def test_changed_workload_is_not_a_fair_repeat(self):
        counts = iter(({'tokens': 1}, {'tokens': 2}))
        timer = iter((0.,1.,2.,3.))
        with self.assertRaises(ValueError):
            measure_fixed_workload(lambda: next(counts), lambda: None, lambda: None,
                lambda: {'allocated':1}, warmups=0, repetitions=2, clock=lambda: next(timer))

    def test_cpu_report_does_not_pass_GPU_completeness(self):
        with self.assertRaises(ValueError):
            validate_complete_point({'point_id':'toy', 'model_sha256':'toy',
                'protocol_manifest_sha256':'toy','coordinates':{},'actual_whole_bytes':1,
                'quality_by_domain':{'text':{},'math':{}},'fixed_runtime':{}})


if __name__ == '__main__':
    unittest.main()
