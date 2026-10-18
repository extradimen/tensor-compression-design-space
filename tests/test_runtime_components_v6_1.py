import unittest
from qai_tensor_robust.runtime_components_v6_1 import summarize_forward_components,validate_worker_uuid


class ComponentsTests(unittest.TestCase):
    def records(self):
        return ([{'input_length':512,'cuda_elapsed_ms':10.}]
            +[{'input_length':1,'cuda_elapsed_ms':2.} for _ in range(31)])*2
    def test_fixed_generation_component_units_and_counts(self):
        d=summarize_forward_components(self.records())
        self.assertAlmostEqual(d['prefill_forward_cuda_seconds'],.020)
        self.assertAlmostEqual(d['decode_forward_cuda_seconds'],.124)
        self.assertEqual(d['generated_tokens'],64)
        self.assertEqual(d['decode_forward_calls'],62)
    def test_wrong_cache_shape_rejected(self):
        rows=self.records();rows[2]={'input_length':512,'cuda_elapsed_ms':2.}
        with self.assertRaises(ValueError):summarize_forward_components(rows)
    def test_early_stop_rejected(self):
        with self.assertRaises(ValueError):summarize_forward_components(self.records()[:-1])
    def test_nonfinite_component_rejected(self):
        rows=self.records();rows[2]={'input_length':1,'cuda_elapsed_ms':float('nan')}
        with self.assertRaises(ValueError):summarize_forward_components(rows)
    def test_two_workers_are_independent_single_UUIDs(self):
        rows=[{'pid':11,'gpu_uuid':'GPU-a'},{'pid':12,'gpu_uuid':'GPU-b'}]
        self.assertEqual(validate_worker_uuid('GPU-a','GPU-a',1,rows,11)['actual_physical_GPU_UUID'],'GPU-a')
        self.assertEqual(validate_worker_uuid('GPU-b','GPU-b',1,rows,12)['actual_physical_GPU_UUID'],'GPU-b')
    def test_ticket_and_actual_device_mismatch_rejected(self):
        with self.assertRaises(ValueError):validate_worker_uuid('GPU-a','GPU-b',1,[],11)
        with self.assertRaises(ValueError):validate_worker_uuid('GPU-a','GPU-a',1,[{'pid':11,'gpu_uuid':'GPU-b'}],11)
    def test_other_process_or_multi_visible_board_rejected(self):
        rows=[{'pid':11,'gpu_uuid':'GPU-a'},{'pid':12,'gpu_uuid':'GPU-a'}]
        with self.assertRaises(ValueError):validate_worker_uuid('GPU-a','GPU-a',1,rows,11)
        with self.assertRaises(ValueError):validate_worker_uuid('GPU-a','GPU-a',2,rows[:1],11)


if __name__=='__main__':unittest.main()
