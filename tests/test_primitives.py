import hashlib
import json
import platform
import time
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from qai_tensor_robust.accounting import WholeModelBudget, decode_payload, deployment_payload
from qai_tensor_robust.comparison import ComparisonContract, require_identical_contract
from qai_tensor_robust.objectives import domain_repair_objective, functional_error, moment_error, second_moment
from qai_tensor_robust.operators import Layout, MatrixFactors, MPO, activation_scale, unfolding_spectra


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.w = torch.randn(12, 24, dtype=torch.float64)
        self.x = torch.randn(13, 24, dtype=torch.float64)
        self.layout = Layout((2, 3, 2), (3, 2, 4))

    def test_exact_mpo_and_arbitrary_batch_shape(self):
        mpo, _ = MPO.from_weight(self.w, self.layout, [100, 100])
        torch.testing.assert_close(mpo.dense_weight(), self.w, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(mpo(self.x), self.x @ self.w.T, rtol=1e-11, atol=1e-11)
        x = self.x[:12].reshape(2, 6, 24)
        torch.testing.assert_close(mpo(x), x @ self.w.T, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(mpo(self.x[0]), self.x[0] @ self.w.T, rtol=1e-11, atol=1e-11)

    def test_second_layout_and_single_core(self):
        for layout in (Layout((3, 4), (6, 4)), Layout((12,), (24,))):
            m, _ = MPO.from_weight(self.w, layout, [100] * (len(layout.input_modes) - 1))
            torch.testing.assert_close(m.dense_weight(), self.w, rtol=1e-11, atol=1e-11)

    def test_tt_svd_bound_and_original_cut_spectra(self):
        mpo, d = MPO.from_weight(self.w, self.layout, [2, 3])
        actual = (self.w - mpo.dense_weight()).square().sum().item()
        self.assertLessEqual(actual, d['tt_svd_frobenius_bound_squared'] + 1e-10)
        spectra = unfolding_spectra(self.w, self.layout)
        self.assertEqual([len(s) for s in spectra], [6, 8])
        for s in spectra:
            torch.testing.assert_close(s.square().sum(), self.w.square().sum())

    def test_direct_forward_no_dense_materialization_and_gradients(self):
        mpo, _ = MPO.from_weight(self.w, self.layout, [2, 3])
        x = self.x[:3].clone().requires_grad_()
        reference = x @ mpo.dense_weight().T
        mpo.dense_weight = lambda: (_ for _ in ()).throw(AssertionError("Dense forward forbidden"))
        torch.testing.assert_close(mpo(x), reference)
        self.assertTrue(torch.autograd.gradcheck(mpo, (x,), eps=1e-6, atol=1e-4))
        mpo(x).square().mean().backward()
        self.assertTrue(all(c.grad is not None and torch.isfinite(c.grad).all() for c in mpo.cores))

    def test_scale_compensation_for_both_representations(self):
        scale = activation_scale(self.x, 'rms')
        for op in (MPO.from_weight(self.w, self.layout, [2, 3], scale)[0],
                   MatrixFactors.from_weight(self.w, 3, scale)):
            torch.testing.assert_close(op(self.x), self.x @ op.dense_weight().T)
        exact, _ = MPO.from_weight(self.w, self.layout, [100, 100], scale)
        torch.testing.assert_close(exact(self.x), self.x @ self.w.T)

    def test_nonzero_mean_second_moment_identity(self):
        z = self.x + 3
        error = self.w / 7
        direct = functional_error(error, z)
        torch.testing.assert_close(direct, moment_error(error, second_moment(z)))
        centered = second_moment(z - z.mean(0))
        self.assertGreater(abs(float(direct - moment_error(error, centered))), 1)

    def test_rms_proxy_is_not_centered_std(self):
        z = torch.tensor([[10., 1.], [10., -1.]], dtype=torch.float64)
        self.assertGreater(float(activation_scale(z)[0]), float(activation_scale(z)[1]))
        self.assertLess(float(activation_scale(z, 'std')[0]), float(activation_scale(z, 'std')[1]))

    def test_weight_error_can_rank_functional_error_backwards(self):
        z = torch.tensor([[100., 0.]], dtype=torch.float64)
        e1 = torch.tensor([[.1, 0.]], dtype=torch.float64)
        e2 = torch.tensor([[0., 1.]], dtype=torch.float64)
        self.assertLess(float(e1.square().sum()), float(e2.square().sum()))
        self.assertGreater(float(functional_error(e1, z)), float(functional_error(e2, z)))

    def test_coala_empirical_optimum_even_with_singular_inputs(self):
        x = self.x.clone()
        x[:, 10:] = x[:, :14]
        op = MatrixFactors.coala_projection(self.w, x, 3)
        target = self.w @ x.T / len(x) ** .5
        optimal = torch.linalg.svdvals(target)[3:].square().sum()
        actual = functional_error(self.w - op.dense_weight(), x)
        torch.testing.assert_close(actual, optimal)

    def test_coala_regularized_optimum(self):
        ridge = .2
        op = MatrixFactors.coala_projection(self.w, self.x, 3, ridge)
        target = torch.cat([self.w @ self.x.T / len(self.x) ** .5, ridge ** .5 * self.w], 1)
        tail = torch.linalg.svdvals(target)[3:].square().sum()
        e = self.w - op.dense_weight()
        torch.testing.assert_close(functional_error(e, self.x) + ridge * e.square().sum(), tail)

    def test_same_domain_objective_and_worst_gradient(self):
        a = torch.tensor(3., requires_grad=True)
        b = torch.tensor(8., requires_grad=True)
        s, d = {'a': a, 'b': b}, {'a': 2., 'b': 4.}
        torch.testing.assert_close(domain_repair_objective(s, d, 'mixture'), torch.tensor(.75))
        loss = domain_repair_objective(s, d, 'worst_domain')
        loss.backward()
        torch.testing.assert_close(a.grad, torch.tensor(0.))
        torch.testing.assert_close(b.grad, torch.tensor(.25))

    def test_exact_payload_roundtrip_and_scale_cost(self):
        plain, _ = MPO.from_weight(self.w, self.layout, [2, 3])
        scaled, _ = MPO.from_weight(self.w, self.layout, [2, 3], activation_scale(self.x))
        payload = deployment_payload(scaled)
        metadata, state = decode_payload(payload)
        self.assertEqual(metadata['kind'], 'mpo')
        for name, value in scaled.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertGreater(len(payload) - len(deployment_payload(plain)), 24 * 8)
        self.assertEqual(payload, deployment_payload(scaled))
        with self.assertRaises(ValueError): decode_payload(payload + b'padding')

    def test_bfloat16_payload_and_bias(self):
        op = MatrixFactors.from_weight(self.w, 3, bias=torch.ones(12, dtype=torch.float64)).to(torch.bfloat16)
        _, state = decode_payload(deployment_payload(op))
        torch.testing.assert_close(state['left'], op.left, rtol=0, atol=0)
        self.assertIn('bias', state)

    def test_whole_model_cap_includes_fixed_bytes(self):
        op = MatrixFactors.from_weight(self.w, 3)
        p = deployment_payload(op)
        budget = WholeModelBudget(1000 + len(p), 1000)
        self.assertEqual(budget.check([p])['unused_bytes'], 0)
        with self.assertRaises(ValueError): WholeModelBudget(len(p), 1).check([p])

    def test_both_representations_can_receive_same_synthetic_repair(self):
        # This tests backward/optimizer plumbing only, not scientific superiority.
        for op in (MPO.from_weight(self.w, self.layout, [2, 3])[0], MatrixFactors.from_weight(self.w, 3)):
            opt = torch.optim.SGD(op.parameters(), lr=1e-4)
            before = (op(self.x) - self.x @ self.w.T).square().mean()
            opt.zero_grad(); before.backward(); opt.step()
            after = (op(self.x) - self.x @ self.w.T).square().mean()
            self.assertLess(float(after.detach()), float(before.detach()))

    def test_invalid_configuration_fails(self):
        with self.assertRaises(ValueError): Layout((2,), (3, 4))
        with self.assertRaises(ValueError): MPO.from_weight(self.w, self.layout, [0, 3])
        with self.assertRaises(ValueError): MatrixFactors.from_weight(self.w, 3, torch.zeros(24))
        with self.assertRaises(ValueError): domain_repair_objective({'a': torch.tensor(1.)}, {'b': 2.}, 'mixture')

    def test_pair_contract_rejects_unmatched_repair_budget_or_goal(self):
        # Mock identifiers and budgets, never a real experiment specification.
        spec = ComparisonContract('synthetic', 'synthetic', ('down',), 10000,
                                  'synthetic-mixture', 'synthetic-targets', 10, 20,
                                  2, 'synthetic-sgd', 1000, 1, 1, 'last', 17)
        self.assertTrue(require_identical_contract(spec, replace(spec)))
        for field, value in [('optimizer_steps', 3), ('objective_fingerprint', 'worst'),
                             ('repair_tokens', 21), ('whole_model_cap_bytes', 10001),
                             ('repair_flops_cap', 1001), ('search_trials', 2)]:
            with self.assertRaises(ValueError): require_identical_contract(spec, replace(spec, **{field: value}))
        for field in ['model_revision', 'repair_flops_cap', 'layer_names']:
            with self.assertRaises(ValueError): replace(spec, **{field: None}).validate()


if __name__ == '__main__':
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Correctness)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'duration_seconds': time.monotonic() - started,
              'torch': torch.__version__, 'python': platform.python_version(),
              'device': 'cpu', 'threads': torch.get_num_threads(),
              'purpose': 'tiny synthetic algorithm correctness only; no model evidence or GPU lease',
              'test_file_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parents[1].glob('src/**/*.py'))
                                if p.name in {'__init__.py', 'operators.py', 'objectives.py', 'accounting.py', 'comparison.py'}}}
    Path('results/cpu_correctness.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
