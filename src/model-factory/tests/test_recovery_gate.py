import copy
import unittest

from evaluation.base.recovery_gate import ROLES, compare


class RecoveryGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"schema": "mei-cpt-recovery-gate-v1", "role_weights": {role: 1 / 6 for role in ROLES},
                       "aggregate_relative_tolerance": 0.0, "role_relative_tolerance": 0.01,
                       "probe_relative_tolerance": 0.0, "numeric_absolute_tolerance": 1e-5}
        self.parent = {"roles": {role: {"valid_loss": 2.0, "predicted_tokens": 4096, "windows": 2}
                                 for role in ROLES}, "probes": {"mean_nll": 4.0}}
        self.candidate = copy.deepcopy(self.parent)
        self.candidate["roles"]["dialogue"]["valid_loss"] = 1.9

    def test_progress_does_not_authorize_promotion_or_full_round(self):
        report = compare(self.parent, self.candidate, self.policy)
        self.assertEqual(report["status"], "continue_bounded_diagnostic")
        self.assertFalse(report["automatic_parent_promotion"])
        self.assertFalse(report["full_1800m_authorized"])

    def test_probe_regression_is_recorded_but_does_not_gate(self):
        # probe 已降级为「仅记录」信号：valid 改善时 probe 噪声退化不再 block gate，
        # 但 probe_guard 仍记录退化事实供观察。
        self.candidate["probes"]["mean_nll"] = 4.2
        report = compare(self.parent, self.candidate, self.policy)
        self.assertEqual(report["status"], "continue_bounded_diagnostic")
        self.assertFalse(report["checks"]["probe_guard"])

    def test_aggregate_improvement_does_not_hide_role_regression(self):
        self.candidate["roles"]["code"]["valid_loss"] = 2.03
        self.assertFalse(compare(self.parent, self.candidate, self.policy)["checks"]["code"])

    def test_missing_coverage_and_nonfinite_scores_fail_closed(self):
        for key, value in (("predicted_tokens", 2048), ("windows", 1), ("valid_loss", float("nan"))):
            candidate = copy.deepcopy(self.candidate)
            candidate["roles"]["code"][key] = value
            with self.assertRaises(ValueError):
                compare(self.parent, candidate, self.policy)

    def test_identical_checkpoint_is_not_measured_progress(self):
        self.assertFalse(compare(self.parent, self.parent, self.policy)["checks"]["measured_improvement"])

    def test_policy_weights_fail_closed(self):
        self.policy["role_weights"]["code"] = 0
        with self.assertRaises(ValueError):
            compare(self.parent, self.candidate, self.policy)
