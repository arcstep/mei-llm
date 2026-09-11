import copy
import unittest

from diagnostics.bounded_cpt_recovery import validate_binding, window_quotas


class BoundedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = {"schema": "mei-cpt-bounded-recovery-v1", "scope": "independent_diagnostic_no_promotion",
                       "batch_size": 1, "grad_accum": 8, "seq_len": 2048, "lr": 3e-5,
                       "compile_train": False, "steps": 611, "parent_tokens_seen": 1500001792,
                       "configuration_change": {"previous_effective_batch_tokens": 2048,
                                                "new_effective_batch_tokens": 16384,
                                                "reason": "explicit_recovery_of_unpaired_batch_regression"},
                       "corpus_reuse": {"decision": "reuse", "scope": "bounded_diagnostic_only",
                                        "formal_reuse_eligible": False,
                                        "known_quality_status": "not_audited_diversity_degraded",
                                        "max_delta_tokens": 611 * 16384},
                       "automatic_parent_promotion": False, "release_eligible": False,
                       "extend_automatically": False, "new_corpus_adopted": False}
        self.metadata = {"tokens_seen": 1500001792, "batch_size": 1, "grad_accum": 1, "seq_len": 2048}
        self.gate = {"status": "blocked", "detail": {"benefit_vs_parent_ok": False, "identity_errors": [],
                     "readiness_passed": True, "quality_evidence_present": True, "exposure_ok": True,
                     "quota_ok": True, "checkpoint_ok": True, "schedule_ok": True}}

    def test_explicit_diagnostic_change_is_valid(self):
        validate_binding(self.config, self.metadata, self.gate)

    def test_integrity_failure_is_never_a_quality_exception(self):
        for key in ("exposure_ok", "quota_ok", "checkpoint_ok", "schedule_ok", "quality_evidence_present"):
            gate = copy.deepcopy(self.gate)
            gate["detail"][key] = False
            with self.assertRaisesRegex(ValueError, "integrity or evidence"):
                validate_binding(self.config, self.metadata, gate)

    def test_no_silent_reuse_or_extension(self):
        for key, value in (("corpus_reuse", {}), ("configuration_change", {}), ("steps", 1221),
                           ("automatic_parent_promotion", True), ("new_corpus_adopted", True)):
            with self.assertRaises(ValueError):
                validate_binding({**self.config, key: value}, self.metadata, self.gate)

    def test_quota_allocation_has_exact_full_windows(self):
        weights = {"fineweb2_hq": 125 / 300, "wiki_zh": 80 / 300, "dialogue": 40 / 300,
                   "structured": 25 / 300, "code": 20 / 300, "wiki_en": 10 / 300}
        quotas = window_quotas(weights, 611)
        self.assertEqual(sum(quotas.values()), 10010624)
        for role, amount in quotas.items():
            self.assertEqual(amount % 2048, 0)
            self.assertLess(abs(amount - 10010624 * weights[role]), 2048)
