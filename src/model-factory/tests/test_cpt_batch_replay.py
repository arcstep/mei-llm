import copy
import tempfile
import unittest
from pathlib import Path
from diagnostics.cpt_batch_replay import validate_replay
from common.periodic_run_report import PeriodicRunReport


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"tokens_seen": 1200006656, "batch_size": 8, "grad_accum": 1, "seq_len": 2048}
        self.schedule = {"kind": "cpt", "sampler": "quota_plan", "allow_repeat": False,
                         "parent_tokens_seen": 1200006656, "curriculum": [{}],
                         "lr": {"base": 0.0003, "final": 0.00003, "horizon_tokens": 299993344,
                                "kind": "cosine_tokens", "token_offset": 1200006656}}
        self.release = {"continuation_checkpoint_eligible": True, "numeric_integrity": {"status": "passed"},
                        "state_sha256": "state", "state_meta_sha256": "metadata", "weights_sha256": "weights"}
        self.terminal = {"tokens_seen": 1500001792, "stage_tokens_drawn": {
            "code": 20000768, "dialogue": 40001536, "fineweb2_hq": 124991488,
            "structured": 25001984, "wiki_en": 10000384, "wiki_zh": 79998976}}
        self.config = {"schema": "mei-cpt-batch-replay-v1", "scope": "independent_diagnostic_no_promotion",
                       "batch_size": 1, "grad_accum": 8, "seq_len": 2048, "parent_tokens_seen": 1200006656,
                       "lr": 0.0003, "compile_train": False, "target_tokens": 1500001792, "steps": 18311,
                       "parent_state": "state", "parent_metadata": "metadata", "parent_weights": "weights",
                       "input_sha256": {"state": "state", "metadata": "metadata", "weights": "weights"},
                       "corpus_reuse": {"decision": "reuse", "scope": "explicit_same_slice_replay",
                                        "formal_reuse_eligible": False, "max_delta_tokens": 299995136},
                       "automatic_parent_promotion": False, "release_eligible": False,
                       "extend_automatically": False, "new_corpus_adopted": False}

    def test_original_exposure_and_lr_retained(self):
        validate_replay(self.config, self.meta, self.release, self.schedule, self.terminal)

    def test_changed_lr_batch_or_target_rejected(self):
        for key, value in (("grad_accum", 1), ("lr", 3e-5), ("target_tokens", 1500010000),
                           ("new_corpus_adopted", True), ("steps", 146482)):
            with self.assertRaises(ValueError):
                validate_replay({**self.config, key: value}, self.meta, self.release, self.schedule, self.terminal)

    def test_canonical_hash_and_source_reuse_required(self):
        config = copy.deepcopy(self.config)
        config["input_sha256"]["state"] = "wrong"
        with self.assertRaises(ValueError):
            validate_replay(config, self.meta, self.release, self.schedule, self.terminal)
        config = copy.deepcopy(self.config)
        config["corpus_reuse"]["formal_reuse_eligible"] = True
        with self.assertRaises(ValueError):
            validate_replay(config, self.meta, self.release, self.schedule, self.terminal)

    def test_unusable_parent_rejected(self):
        release = {**self.release, "continuation_checkpoint_eligible": False}
        with self.assertRaises(ValueError):
            validate_replay(self.config, self.meta, release, self.schedule, self.terminal)


class PeriodicReportTests(unittest.TestCase):
    def test_report_is_local_and_terminal_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "failure.json").write_text('{"status":"failed","error":"fixture"}')
            reporter = PeriodicRunReport(directory)
            report = reporter.snapshot("scheduled_2h")
            self.assertEqual(reporter.interval, 7200)
            self.assertEqual(report["evidence"]["failure.json"]["status"], "failed")
            self.assertEqual(report["delivery"], "local_report_and_process_stdout_no_chat_wakeup")
