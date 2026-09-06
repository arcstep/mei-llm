import json
from pathlib import Path
import tempfile
import unittest

import training.cpt.audit_cpt_synthetic_diversity_51m as audit


class SyntheticDiversityAuditTests(unittest.TestCase):
    def test_unique_ids_do_not_hide_source_loss_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metrics = root / "metrics.jsonl"
            rows = []
            counters = {role: 0 for role in audit.ROLES}
            for index in range(80):
                role = "structure" if index % 2 == 0 else "colloquial"
                counters[role] += 2_048
                rows.append(
                    {
                        "step": index + 1,
                        "loss": 0.01,
                        "source_tokens_drawn": dict(counters),
                    }
                )
            metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            inputs = {}
            for role in audit.SYNTHETIC_ROLES:
                path = root / f"{role}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({"text": f"固定模板，仅编号 {index}"}) + "\n"
                        for index in range(25)
                    ),
                    encoding="utf-8",
                )
                inputs[role] = path
            receipt = audit.build_receipt(metrics, inputs)
            self.assertEqual(receipt["terminal_status"], "degraded")
            self.assertTrue(receipt["corpus_diversity_degraded"])
            for role in audit.SYNTHETIC_ROLES:
                self.assertEqual(receipt["roles"][role]["status"], "degraded")
                self.assertEqual(
                    receipt["roles"][role]["raw_identity"]["exact_unique_ratio"],
                    1.0,
                )
                self.assertIn(
                    "source_loss_template_collapse",
                    receipt["roles"][role]["reasons"],
                )

    def test_diverse_loss_roles_pass_and_inputs_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metrics = root / "metrics.jsonl"
            rows = []
            counters = {role: 0 for role in audit.ROLES}
            for index in range(82):
                role = "structure" if index % 2 == 0 else "colloquial"
                counters[role] += 2_048
                rows.append(
                    {
                        "step": index + 1,
                        "loss": 2.0 + (index % 7) / 10,
                        "source_tokens_drawn": dict(counters),
                    }
                )
            metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            inputs = {}
            for role in audit.SYNTHETIC_ROLES:
                path = root / f"{role}.jsonl"
                path.write_text(
                    "".join(json.dumps({"text": f"样本 {index}"}) + "\n" for index in range(4)),
                    encoding="utf-8",
                )
                inputs[role] = path
            receipt = audit.build_receipt(metrics, inputs)
            self.assertEqual(receipt["terminal_status"], "passed")
            self.assertFalse(receipt["corpus_diversity_degraded"])
            self.assertEqual(
                receipt["training_evidence"]["sha256"], audit.sha256_file(metrics)
            )

    def test_audit_receipt_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            audit.atomic_write_json(path, {"status": "passed"})
            audit.atomic_write_json(path, {"status": "passed"})
            with self.assertRaises(FileExistsError):
                audit.atomic_write_json(path, {"status": "degraded"})


if __name__ == "__main__":
    unittest.main()
