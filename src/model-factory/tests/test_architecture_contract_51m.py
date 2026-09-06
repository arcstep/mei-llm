from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ARCH = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file()) / "src/architecture/mei-1.2-51m"
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from architecture_contract import (
    contract_bundle,
    load_spec,
    resolve_legacy_weight_contract,
    validate_npz_weight_geometry,
)


class ArchitectureContract51MTest(unittest.TestCase):
    def test_exact_weight_geometry(self) -> None:
        bundle = contract_bundle()
        self.assertEqual(bundle["weight_contract"]["trainable_params"], 51_463_797)
        self.assertEqual(len(bundle["weight_contract"]["tensor_order"]), 400)

    def test_runtime_and_training_policy_do_not_change_weight_identity(self) -> None:
        spec = load_spec()
        original = contract_bundle(spec)
        runtime_edit = copy.deepcopy(spec)
        runtime_edit["runtime_profile"]["stable_prefix_profiles"]["compact"] = 896
        runtime = contract_bundle(runtime_edit)
        self.assertEqual(original["weight_contract_sha256"], runtime["weight_contract_sha256"])
        self.assertNotEqual(original["runtime_profile_sha256"], runtime["runtime_profile_sha256"])
        training_edit = copy.deepcopy(spec)
        training_edit["training_aux"]["mtp"]["enabled_by_default"] = True
        training = contract_bundle(training_edit)
        self.assertEqual(original["weight_contract_sha256"], training["weight_contract_sha256"])
        self.assertNotEqual(original["training_aux_sha256"], training["training_aux_sha256"])

    def test_legacy_300m_and_600m_hashes_resolve_only_to_weight_contract(self) -> None:
        expected = contract_bundle()["weight_contract_sha256"]
        for legacy in (
            "84568ca0ecccab4316a39a8ec6d7e31901ea17218c084ac2e1140eafa05c9c41",
            "f02aebda393c7fca181af7b7225581577f689531245dec00ac549de1482c5ea7",
            "c52dad88b3ef52378ae4377a114e3010cdd8f343ab8ebb2a4537356c95a13857",
        ):
            self.assertEqual(resolve_legacy_weight_contract(legacy), expected)
        self.assertIsNone(resolve_legacy_weight_contract("0" * 64))

    def test_frozen_300m_npz_headers_match_the_canonical_geometry(self) -> None:
        base = ARCH.parents[2] / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1"
        weights = base / "mei-1.0-51m-base-scratch300m-v1.npz"
        report = validate_npz_weight_geometry(weights)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["actual_params"], 51_463_797)

    def test_lineage_is_not_self_parented(self) -> None:
        spec = load_spec()
        self.assertNotEqual(spec.get("parent_architecture"), spec["architecture_id"])
        self.assertTrue(spec["compatibility"]["runtime_profile_is_not_weight_identity"])

    def test_release_pins_all_three_normalized_contracts(self) -> None:
        release = json.loads((ARCH / "RELEASE.json").read_text(encoding="utf-8"))
        bundle = contract_bundle()
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        ):
            self.assertEqual(release[key], bundle[key])


if __name__ == "__main__":
    unittest.main()
