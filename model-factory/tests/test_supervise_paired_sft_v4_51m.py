from __future__ import annotations

import unittest
from pathlib import Path

import orchestration.supervise_paired_sft_v4_51m as supervisor


class PairedSupervisorTests(unittest.TestCase):
    def test_runtime_command_is_current_source_and_explicit(self) -> None:
        command = supervisor.runtime_command(
            productization_run=Path("product"),
            package=Path("package"),
            master=Path("master.npz"),
            base_release=Path("RELEASE.json"),
            base_weights=Path("base.npz"),
            downstream_run=Path("downstream"),
        )
        self.assertIn("--current-source-reevaluation", command)
        self.assertIn("--productization-run", command)
        self.assertIn("--master", command)


if __name__ == "__main__":
    unittest.main()
