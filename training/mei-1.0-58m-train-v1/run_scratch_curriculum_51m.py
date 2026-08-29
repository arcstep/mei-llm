#!/usr/bin/env python3
"""Run the mei-1.0-51m scratch curriculum without touching legacy 58M runs."""

from __future__ import annotations

import os

os.environ["MEI_ARCHITECTURE_ID"] = "mei-1.0-51m-arch-v1"

from run_scratch_curriculum import main


if __name__ == "__main__":
    raise SystemExit(main())
