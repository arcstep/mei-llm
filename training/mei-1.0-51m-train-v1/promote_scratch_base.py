#!/usr/bin/env python3
"""Retired direct-CURRENT scratch promotion entrypoint."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "promote_scratch_base.py is retired; the frozen scratch300m base already exists. "
        "Future base changes must use base_candidate_51m.py and explicit finalize-current.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
