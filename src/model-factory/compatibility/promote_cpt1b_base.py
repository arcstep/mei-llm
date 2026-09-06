#!/usr/bin/env python3
"""Retired unsafe 1B promotion entrypoint; kept as a fail-closed shim."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "promote_cpt1b_base.py is retired; use base_candidate_51m.py "
        "register-base-candidate, then propose-freeze, then explicit finalize-current",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
