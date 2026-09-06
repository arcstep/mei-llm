#!/usr/bin/env python3
"""Retired direct-CURRENT 51M scratch promotion entrypoint.

The frozen 300M base is historical and immutable. Future base candidates must
pass through the explicit register/propose/finalize boundary implemented by
``base_candidate_51m.py``. Keeping this filename as a fail-closed shim avoids
silently reviving an older script that both copied artifacts and changed
``CURRENT.json`` in one operation.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "promote_scratch_base_51m.py is retired; the frozen scratch300m base "
        "already exists. Future base changes must use base_candidate_51m.py "
        "register-base-candidate, then propose-freeze, then an explicitly "
        "authorized finalize-current.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
