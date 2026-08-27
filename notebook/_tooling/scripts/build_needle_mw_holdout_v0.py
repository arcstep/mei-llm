#!/usr/bin/env python3
"""Thin wrapper: build MW holdout only. Never writes v1/v2 banks."""

from build_needle_mw_assets_v0 import main
import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--only", "eval", *sys.argv[1:]]
    raise SystemExit(main())
