#!/usr/bin/env python3
"""Thin wrapper: build MW SFT only. Does not touch home-sft or training entrypoints."""

from build_needle_mw_assets_v0 import main
import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--only", "sft", *sys.argv[1:]]
    raise SystemExit(main())
