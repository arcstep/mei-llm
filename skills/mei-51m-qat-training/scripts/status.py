#!/usr/bin/env python3
import sys

from _dispatch import run

raise SystemExit(run("status", sys.argv[1:]))
