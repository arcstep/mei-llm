#!/usr/bin/env python3
import sys

from _dispatch import run

raise SystemExit(run("plan", sys.argv[1:]))
