#!/usr/bin/env python3
import sys

from _dispatch import run

raise SystemExit(run("run", sys.argv[1:]))
