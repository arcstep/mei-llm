"""Write report.md for a run dir via mei_eval (path-fallback)."""

from __future__ import annotations

import sys
from pathlib import Path


def write_report(out_dir: Path) -> Path | None:
    """Best-effort: never fail the runner if report generation breaks."""
    try:
        try:
            from mei_eval.report import write_run_report
        except ImportError:
            mei_projects = Path(__file__).resolve().parents[2]
            src = mei_projects / "tools" / "mei-eval" / "python" / "src"
            if src.is_dir():
                sys.path.insert(0, str(src))
            from mei_eval.report import write_run_report

        path = write_run_report(out_dir)
        print(f"wrote {path}")
        return path
    except Exception as exc:  # noqa: BLE001 — report is derived, non-fatal
        print(f"warn: report.md skipped: {exc}", file=sys.stderr)
        return None
