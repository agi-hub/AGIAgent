#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner: one task -> one AGIAgent session -> one output_* directory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_NAME = "tale_alfworld"
DATASET_NAME = "ALFWorld"
TOOL_PREFIX = APP_NAME

APP_DIR = Path(__file__).resolve().parent
AGI_ROOT = APP_DIR.parent.parent.resolve()
WORKSPACE_ROOT = AGI_ROOT.parent.resolve()
if str(AGI_ROOT) not in sys.path:
    sys.path.insert(0, str(AGI_ROOT))

from src.tools.tale_alfworld_tools import TaleAlfworldTools as ToolClass


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "task"


def _slice(tasks: List[str], offset: int, count: int) -> List[str]:
    s = max(0, int(offset))
    if s >= len(tasks):
        return []
    if int(count) <= 0:
        return tasks[s:]
    return tasks[s:s + int(count)]


def _requirement(env_name: str) -> str:
    return (
        f"Run only one {DATASET_NAME} task: {env_name}. "
        f"Only {TOOL_PREFIX}_action is allowed. "
        f"The first call must include env_name=\"{env_name}\". "
        f"Then keep calling {TOOL_PREFIX}_action(action=...) each round until done=true. "
        "At the end, output best_score/current_score/full_score/score_rate."
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    v = _safe_float(value)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _avg(values: List[Any]) -> Optional[float]:
    nums: List[float] = []
    for value in values:
        v = _safe_float(value)
        if v is not None:
            nums.append(v)
    if not nums:
        return None
    return sum(nums) / len(nums)


def _load_score_summary(output_dir: Path) -> Dict[str, Any]:
    summary_path = output_dir / "logs" / "score_summary.json"
    if not summary_path.exists():
        return {
            "has_metrics": False,
            "reason": f"score summary file not found: {summary_path}",
        }

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "has_metrics": False,
            "error": f"failed to parse {summary_path}: {e}",
        }

    if not isinstance(data, dict):
        return {
            "has_metrics": False,
            "error": f"invalid score summary format in {summary_path}",
        }

    if "has_metrics" not in data:
        data["has_metrics"] = False

    data["score_summary_path"] = str(summary_path.resolve())
    return data


def _compute_dataset_score_averages(manifest: Dict[str, Any]) -> None:
    runs = manifest.get("runs", [])
    scored_summaries: List[Dict[str, Any]] = []

    for run in runs:
        summary = run.get("score_summary")
        if isinstance(summary, dict) and summary.get("has_metrics"):
            scored_summaries.append(summary)

    manifest["scored_runs"] = len(scored_summaries)
    manifest["dataset_average_score_rate"] = _avg([s.get("score_rate") for s in scored_summaries])


def main() -> int:
    p = argparse.ArgumentParser(description=f"Run all {DATASET_NAME} tasks with isolated sessions")
    p.add_argument("--task-count", type=int, default=0)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--loops", type=int, default=100)
    p.add_argument("--output-prefix", type=str, default="output")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tool = ToolClass(workspace_root=str(WORKSPACE_ROOT))
    list_method = getattr(tool, f"{TOOL_PREFIX}_list_tasks")
    ret: Dict[str, Any] = list_method()
    if ret.get("status") != "success":
        print(json.dumps(ret, ensure_ascii=False, indent=2))
        return 1

    tasks = ret.get("tasks", [])
    selected = _slice(tasks, args.task_offset, args.task_count)
    if not selected:
        print(json.dumps({"status": "error", "message": "No tasks selected."}, ensure_ascii=False, indent=2))
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = AGI_ROOT / f"output_{APP_NAME}_batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "status": "running",
        "app": APP_NAME,
        "dataset": DATASET_NAME,
        "timestamp": ts,
        "task_total": len(tasks),
        "task_selected": len(selected),
        "runs": [],
    }

    for i, env_name in enumerate(selected, 1):
        out_dir = f"{args.output_prefix}_{APP_NAME}_{i:02d}_{_safe_name(env_name)}_{ts}"
        cmd = [
            args.python_exe,
            "agia.py",
            "--app", APP_NAME,
            "--dir", out_dir,
            "--loops", str(int(args.loops)),
            _requirement(env_name=env_name),
        ]
        run_output_dir = (AGI_ROOT / out_dir).resolve()
        item = {
            "index": i,
            "env_name": env_name,
            "output_dir": str(run_output_dir),
            "command": cmd,
            "status": "running",
        }

        if args.dry_run:
            item["status"] = "dry_run"
            item["returncode"] = None
            manifest["runs"].append(item)
            print("DRY RUN:", " ".join(cmd))
            continue

        print(f"[{i}/{len(selected)}] Running {env_name} -> {out_dir}")
        cp = subprocess.run(cmd, cwd=str(AGI_ROOT), check=False)
        item["returncode"] = cp.returncode
        item["status"] = "success" if cp.returncode == 0 else "error"

        score_summary = _load_score_summary(run_output_dir)
        item["score_summary"] = score_summary
        item["score_summary_path"] = str((run_output_dir / "logs" / "score_summary.json").resolve())

        if score_summary.get("has_metrics"):
            item["best_score"] = score_summary.get("best_score")
            item["current_score"] = score_summary.get("current_score")
            item["full_score"] = score_summary.get("full_score")
            item["score_rate"] = score_summary.get("score_rate")
            print(
                "   score summary: "
                f"best={_fmt(item.get('best_score'))}, "
                f"current={_fmt(item.get('current_score'))}, "
                f"full={_fmt(item.get('full_score'))}, "
                f"score_rate={_fmt(item.get('score_rate'))}"
            )
        else:
            reason = score_summary.get("reason") or score_summary.get("error") or "no score metrics"
            print(f"   score summary unavailable: {reason}")

        manifest["runs"].append(item)
        _compute_dataset_score_averages(manifest)
        (batch_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    errs = sum(1 for r in manifest["runs"] if r["status"] == "error")
    manifest["status"] = "success" if errs == 0 else "partial_success"
    manifest["error_count"] = errs
    _compute_dataset_score_averages(manifest)

    avg_score_rate = manifest.get("dataset_average_score_rate")
    if avg_score_rate is not None:
        print(
            f"Dataset average score rate ({DATASET_NAME}): "
            f"{_fmt(avg_score_rate)} based on {manifest.get('scored_runs', 0)} scored runs"
        )
    else:
        print(f"Dataset average score rate ({DATASET_NAME}): N/A (no scored runs)")

    (batch_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if errs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
