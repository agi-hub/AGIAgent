#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-action TALE-Suite tool implementation for dataset: scienceworld
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional


class TaleScienceworldTools:
    DATASET_KEY = "scienceworld"
    TOOL_PREFIX = "tale_scienceworld"

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self._runtime_cache: Dict[str, Any] = {}
        self._active_session: Optional[Dict[str, Any]] = None

    def tale_scienceworld_list_tasks(self, tale_suite_root: Optional[str] = None) -> Dict[str, Any]:
        runtime = self._ensure_runtime(tale_suite_root)
        if runtime["status"] != "success":
            return runtime
        return {
            "status": "success",
            "dataset": self.DATASET_KEY,
            "tasks": runtime["tasks"],
            "count": len(runtime["tasks"]),
            "tale_suite_root": runtime["tale_suite_root"],
        }

    def tale_scienceworld_action(
        self,
        action: str,
        env_name: str = "",
        task_index: int = 0,
        game_seed: Optional[int] = None,
        admissible_commands: bool = False,
        restart: bool = False,
        tale_suite_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        runtime = self._ensure_runtime(tale_suite_root)
        if runtime["status"] != "success":
            return runtime

        action_text = (action or "").strip()
        if not action_text:
            return {"status": "error", "message": "action must be non-empty."}

        if restart and self._active_session is not None:
            self._close_active_session()

        session_started = False
        if self._active_session is None:
            start_res = self._start_session(
                runtime=runtime,
                env_name=env_name,
                task_index=task_index,
                game_seed=game_seed,
                admissible_commands=admissible_commands,
            )
            if start_res.get("status") != "success":
                return start_res
            session_started = True

        session = self._active_session
        assert session is not None

        if env_name.strip() and env_name.strip() != session["env_name"]:
            return {
                "status": "error",
                "message": (
                    f"Active session env is {session['env_name']}, but received env_name={env_name}. "
                    "Use restart=true to start another task."
                ),
                "active_env_name": session["env_name"],
            }

        obs_before = session["last_observation"]

        try:
            step_out = session["env"].step(action_text)
        except Exception as e:
            return {
                "status": "error",
                "message": f"env.step failed: {e}",
                "env_name": session["env_name"],
                "action": action_text,
            }

        info: Dict[str, Any]
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs_after, _reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs_after, _reward, done, info = step_out
            done = bool(done)
        else:
            return {
                "status": "error",
                "message": f"Unexpected env.step return: {type(step_out).__name__}",
            }

        info = info or {}
        session["steps_taken"] += 1

        session["done"] = done
        session["last_action"] = action_text
        session["last_observation"] = obs_after
        session["last_info"] = info

        current_score = self._safe_number(info.get("score"), 0)
        full_score = self._safe_number(info.get("max_score"), 0)
        moves = self._safe_number(info.get("moves"), 0)

        session["current_score"] = current_score
        session["full_score"] = max(
            self._safe_number(session.get("full_score"), 0),
            full_score,
        )
        session["best_score"] = max(
            self._safe_number(session.get("best_score"), 0),
            current_score,
        )

        score_rate = (
            float(session["best_score"]) / float(session["full_score"])
            if float(session["full_score"]) > 0
            else 0.0
        )

        session_closed = False
        close_note = ""
        if done:
            close_note = self._close_active_session()
            session_closed = True

        return {
            "status": "success",
            "dataset": self.DATASET_KEY,
            "env_name": session["env_name"],
            "action": action_text,
            "session_started": session_started,
            "session_closed": session_closed,
            "message": "Action executed." + (f" {close_note}" if close_note else ""),
            "steps_taken": session["steps_taken"],
            "done": done,
            "won": bool(info.get("won", False)),
            "lost": bool(info.get("lost", False)),
            "best_score": self._safe_number(session.get("best_score"), current_score),
            "current_score": current_score,
            "full_score": self._safe_number(session.get("full_score"), full_score),
            "score_rate": score_rate,
            "moves": moves,
            "observation_before": self._as_text(obs_before),
            "observation": self._as_text(obs_after),
            "feedback": self._as_text(info.get("feedback", obs_after)),
            "admissible_commands": info.get("admissible_commands"),
        }

    def _start_session(
        self,
        runtime: Dict[str, Any],
        env_name: str,
        task_index: int,
        game_seed: Optional[int],
        admissible_commands: bool,
    ) -> Dict[str, Any]:
        tasks = runtime["tasks"]
        if not tasks:
            return {"status": "error", "message": f"No tasks for dataset {self.DATASET_KEY}."}

        selected = (env_name or "").strip()
        if selected:
            if selected not in tasks:
                return {
                    "status": "error",
                    "message": f"Unknown env_name: {selected}",
                    "available_tasks": tasks,
                }
        else:
            idx = int(task_index)
            if idx < 0 or idx >= len(tasks):
                return {
                    "status": "error",
                    "message": f"task_index out of range: {idx}",
                    "task_count": len(tasks),
                }
            selected = tasks[idx]

        gym = runtime["gym"]
        try:
            env = gym.make(
                f"tales/{selected}-v0",
                disable_env_checker=True,
                admissible_commands=bool(admissible_commands),
            )
            obs, info = env.reset(seed=game_seed)
        except Exception as e:
            return {"status": "error", "message": f"Failed to start env {selected}: {e}"}

        self._active_session = {
            "created_at": time.time(),
            "dataset": self.DATASET_KEY,
            "env_name": selected,
            "env": env,
            "steps_taken": 0,
            "done": False,
            "last_action": None,
            "last_observation": obs,
            "last_info": info or {},
            "current_score": self._safe_number((info or {}).get("score"), 0),
            "best_score": self._safe_number((info or {}).get("score"), 0),
            "full_score": self._safe_number((info or {}).get("max_score"), 0),
        }
        return {"status": "success"}

    def _close_active_session(self) -> str:
        if self._active_session is None:
            return ""
        msg = "Episode closed."
        try:
            self._active_session["env"].close()
        except Exception as e:
            msg = f"Episode closed with env.close error: {e}"
        self._active_session = None
        return msg

    def _ensure_runtime(self, tale_suite_root: Optional[str]) -> Dict[str, Any]:
        root_res = self._resolve_tale_suite_root(tale_suite_root)
        if root_res["status"] != "success":
            return root_res

        tale_root = root_res["tale_suite_root"]
        cached = self._runtime_cache.get(tale_root)
        if cached is not None:
            return cached

        if tale_root not in sys.path:
            sys.path.insert(0, tale_root)

        try:
            import gymnasium as gym
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to import gymnasium: {e}",
            }

        try:
            import tales
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to import tales: {e}",
                "tale_suite_root": tale_root,
            }

        tasks = sorted(list(tales.envs_per_task.get(self.DATASET_KEY, [])))
        runtime = {
            "status": "success",
            "tale_suite_root": tale_root,
            "gym": gym,
            "tasks": tasks,
        }
        self._runtime_cache[tale_root] = runtime
        return runtime

    def _resolve_tale_suite_root(self, tale_suite_root: Optional[str]) -> Dict[str, Any]:
        candidates: List[str] = []
        if tale_suite_root:
            candidates.append(tale_suite_root)

        env_root = os.environ.get("TALE_SUITE_ROOT")
        if env_root:
            candidates.append(env_root)

        file_based = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "tale-suite")
        )
        candidates.append(file_based)
        candidates.append(os.path.abspath(os.path.join(self.workspace_root, "tale-suite")))
        candidates.append(os.path.abspath(os.path.join(self.workspace_root, "..", "tale-suite")))

        checked: List[str] = []
        for raw in candidates:
            if not raw:
                continue
            p = os.path.abspath(raw)
            if p in checked:
                continue
            checked.append(p)
            if os.path.isfile(os.path.join(p, "benchmark.py")) and os.path.isdir(os.path.join(p, "tales")):
                return {"status": "success", "tale_suite_root": p}

        return {
            "status": "error",
            "message": "Unable to locate tale-suite root.",
            "checked_paths": checked,
        }

    @staticmethod
    def _safe_number(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return ""
