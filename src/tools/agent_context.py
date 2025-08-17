#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent context: a simple place to set/get current agent_id and log directory.
Not tied to the print system.
"""

from typing import Optional
from contextvars import ContextVar

_current_agent_id: ContextVar[Optional[str]] = ContextVar("current_agent_id", default=None)
_current_log_dir: ContextVar[Optional[str]] = ContextVar("current_log_dir", default=None)


def set_current_agent_id(agent_id: Optional[str]) -> None:
	"""Set current agent id for this execution context."""
	_current_agent_id.set(agent_id)


def get_current_agent_id() -> Optional[str]:
	"""Get current agent id for this execution context."""
	return _current_agent_id.get()


def set_current_log_dir(log_dir: Optional[str]) -> None:
	"""Set current log directory for this execution context."""
	_current_log_dir.set(log_dir)


def get_current_log_dir() -> Optional[str]:
	"""Get current log directory for this execution context."""
	return _current_log_dir.get()

