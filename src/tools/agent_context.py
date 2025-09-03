#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent context: a simple place to set/get current agent_id and log directory.
Not tied to the print system.
"""

from typing import Optional, Dict, Any
from contextvars import ContextVar

_current_agent_id: ContextVar[Optional[str]] = ContextVar("current_agent_id", default=None)
_current_log_dir: ContextVar[Optional[str]] = ContextVar("current_log_dir", default=None)

# Global registry for FastMCP wrappers per agent
_agent_fastmcp_wrappers: Dict[str, Any] = {}


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


def set_agent_fastmcp_wrapper(agent_id: str, wrapper: Any) -> None:
	"""Set FastMCP wrapper for a specific agent."""
	_agent_fastmcp_wrappers[agent_id] = wrapper


def get_agent_fastmcp_wrapper(agent_id: str) -> Optional[Any]:
	"""Get FastMCP wrapper for a specific agent."""
	return _agent_fastmcp_wrappers.get(agent_id)


def has_agent_fastmcp_wrapper(agent_id: str) -> bool:
	"""Check if a specific agent has a FastMCP wrapper."""
	return agent_id in _agent_fastmcp_wrappers


def remove_agent_fastmcp_wrapper(agent_id: str) -> None:
	"""Remove FastMCP wrapper for a specific agent."""
	if agent_id in _agent_fastmcp_wrappers:
		del _agent_fastmcp_wrappers[agent_id]


def get_all_agent_fastmcp_wrappers() -> Dict[str, Any]:
	"""Get all agent FastMCP wrappers (for debugging)."""
	return _agent_fastmcp_wrappers.copy()

