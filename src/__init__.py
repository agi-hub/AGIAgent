#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Agent - AI-powered intelligent code generation and autonomous task execution system

Copyright (c) 2025 AGI Agent Research Group.
Licensed under the Apache License, Version 2.0
"""

from .main import AGIAgentClient, create_client, AGIAgentMain

__version__ = "0.1.0"
__author__ = "AGI Agent Team"
__email__ = "contact@agia.ai"
__description__ = "AI Code Auto-Generator - Intelligent code generation and task execution system based on Claude Sonnet"

# Public API
__all__ = [
    'AGIAgentClient',
    'create_client', 
    'AGIAgentMain',
    '__version__',
    '__author__',
    '__email__',
    '__description__'
] 