#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Bot - AI-powered intelligent code generation and autonomous task execution system

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

from .main import AGIBotClient, create_client, AGIBotMain

__version__ = "0.1.0"
__author__ = "AGI Bot Team"
__email__ = "contact@agibot.ai"
__description__ = "AI Code Auto-Generator - Intelligent code generation and task execution system based on Claude Sonnet"

# Public API
__all__ = [
    'AGIBotClient',
    'create_client', 
    'AGIBotMain',
    '__version__',
    '__author__',
    '__email__',
    '__description__'
] 