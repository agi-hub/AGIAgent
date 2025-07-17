#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Multi-round Task Executor Package

Execute tasks in todo.csv, perform multiple rounds of calls for each task, save history and summaries.
"""

from .executor import MultiRoundTaskExecutor
from .task_loader import TaskLoader
from .summary_generator import SummaryGenerator
from .report_generator import ReportGenerator
from .debug_recorder import DebugRecorder
from .task_checker import TaskChecker
from .config import *

__version__ = "0.1.0"
__all__ = [
    "MultiRoundTaskExecutor",
    "TaskLoader", 
    "SummaryGenerator",
    "ReportGenerator",
    "DebugRecorder",
    "TaskChecker"
]