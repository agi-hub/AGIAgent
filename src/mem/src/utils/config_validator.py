#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Configuration Validator
"""

import os
from typing import Dict, Any, List
from .exceptions import ConfigError


class ConfigValidator:
    """Configuration validator"""

    REQUIRED_FIELDS = [
        "api_key",
        "api_base",
        "mem_model",
        "embedding_model"
    ]

    OPTIONAL_FIELDS = [
        "max_tokens",
        "similarity_threshold",
        "default_top_k",
        "streaming"
    ]

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration

        Args:
            config: Configuration dictionary

        Returns:
            List of error messages
        """
        errors = []

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in config or not config[field]:
                errors.append(f"Missing required configuration item: {field}")

        # Validate API key format
        if "api_key" in config and config["api_key"]:
            if not config["api_key"].startswith("sk-"):
                errors.append("API key format is incorrect, should start with 'sk-'")

        # Validate value ranges
        if "similarity_threshold" in config:
            try:
                threshold = float(config["similarity_threshold"])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("similarity_threshold should be between 0.0 and 1.0")
            except ValueError:
                errors.append("similarity_threshold should be a number")

        if "max_tokens" in config:
            try:
                tokens = int(config["max_tokens"])
                if tokens <= 0:
                    errors.append("max_tokens should be a positive integer")
            except ValueError:
                errors.append("max_tokens should be an integer")

        return errors

    @classmethod
    def validate_storage_path(cls, path: str) -> List[str]:
        """
        Validate storage path

        Args:
            path: Storage path

        Returns:
            List of error messages
        """
        errors = []

        if not path:
            errors.append("Storage path cannot be empty")
            return errors

        # Check if path is writable
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, IOError) as e:
            errors.append(f"Storage path is not writable: {e}")

        return errors
