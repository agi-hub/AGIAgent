#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities package for the AGI Agent project.

This package contains various utility modules for parsing, formatting,
and processing data used throughout the project.
"""

from .parse import (
    fix_json_escapes,
    smart_escape_quotes_in_json_values,
    rebuild_json_structure,
    fix_long_json_with_code,
    parse_python_params_manually,
    convert_parameter_value,
    # Backward compatibility aliases
    _fix_json_escapes,
    _smart_escape_quotes_in_json_values,
    _rebuild_json_structure,
    _fix_long_json_with_code,
    _parse_python_params_manually,
    _convert_parameter_value,
)

__all__ = [
    'fix_json_escapes',
    'smart_escape_quotes_in_json_values',
    'rebuild_json_structure',
    'fix_long_json_with_code',
    'parse_python_params_manually',
    'convert_parameter_value',
    # Backward compatibility aliases
    '_fix_json_escapes',
    '_smart_escape_quotes_in_json_values',
    '_rebuild_json_structure',
    '_fix_long_json_with_code',
    '_parse_python_params_manually',
    '_convert_parameter_value',
] 