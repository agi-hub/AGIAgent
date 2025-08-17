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
AGI Bot project installation script
"""

# Application name macro definition
APP_NAME = "AGIBot"

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agibot",
    version="1.0.0",
    author=f"{APP_NAME} Team",
    description="AI Code Auto-Generator - Intelligent code generation and task execution system based on Claude Sonnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agibot/agibot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core API libraries
        "requests>=2.20.0",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        
        # Basic scientific computing libraries
        "numpy>=1.18.0",
        
        # Machine learning and text processing
        "scikit-learn>=0.22.0",
        "jieba>=0.35.0",  # Chinese text segmentation library
        
        # Web automation
        "playwright>=1.20.0",
        
        # Data processing and file operations
        "tqdm>=4.30.0",
        "pandoc>=2.0.0",
        
        #MCP
        "cli-mcp>=0.1.0",
        
        # GUI dependencies
        "flask>=2.0.0",
        "flask-socketio>=5.0.0",
        "eventlet>=0.30.0",
    ],
    extras_require={
        "dev": [
            # Testing framework
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-asyncio>=0.21.0",
            
            # Code quality checks
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
            
            # Development tools
            "pre-commit>=3.0.0",
            "jupyterlab>=4.0.0",
            
            # Documentation generation
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "minimal": [
            "requests>=2.28.0",  # Basic functionality only
        ],
        "advanced": [
            # Semantic search related
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            
            # Colored logging
            "colorlog>=6.6.0",
            
            # Advanced data processing
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agibot=agibot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.sh"],
    },
) 
