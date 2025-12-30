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

Embedding client module
"""

import os
import json
import requests
import numpy as np
from typing import Dict, List, Optional, Union, Any

from ..utils.exceptions import EmbeddingError, ConfigError
from ..utils.logger import get_logger
from ..utils.config import ConfigLoader

logger = get_logger(__name__)


class EmbeddingClient:
    """Embedding client class, only supports API calls"""

    def __init__(self, config_file: str = "config.txt"):
        """
        Initialize Embedding client

        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        # Use ConfigLoader for multi-file configuration support
        self.config_loader = ConfigLoader(config_file)

        # Get embedding-specific API configuration with fallback to general config
        self.api_key = self._get_embedding_api_key()
        self.api_base = self._get_embedding_api_base()
        self.embedding_model_name = self.config_loader.get('embedding_model', 'bge_m3')

        # Validate API configuration
        if not all([self.api_key, self.api_base]):
            missing_configs = []
            if not self.api_key:
                missing_configs.append("embedding_model_api_key or api_key")
            if not self.api_base:
                missing_configs.append("embedding_model_api_base or api_base")
            raise ConfigError(f"API configuration incomplete, missing: {', '.join(missing_configs)}")

        logger.info(f"Embedding client initialization completed, using API: {self.embedding_model_name}")

    def _get_embedding_api_key(self) -> str:
        """
        Get embedding model API key with fallback
        
        Returns:
            API key string
        """
        # Priority: embedding_model_api_key > api_key
        api_key = self.config_loader.get('embedding_model_api_key')
        if api_key:
            logger.info("Using embedding_model_api_key for Embedding client")
            return api_key
            
        api_key = self.config_loader.get('api_key')
        if api_key:
            logger.info("Using fallback api_key for Embedding client")
            return api_key
            
        return ""

    def _get_embedding_api_base(self) -> str:
        """
        Get embedding model API base URL with fallback
        
        Returns:
            API base URL string
        """
        # Priority: embedding_model_api_base > api_base
        api_base = self.config_loader.get('embedding_model_api_base')
        if api_base:
            logger.info("Using embedding_model_api_base for Embedding client")
            return api_base
            
        api_base = self.config_loader.get('api_base')
        if api_base:
            logger.info("Using fallback api_base for Embedding client")
            return api_base
            
        return ""

    def _load_config(self) -> Dict[str, str]:
        """
        Read configuration file (deprecated, kept for backward compatibility)

        Returns:
            Configuration dictionary
        """
        # This method is kept for backward compatibility
        # but we now use ConfigLoader instead
        # Convert to Dict[str, str] for backward compatibility
        config = {}
        for key, value in self.config_loader.config.items():
            config[str(key)] = str(value)
        return config

    def _get_api_endpoint(self) -> str:
        """
        Get embedding API endpoint

        Returns:
            API endpoint URL
        """
        # Handle different API providers
        if 'openai' in self.api_base or 'siliconflow' in self.api_base:
            # OpenAI compatible API
            return f"{self.api_base}/embeddings"
        else:
            # Default to embeddings endpoint
            return f"{self.api_base}/embeddings"

    def _prepare_api_request_data(self, text: str) -> Dict:
        """
        Prepare API request data

        Args:
            text: Input text

        Returns:
            Request data dictionary
        """
        return {
            "model": self.embedding_model_name,
            "input": text,
            "encoding_format": "float"
        }

    def _prepare_api_headers(self) -> Dict[str, str]:
        """
        Prepare API request headers

        Returns:
            Request headers dictionary
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding vector

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            data = self._prepare_api_request_data(text)
            headers = self._prepare_api_headers()
            endpoint = self._get_api_endpoint()

            logger.info(f"Calling embedding API: {self.embedding_model_name}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()

            # Handle different API response formats
            if 'data' in result and len(result['data']) > 0:
                embedding = result['data'][0]['embedding']
            elif 'embedding' in result:
                embedding = result['embedding']
            else:
                logger.error(f"Unsupported API response format: {result}")
                raise EmbeddingError(f"Unsupported API response format: {result}")

            embedding_array = np.array(embedding, dtype=np.float32)

            # Normalize embedding
            embedding_array = embedding_array / np.linalg.norm(embedding_array)

            logger.info(f"API generated embedding successfully, dimensions: {embedding_array.shape}")
            return embedding_array

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise EmbeddingError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Failed to process API response: {e}")
            raise EmbeddingError(f"Failed to process API response: {e}")

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get embedding system information

        Returns:
            System information dictionary
        """
        info = {
            "use_api": True,
            "embedding_model_name": self.embedding_model_name,
            "api_base": self.api_base,
            "device": None
        }
        return info
