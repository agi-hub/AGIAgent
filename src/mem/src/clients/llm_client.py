"""
LLM Client Module
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
import logging

from ..utils.exceptions import LLMClientError, ConfigError
from ..utils.logger import get_logger
from ..utils.config import ConfigLoader

logger = get_logger(__name__)


class LLMClient:
    """Large Language Model client class for calling various LLM APIs"""

    def __init__(self, config_file: str = "config.txt"):
        """
        Initialize LLM client

        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        # Use ConfigLoader for multi-file configuration support
        self.config_loader = ConfigLoader(config_file)
        
        # Get memory-specific API configuration with fallback to general config
        self.api_key = self._get_mem_api_key()
        self.api_base = self._get_mem_api_base()
        self.model = self.config_loader.get('mem_model')
        self.max_tokens = self.config_loader.get_int('max_tokens', 8192)
        self.streaming = self.config_loader.get_bool('streaming', False)

        if not all([self.api_key, self.api_base, self.model]):
            missing_configs = []
            if not self.api_key:
                missing_configs.append("mem_model_api_key or api_key")
            if not self.api_base:
                missing_configs.append("mem_model_api_base or api_base") 
            if not self.model:
                missing_configs.append("mem_model")
            raise ConfigError(f"Missing required API configuration information: {', '.join(missing_configs)}")

    def _get_mem_api_key(self) -> str:
        """
        Get memory model API key with fallback
        
        Returns:
            API key string
        """
        # Priority: mem_model_api_key > api_key
        api_key = self.config_loader.get('mem_model_api_key')
        if api_key:
            logger.info("Using mem_model_api_key for LLM client")
            return api_key
            
        api_key = self.config_loader.get('api_key')
        if api_key:
            logger.info("Using fallback api_key for LLM client")
            return api_key
            
        return ""

    def _get_mem_api_base(self) -> str:
        """
        Get memory model API base URL with fallback
        
        Returns:
            API base URL string
        """
        # Priority: mem_model_api_base > api_base
        api_base = self.config_loader.get('mem_model_api_base')
        if api_base:
            logger.info("Using mem_model_api_base for LLM client")
            return api_base
            
        api_base = self.config_loader.get('api_base')
        if api_base:
            logger.info("Using fallback api_base for LLM client")
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
        Determine the correct endpoint based on API base URL

        Returns:
            API endpoint URL
        """
        # Handle different API providers
        if 'anthropic' in self.api_base:
            # Anthropic API uses different endpoint
            return f"{self.api_base}/messages"
        elif 'openai' in self.api_base or 'siliconflow' in self.api_base or 'deepseek' in self.api_base:
            # OpenAI compatible API
            return f"{self.api_base}/chat/completions"
        else:
            # Default to chat/completions endpoint
            return f"{self.api_base}/chat/completions"

    def _prepare_request_data(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int, stream: bool) -> Dict[str, Any]:
        """
        Prepare request data based on API type

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            stream: Whether to use streaming output

        Returns:
            Request data dictionary
        """
        if 'anthropic' in self.api_base:
            # Anthropic API format
            return {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        else:
            # OpenAI compatible format
            return {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

    def _prepare_headers(self) -> Dict[str, str]:
        """
        Prepare request headers based on API type

        Returns:
            Request headers dictionary
        """
        if 'anthropic' in self.api_base:
            # Anthropic API uses x-api-key
            return {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        else:
            # OpenAI compatible format uses Bearer token
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None
    ) -> Union[str, Dict]:
        """
        Call large language model to generate response

        Args:
            prompt: User input prompt
            system_prompt: System prompt
            temperature: Temperature parameter to control output randomness
            max_tokens: Maximum token count, None uses default value from config file
            stream: Whether to use streaming output, None uses default value from config file

        Returns:
            Model response content
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if stream is None:
            stream = self.streaming

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request data
        data = self._prepare_request_data(
            messages, temperature, max_tokens, stream)
        headers = self._prepare_headers()
        endpoint = self._get_api_endpoint()

        try:
            logger.info(f"Calling model: {self.model} at endpoint: {endpoint}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return self._handle_normal_response(response)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise LLMClientError(f"Failed to call LLM API: {e}")
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            raise LLMClientError(f"Failed to process model response: {e}")

    def generate_response_with_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Call large language model to generate response with tool calls

        Args:
            prompt: User input prompt
            system_prompt: System prompt
            tools: Tool definition list
            temperature: Temperature parameter to control output randomness
            max_tokens: Maximum token count, None uses default value from config file
            tool_choice: Tool selection strategy, "auto", "none", or specific tool name

        Returns:
            Model response containing tool calls
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request data
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        # Add tool-related parameters
        if tools:
            data["tools"] = tools
            data["tool_choice"] = tool_choice

        headers = self._prepare_headers()
        endpoint = self._get_api_endpoint()

        try:
            logger.info(f"Calling model: {self.model} at endpoint: {endpoint} (with tools)")
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise LLMClientError(f"Failed to call LLM API: {e}")
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            raise LLMClientError(f"Failed to process model response: {e}")

    def _handle_normal_response(self, response: requests.Response) -> str:
        """
        Handle normal response

        Args:
            response: HTTP response object

        Returns:
            Model response text
        """
        try:
            result = response.json()

            if 'anthropic' in self.api_base:
                # Anthropic API response format
                return result['content'][0]['text']
            else:
                # OpenAI compatible format
                return result['choices'][0]['message']['content']

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse response: {e}")
            raise LLMClientError(f"Failed to parse model response: {e}")

    def _handle_stream_response(self, response: requests.Response) -> Dict:
        """
        Handle streaming response

        Args:
            response: HTTP response object

        Returns:
            Streaming response dictionary
        """
        return {
            "stream": True,
            "response": response,
            "model": self.model
        }

    def get_stream_content(self, stream_response: Dict) -> str:
        """
        Extract complete content from streaming response

        Args:
            stream_response: Streaming response dictionary

        Returns:
            Complete response content
        """
        if not stream_response.get("stream"):
            raise LLMClientError("Not a streaming response")

        response = stream_response["response"]
        content = ""

        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break

                        try:
                            json_data = json.loads(data)
                            if 'anthropic' in self.api_base:
                                # Anthropic API streaming format
                                if 'content' in json_data and json_data['content']:
                                    content += json_data['content'][0]['text']
                            else:
                                # OpenAI compatible format
                                if 'choices' in json_data and json_data['choices']:
                                    delta = json_data['choices'][0].get(
                                        'delta', {})
                                    if 'content' in delta:
                                        content += delta['content']
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Failed to process streaming response: {e}")
            raise LLMClientError(f"Failed to process streaming response: {e}")

        return content


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    config_file: str = "config.txt",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: Optional[bool] = None
) -> str:
    """
    Convenience function: Call large language model

    Args:
        prompt: User input prompt
        system_prompt: System prompt
        config_file: Configuration file path
        temperature: Temperature parameter
        max_tokens: Maximum token count
        stream: Whether to use streaming output

    Returns:
        Model response content
    """
    client = LLMClient(config_file)
    response = client.generate_response(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
    )

    if isinstance(response, dict) and response.get("stream"):
        return client.get_stream_content(response)
    elif isinstance(response, str):
        return response
    else:
        # If response is a dict but not a stream, convert to string
        return str(response)
