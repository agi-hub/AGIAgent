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

from typing import Any


def test_api_connection(client: Any, model: str, is_claude: bool) -> bool:
    """
    Test API connection for LLM service.
    
    Args:
        client: The LLM client (Anthropic or OpenAI)
        model: Model name to test
        is_claude: Whether this is a Claude model
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    # For safety, import print_current here to avoid circular imports
    try:
        from tools.print_system import print_current
    except ImportError:
        def print_current(msg):
            print(msg)
    
    print_current("🔍 Testing API connection...")
    
    try:
        if is_claude:
            # Claude API
            response = client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Hello, please respond with 'Connection test successful'"}],
                temperature=0.1
            )
            
            if hasattr(response, 'content') and response.content:
                content = response.content[0].text if response.content else ""
                print_current(f"✅ Claude API connection successful! Response: {content[:50]}...")
                return True
            else:
                print_current("❌ Claude API response format exception")
                return False
        else:
            # OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello, please respond with 'Connection test successful'"}],
                max_tokens=50,
                temperature=0.1
            )
            
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                print_current(f"✅ OpenAI API connection successful! Response: {content[:50]}...")
                return True
            else:
                print_current("❌ OpenAI API response format exception")
                return False
                
    except Exception as e:
        print_current(f"❌ API connection test failed: {e}")
        print_current(f"🔧 Please check the following items:")
        print_current(f"   1. API key validity")
        print_current(f"   2. Network connection")
        print_current(f"   3. API service endpoint accessibility")
        print_current(f"   4. API quota")
        return False


def test_anthropic_connection(api_key: str, model: str, api_base: str = None) -> bool:
    """
    Test connection to Anthropic Claude API.
    
    Args:
        api_key: Anthropic API key
        model: Claude model name
        api_base: API base URL (optional)
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Dynamic import of Anthropic
        from anthropic import Anthropic
        
        if api_base:
            client = Anthropic(api_key=api_key, base_url=api_base)
        else:
            client = Anthropic(api_key=api_key)
        
        return test_api_connection(client, model, is_claude=True)
        
    except ImportError:
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        print_current("❌ Anthropic library not installed, please run: pip install anthropic")
        return False
    except Exception as e:
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        print_current(f"❌ Failed to initialize Anthropic client: {e}")
        return False


def test_openai_connection(api_key: str, model: str, api_base: str) -> bool:
    """
    Test connection to OpenAI-compatible API.
    
    Args:
        api_key: API key
        model: Model name
        api_base: API base URL
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, base_url=api_base)
        
        return test_api_connection(client, model, is_claude=False)
        
    except ImportError:
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        print_current("❌ OpenAI library not installed, please run: pip install openai")
        return False
    except Exception as e:
        try:
            from tools.print_system import print_current
        except ImportError:
            def print_current(msg):
                print(msg)
        print_current(f"❌ Failed to initialize OpenAI client: {e}")
        return False


# Wrapper function that auto-detects the model type
def test_llm_connection(api_key: str, model: str, api_base: str) -> bool:
    """
    Test LLM connection with auto-detection of model type.
    
    Args:
        api_key: API key
        model: Model name
        api_base: API base URL
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    # Detect if it's a Claude model
    is_claude = model.lower().startswith('claude')
    
    if is_claude:
        # Adjust api_base for Claude models if needed
        if not api_base.endswith('/anthropic'):
            if api_base.endswith('/v1'):
                api_base = api_base[:-3] + '/anthropic'
            else:
                api_base = api_base.rstrip('/') + '/anthropic'
        
        return test_anthropic_connection(api_key, model, api_base)
    else:
        return test_openai_connection(api_key, model, api_base)


if __name__ == "__main__":
    """Simple command-line test interface."""
    import sys
    import os
    
    # Add parent directory to path to import config_loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from config_loader import get_api_key, get_api_base, get_model
        
        api_key = get_api_key()
        model = get_model()
        api_base = get_api_base()
        
        if not all([api_key, model, api_base]):
            print("❌ Missing configuration. Please check config.txt")
            sys.exit(1)
        
        success = test_llm_connection(api_key, model, api_base)
        
        if success:
            print("✅ API connection test passed!")
            sys.exit(0)
        else:
            print("❌ API connection test failed!")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1) 