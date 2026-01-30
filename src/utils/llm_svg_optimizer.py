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

大模型驱动的SVG优化器
使用Anthropic Claude或OpenAI GPT重新生成优化的SVG代码
"""

import json
import os
import re
import requests
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET

from advanced_svg_optimizer import AdvancedSVGOptimizer, OptimizationLevel, OptimizationReport


class LLMProvider(Enum):
    """支持的大模型提供商"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """大模型配置"""
    provider: LLMProvider
    api_key: str
    model: str = None
    base_url: str = None
    
    def __post_init__(self):
        if self.model is None:
            if self.provider == LLMProvider.ANTHROPIC:
                self.model = "claude-3-sonnet-20240229"
            else:  # OpenAI
                self.model = "gpt-4"
        
        if self.base_url is None:
            if self.provider == LLMProvider.ANTHROPIC:
                self.base_url = "https://api.anthropic.com/v1/messages"
            else:  # OpenAI
                self.base_url = "https://api.openai.com/v1/chat/completions"


class LLMSVGOptimizer:
    """大模型驱动的SVG优化器"""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.detector = AdvancedSVGOptimizer(OptimizationLevel.STANDARD)
        
        # 英文提示词模板
        self.prompt_template = """
You are an expert SVG designer tasked with completely redesigning an SVG diagram. I will provide you with:
1. Original SVG code
2. List of detected layout issues

Your task is to create a COMPLETELY NEW SVG design that:
- Extracts ONLY the text content and logical relationships from the original
- IGNORES all original positioning, colors, shapes, and layout
- Redesigns the layout from scratch with better spacing and organization
- Creates a fresh, modern, and well-organized visual representation

## Original SVG Code (for text content extraction only):
```xml
{original_svg}
```

## Detected Layout Issues (to avoid):
{issues_list}

## Design Requirements:
1. **Complete Layout Redesign**: Do NOT copy any positioning from the original
2. **Extract Text Only**: Identify all text elements and their logical relationships
3. **New Visual Hierarchy**: Create proper spacing, grouping, and flow
4. **Modern Design**: Use clean layouts, proper margins, and visual balance
5. **Prevent Issues**: Ensure no overlaps, proper text sizing, adequate spacing
6. **Maintain Logic**: Keep the conceptual relationships between elements clear
7. **Fresh Colors**: Use a modern, harmonious color palette
8. **Better Typography**: Improve text sizing and positioning for readability

## Analysis Steps:
1. Extract all text content from the original SVG
2. Identify the logical structure and relationships
3. Design a completely new layout that better represents these relationships
4. Choose new positions, colors, and arrangements
5. Ensure excellent spacing and no overlaps

## Response Format:
You MUST respond with a JSON object in this exact format:

```json
{{
  "optimized_svg": "<!-- Complete NEW SVG code here -->",
  "changes_made": [
    "Completely redesigned layout with improved spacing",
    "Reorganized elements for better visual hierarchy",
    "Applied modern color scheme",
    "Enhanced typography and text positioning"
  ],
  "issues_fixed": [
    "Eliminated all overlapping elements",
    "Improved text readability",
    "Created proper visual balance"
  ]
}}
```

The "optimized_svg" field must contain a completely redesigned SVG that looks significantly different from the original while preserving the core information and logical relationships.
"""

    def optimize_svg_with_llm(self, svg_content: str, output_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        使用大模型优化SVG
        
        Args:
            svg_content: 原始SVG内容
            output_path: 输出文件路径（可选）
            
        Returns:
            Tuple[优化后的SVG内容, 优化报告]
        """
        print("🔍 检测SVG问题...")
        # 使用现有检测器找出问题
        issues = self.detector._detect_issues(svg_content)
        
        if not issues:
            print("✅ 未检测到问题，返回原始SVG")
            return svg_content, {
                "issues_detected": [],
                "changes_made": [],
                "issues_fixed": [],
                "llm_used": False
            }
        
        print(f"📋 检测到 {len(issues)} 个问题")
        for i, issue in enumerate(issues[:5], 1):
            print(f"  {i}. {issue}")
        if len(issues) > 5:
            print(f"  ... 还有 {len(issues)-5} 个问题")
        
        # 构建提示词
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        prompt = self.prompt_template.format(
            original_svg=svg_content,
            issues_list=issues_text
        )
        
        print("🤖 发送请求到大模型...")
        # 调用大模型
        response = self._call_llm(prompt)
        
        # 解析响应
        optimized_svg, report = self._parse_llm_response(response, issues)
        
        # 保存文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(optimized_svg)
            print(f"💾 优化后的SVG已保存到: {output_path}")
        
        return optimized_svg, report
    
    def _call_llm(self, prompt: str) -> str:
        """调用大模型API"""
        if self.llm_config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(prompt)
        else:
            return self._call_openai(prompt)
    
    def _call_anthropic(self, prompt: str) -> str:
        """调用Anthropic Claude API"""
        # 检查是否是原生Anthropic API还是OpenAI兼容格式
        if 'anthropic.com' in self.llm_config.base_url:
            # 原生Anthropic API格式
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.llm_config.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.llm_config.model,
                "max_tokens": 8000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                self.llm_config.base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["content"][0]["text"]
        else:
            # OpenAI兼容格式（大多数代理和第三方API）
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.api_key}"
            }
            
            data = {
                "model": self.llm_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.3
            }
            
            response = requests.post(
                self.llm_config.base_url + "/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI GPT API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}"
        }
        
        data = {
            "model": self.llm_config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 8000,
            "temperature": 0.3
        }
        
        response = requests.post(
            self.llm_config.base_url + "/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response: str, original_issues: List[str]) -> Tuple[str, Dict]:
        """解析大模型响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个响应作为JSON
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            # 验证必需字段
            if "optimized_svg" not in data:
                raise ValueError("响应中缺少 'optimized_svg' 字段")
            
            optimized_svg = data["optimized_svg"]
            
            # 验证SVG格式
            if not optimized_svg.strip().startswith("<svg"):
                # 如果不是以<svg开头，尝试提取SVG部分
                svg_match = re.search(r'(<svg.*?</svg>)', optimized_svg, re.DOTALL)
                if svg_match:
                    optimized_svg = svg_match.group(1)
                else:
                    raise ValueError("无法找到有效的SVG代码")
            
            # 验证SVG是否是有效的XML
            try:
                ET.fromstring(optimized_svg)
            except ET.ParseError as e:
                raise ValueError(f"生成的SVG不是有效的XML: {e}")
            
            report = {
                "issues_detected": original_issues,
                "changes_made": data.get("changes_made", []),
                "issues_fixed": data.get("issues_fixed", []),
                "llm_used": True,
                "llm_provider": self.llm_config.provider.value,
                "llm_model": self.llm_config.model
            }
            
            return optimized_svg, report
            
        except json.JSONDecodeError as e:
            raise Exception(f"无法解析大模型响应为JSON: {e}\n\n响应内容:\n{response}")
        except Exception as e:
            raise Exception(f"解析大模型响应时出错: {e}\n\n响应内容:\n{response}")
    
    def optimize_svg_file(self, input_file: str, output_file: str) -> Dict:
        """
        优化SVG文件
        
        Args:
            input_file: 输入SVG文件路径
            output_file: 输出SVG文件路径
            
        Returns:
            优化报告
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        optimized_svg, report = self.optimize_svg_with_llm(svg_content, output_file)
        
        return report


def create_llm_optimizer_from_config() -> LLMSVGOptimizer:
    """从config/config.txt配置文件创建LLM优化器"""
    # 导入config_loader
    import sys
    import os
    
    # 添加src目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    try:
        from config_loader import get_api_key, get_api_base, get_model
        
        # 从config.txt读取配置
        api_key = get_api_key()
        api_base = get_api_base()
        model = get_model()
        
        if not api_key or api_key == 'your key':
            raise ValueError(
                "未配置有效的API密钥，请在config/config.txt中设置api_key\n"
                "或设置环境变量 ANTHROPIC_API_KEY 或 OPENAI_API_KEY"
            )
        
        if not api_base:
            raise ValueError("未配置API base URL，请在config/config.txt中设置api_base")
        
        print(f"🔧 从config.txt读取配置:")
        print(f"  API Base: {api_base}")
        print(f"  Model: {model}")
        
        # 根据API base或model判断使用哪个提供商
        if api_base and model:
            api_base_lower = api_base.lower()
            model_lower = model.lower() if model else ""
            
            if 'anthropic' in api_base_lower or 'claude' in model_lower:
                provider = LLMProvider.ANTHROPIC
                print("🤖 识别为 Anthropic Claude API")
            elif 'openai' in api_base_lower or 'gpt' in model_lower:
                provider = LLMProvider.OPENAI  
                print("🤖 识别为 OpenAI GPT API")
            else:
                # 默认尝试OpenAI格式（兼容更多API）
                provider = LLMProvider.OPENAI
                print("🤖 使用 OpenAI 兼容格式")
        else:
            # 回退到环境变量检查
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if anthropic_key:
                provider = LLMProvider.ANTHROPIC
                api_key = anthropic_key
                print("🔧 回退到环境变量 ANTHROPIC_API_KEY")
            elif openai_key:
                provider = LLMProvider.OPENAI
                api_key = openai_key
                print("🔧 回退到环境变量 OPENAI_API_KEY")
            else:
                # 默认使用OpenAI格式
                provider = LLMProvider.OPENAI
                print("🔧 使用默认 OpenAI 格式")
        
        config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=api_base
        )
        
        return LLMSVGOptimizer(config)
        
    except ImportError as e:
        print(f"⚠️ 无法导入config_loader: {e}")
        # 回退到环境变量
        return create_llm_optimizer_from_env()
    except Exception as e:
        print(f"⚠️ 读取配置失败: {e}")
        # 回退到环境变量
        return create_llm_optimizer_from_env()


def create_llm_optimizer_from_env() -> LLMSVGOptimizer:
    """从环境变量创建LLM优化器（回退方案）"""
    # 检查Anthropic API Key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if anthropic_key:
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=anthropic_key
        )
        print("🔧 使用 Anthropic Claude (环境变量)")
    elif openai_key:
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=openai_key
        )
        print("🔧 使用 OpenAI GPT (环境变量)")
    else:
        raise ValueError(
            "请在config/config.txt中配置api_key和api_base，\n"
            "或设置环境变量 ANTHROPIC_API_KEY 或 OPENAI_API_KEY\n"
            "例如: export ANTHROPIC_API_KEY='your-key-here'"
        )
    
    return LLMSVGOptimizer(config)


# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python llm_svg_optimizer.py <input.svg> <output.svg>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        optimizer = create_llm_optimizer_from_config()
        report = optimizer.optimize_svg_file(input_file, output_file)
        
        print("\n" + "="*50)
        print("📊 优化报告")
        print("="*50)
        print(f"检测到问题: {len(report['issues_detected'])}个")
        print(f"使用的大模型: {report.get('llm_provider', 'unknown')} - {report.get('llm_model', 'unknown')}")
        
        if report['changes_made']:
            print("\n🔧 主要修改:")
            for change in report['changes_made']:
                print(f"  • {change}")
        
        if report['issues_fixed']:
            print("\n✅ 修复的问题:")
            for fix in report['issues_fixed']:
                print(f"  • {fix}")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
