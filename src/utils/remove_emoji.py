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

Standalone script to remove emoji from markdown files
为bash脚本提供emoji删除功能
"""

import re
import sys
import os
import tempfile


def remove_emoji_from_text(text):
    """
    从文本中删除emoji字符
    保留普通的中文、英文、数字和标点符号
    """
    if not text:
        return text
    
    # 使用正则表达式删除emoji
    # 匹配各种emoji Unicode范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 杂项符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何形状扩展
        "\U0001F800-\U0001F8FF"  # 补充箭头-C
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 棋牌符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A
        "\U00002600-\U000026FF"  # 杂项符号
        "\U00002700-\U000027BF"  # 装饰符号
        "\U0001F1E6-\U0001F1FF"  # 地区指示符号（国旗）
        "\U00002B50-\U00002B55"  # 星星等
        "\U0000FE00-\U0000FE0F"  # 变体选择器
        "]+", 
        flags=re.UNICODE
    )
    
    # 删除emoji
    text_without_emoji = emoji_pattern.sub('', text)
    
    # 清理多余的空格，但保留换行符
    # 将多个连续的空格合并为一个，但保留换行符
    text_without_emoji = re.sub(r'[ \t]+', ' ', text_without_emoji)  # 只合并空格和tab
    text_without_emoji = re.sub(r' *\n *', '\n', text_without_emoji)  # 清理换行符前后的空格
    text_without_emoji = re.sub(r'\n{3,}', '\n\n', text_without_emoji)  # 限制连续换行符数量
    
    return text_without_emoji.strip()


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python remove_emoji.py <input_markdown_file>", file=sys.stderr)
        print("Output: Temporary file path or 'UNCHANGED' if no emoji found", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    try:
        # 读取原始markdown文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 删除emoji
        cleaned_content = remove_emoji_from_text(content)
        
        # 如果内容没有变化，输出UNCHANGED
        if cleaned_content == content:
            print("UNCHANGED")
            sys.exit(0)
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.md', prefix='emoji_free_')
        
        try:
            # 写入清理后的内容
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                temp_file.write(cleaned_content)
            
            # 输出临时文件路径
            print(temp_path)
            sys.exit(0)
            
        except Exception as e:
            # 如果写入失败，关闭并删除临时文件
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
