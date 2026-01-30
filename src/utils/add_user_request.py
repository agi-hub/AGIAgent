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

脚本：将用户需求以JSON邮件形式写入manager的inbox邮箱
用法: 
    python add_user_request.py "用户需求内容"
    python add_user_request.py -d /path/to/output "用户需求内容"
    python add_user_request.py (交互式输入)
"""

import os
import json
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path


def find_next_extmsg_id(inbox_dir):
    """查找下一个可用的extmsg_XXXXXX编号（六位数字）"""
    if not os.path.exists(inbox_dir):
        return 1
    
    max_id = 0
    pattern = re.compile(r'extmsg_(\d+)\.json')
    
    for filename in os.listdir(inbox_dir):
        match = pattern.match(filename)
        if match:
            msg_id = int(match.group(1))
            max_id = max(max_id, msg_id)
    
    return max_id + 1


def create_user_request_message(content_text, output_dir):
    """创建用户需求邮件并写入manager的inbox"""
    # 确定inbox目录
    inbox_dir = os.path.join(output_dir, "mailboxes", "manager", "inbox")
    
    # 确保目录存在
    os.makedirs(inbox_dir, exist_ok=True)
    
    # 查找下一个可用的extmsg编号（六位数字格式）
    next_id = find_next_extmsg_id(inbox_dir)
    message_id = f"extmsg_{next_id:06d}"
    
    # 创建消息对象
    message = {
        "message_id": message_id,
        "sender_id": "user",
        "receiver_id": "manager",
        "message_type": "collaboration",
        "content": {
            "text": content_text
        },
        "priority": 2,
        "requires_response": False,
        "timestamp": datetime.now().isoformat(),
        "delivered": False,
        "read": False
    }
    
    # 写入文件
    file_path = os.path.join(inbox_dir, f"{message_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(message, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 成功创建用户需求邮件:")
    print(f"   文件路径: {file_path}")
    print(f"   消息ID: {message_id}")
    print(f"   内容: {content_text[:50]}{'...' if len(content_text) > 50 else ''}")
    
    return file_path


def find_latest_output_dir(script_dir):
    """查找最新的output目录"""
    output_dirs = []
    for item in os.listdir(script_dir):
        if item.startswith("output_") and os.path.isdir(os.path.join(script_dir, item)):
            output_dirs.append(item)
    
    if output_dirs:
        # 按时间戳排序，使用最新的
        output_dirs.sort(reverse=True)
        return os.path.join(script_dir, output_dirs[0])
    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将用户需求以JSON邮件形式写入manager的inbox邮箱',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "用户需求内容"
  %(prog)s -d /path/to/output "用户需求内容"
  %(prog)s -d output_20251211_091251 "用户需求内容"
  %(prog)s  # 交互式输入
        """
    )
    parser.add_argument(
        '-d', '--dir',
        dest='output_dir',
        help='指定输出目录（如果不指定，则自动查找最新的output_*目录）'
    )
    parser.add_argument(
        'content',
        nargs='*',
        help='用户需求内容'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定输出目录
    if args.output_dir:
        # 用户指定了目录
        if os.path.isabs(args.output_dir):
            output_dir = args.output_dir
        else:
            # 相对路径，相对于脚本目录
            output_dir = os.path.join(script_dir, args.output_dir)
        
        if not os.path.exists(output_dir):
            print(f"❌ 错误: 指定的目录不存在: {output_dir}")
            sys.exit(1)
        
        print(f"📁 使用指定输出目录: {output_dir}")
    else:
        # 自动查找最新的output目录
        latest_dir = find_latest_output_dir(script_dir)
        if latest_dir:
            output_dir = latest_dir
            print(f"📁 自动找到最新输出目录: {output_dir}")
        else:
            # 如果没有找到output目录，使用当前目录
            output_dir = script_dir
            print(f"⚠️  未找到output目录，使用当前目录: {output_dir}")
    
    # 获取用户需求内容
    if args.content:
        # 从命令行参数获取
        content = " ".join(args.content)
    else:
        # 交互式输入
        print("\n请输入用户需求:")
        content = input("> ").strip()
        
        if not content:
            print("❌ 错误: 需求内容不能为空")
            sys.exit(1)
    
    # 创建消息
    try:
        create_user_request_message(content, output_dir)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

