#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向运行中的agent发送任务消息
用法:
    python send_task.py "任务内容"
    python send_task.py -a agent_001 "进行Python异步编程调研"
    python send_task.py --agent agent_001 --dir ./agents_output "任务内容"
    python send_task.py --all "发送给所有agent的任务"
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


def send_message_to_agent(agent_id, content_text, output_dir, sender_id="user"):
    """发送消息到指定agent的inbox"""
    # 确定inbox目录
    inbox_dir = os.path.join(output_dir, "mailboxes", agent_id, "inbox")
    
    # 确保目录存在
    os.makedirs(inbox_dir, exist_ok=True)
    
    # 查找下一个可用的extmsg编号（六位数字格式）
    next_id = find_next_extmsg_id(inbox_dir)
    message_id = f"extmsg_{next_id:06d}"
    
    # 创建消息对象
    message = {
        "message_id": message_id,
        "sender_id": sender_id,
        "receiver_id": agent_id,
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
    
    print(f"✅ 成功发送消息到 {agent_id}:")
    print(f"   文件路径: {file_path}")
    print(f"   消息ID: {message_id}")
    print(f"   内容: {content_text[:50]}{'...' if len(content_text) > 50 else ''}")
    
    return file_path


def find_agent_dirs(base_dir):
    """查找所有agent目录"""
    agent_dirs = []
    if not os.path.exists(base_dir):
        return agent_dirs
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("agent_"):
            # 检查是否有mailboxes目录
            mailboxes_dir = os.path.join(item_path, "mailboxes")
            if os.path.exists(mailboxes_dir):
                agent_id = item
                agent_dirs.append((agent_id, item_path))
    
    return agent_dirs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='向运行中的agent发送任务消息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "进行Python异步编程调研"
  %(prog)s -a agent_001 "进行Python异步编程调研"
  %(prog)s --agent agent_001 --dir ./agents_output "任务内容"
  %(prog)s --all --dir ./agents_output "发送给所有agent的任务"
        """
    )
    parser.add_argument(
        '-a', '--agent',
        dest='agent_id',
        help='目标agent ID（例如: agent_001），如果不指定且不使用--all，则发送给第一个找到的agent'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='发送给所有找到的agent'
    )
    parser.add_argument(
        '-d', '--dir',
        dest='output_dir',
        help='输出目录基础路径（默认: 当前目录下的agents_output）'
    )
    parser.add_argument(
        'content',
        nargs='*',
        help='任务内容'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定输出目录基础路径
    if args.output_dir:
        base_output_dir = args.output_dir
        if not os.path.isabs(base_output_dir):
            base_output_dir = os.path.join(script_dir, base_output_dir)
    else:
        base_output_dir = os.path.join(script_dir, "agents_output")
    
    # 获取任务内容
    if args.content:
        content = " ".join(args.content)
    else:
        # 交互式输入
        print("\n请输入任务内容:")
        content = input("> ").strip()
        
        if not content:
            print("❌ 错误: 任务内容不能为空")
            sys.exit(1)
    
    # 查找agent目录
    if args.all:
        # 发送给所有agent
        agent_dirs = find_agent_dirs(base_output_dir)
        if not agent_dirs:
            print(f"❌ 错误: 在 {base_output_dir} 中未找到任何agent目录")
            sys.exit(1)
        
        print(f"📨 发送任务到 {len(agent_dirs)} 个Agent:")
        for agent_id, output_dir in agent_dirs:
            send_message_to_agent(agent_id, content, output_dir, sender_id="user")
        print(f"✅ 任务已发送给所有 {len(agent_dirs)} 个Agent")
        
    elif args.agent_id:
        # 发送给指定agent
        agent_id = args.agent_id
        output_dir = os.path.join(base_output_dir, agent_id)
        
        if not os.path.exists(output_dir):
            print(f"❌ 错误: Agent目录不存在: {output_dir}")
            sys.exit(1)
        
        send_message_to_agent(agent_id, content, output_dir, sender_id="user")
        
    else:
        # 发送给第一个找到的agent
        agent_dirs = find_agent_dirs(base_output_dir)
        if not agent_dirs:
            print(f"❌ 错误: 在 {base_output_dir} 中未找到任何agent目录")
            print(f"   请使用 -a 指定agent ID，或使用 --all 发送给所有agent")
            sys.exit(1)
        
        agent_id, output_dir = agent_dirs[0]
        print(f"📨 未指定agent，发送给第一个找到的agent: {agent_id}")
        send_message_to_agent(agent_id, content, output_dir, sender_id="user")


if __name__ == "__main__":
    main()

