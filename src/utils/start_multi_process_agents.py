#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动多个agent进程，让它们进入IDLE状态等待任务，并持续扫描邮箱进行消息传递
用法:
    python start_agents.py
    python start_agents.py --agent-count 2
"""

import os
import sys
import json
import time
import multiprocessing
import argparse
import threading
from pathlib import Path

# 添加src目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

from src.main import AGIAgentMain
from src.config_loader import get_api_key, get_model, get_api_base


def process_messages(base_output_dir, scan_interval=2):
    """
    持续扫描所有agent的outbox，将消息传递到目标agent的inbox，并移动到sent目录
    
    Args:
        base_output_dir: agent输出目录的基础路径
        scan_interval: 扫描间隔（秒）
    """
    print(f"📬 消息传递服务已启动，扫描间隔: {scan_interval}秒")
    
    while True:
        try:
            # 扫描所有agent目录
            if not os.path.exists(base_output_dir):
                time.sleep(scan_interval)
                continue
            
            # 获取所有agent目录
            agent_dirs = []
            for item in os.listdir(base_output_dir):
                item_path = os.path.join(base_output_dir, item)
                if os.path.isdir(item_path) and item.startswith('agent_'):
                    agent_dirs.append((item, item_path))
            
            # 处理每个agent的outbox
            for agent_id, agent_dir in agent_dirs:
                outbox_dir = os.path.join(agent_dir, "mailboxes", agent_id, "outbox")
                
                if not os.path.exists(outbox_dir):
                    continue
                
                # 获取outbox中的所有消息文件
                outbox_files = [f for f in os.listdir(outbox_dir) if f.endswith('.json')]
                
                for filename in outbox_files:
                    try:
                        message_path = os.path.join(outbox_dir, filename)
                        
                        # 读取消息
                        with open(message_path, 'r', encoding='utf-8') as f:
                            message_data = json.load(f)
                        
                        receiver_id = message_data.get('receiver_id')
                        sender_id = message_data.get('sender_id', agent_id)
                        
                        if not receiver_id:
                            print(f"⚠️  消息 {filename} 缺少 receiver_id，跳过")
                            continue
                        
                        # 确定目标inbox目录：{receiver_id}/mailboxes/{sender_id}/inbox
                        receiver_dir = os.path.join(base_output_dir, receiver_id)
                        target_inbox_dir = os.path.join(receiver_dir, "mailboxes", sender_id, "inbox")
                        
                        # 确保目标目录存在
                        os.makedirs(target_inbox_dir, exist_ok=True)
                        
                        # 确定sent目录：{sender_id}/mailboxes/{sender_id}/sent
                        sent_dir = os.path.join(agent_dir, "mailboxes", agent_id, "sent")
                        os.makedirs(sent_dir, exist_ok=True)
                        
                        # 更新消息状态
                        message_data['delivered'] = True
                        
                        # 复制到目标inbox
                        target_inbox_path = os.path.join(target_inbox_dir, filename)
                        with open(target_inbox_path, 'w', encoding='utf-8') as f:
                            json.dump(message_data, f, indent=2, ensure_ascii=False)
                        
                        # 复制到sent目录
                        sent_path = os.path.join(sent_dir, filename)
                        with open(sent_path, 'w', encoding='utf-8') as f:
                            json.dump(message_data, f, indent=2, ensure_ascii=False)
                        
                        # 从outbox删除
                        os.remove(message_path)
                        
                        print(f"✅ 消息已传递: {sender_id} -> {receiver_id} ({filename})")
                        
                    except Exception as e:
                        print(f"❌ 处理消息 {filename} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 等待下一次扫描
            time.sleep(scan_interval)
            
        except KeyboardInterrupt:
            print("\n📬 消息传递服务收到中断信号，正在停止...")
            break
        except Exception as e:
            print(f"❌ 消息传递服务出错: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(scan_interval)


def run_agent(agent_id, output_dir, api_key, model, api_base, debug_mode=False):
    """运行单个agent进程"""
    try:
        print(f"🚀 启动Agent {agent_id}，输出目录: {output_dir}")
        
        # 设置agent_id到agent context
        from src.tools.agent_context import set_current_agent_id
        set_current_agent_id(agent_id)
        
        # 创建AGIAgentMain实例
        main_app = AGIAgentMain(
            out_dir=output_dir,
            api_key=api_key,
            model=model,
            api_base=api_base,
            debug_mode=debug_mode,
            detailed_summary=True,
            single_task_mode=True,
            interactive_mode=False,
            continue_mode=False
        )
        
        # 注册agent到消息系统
        # 注意：MessageRouter 期望 workspace_root 是包含 workspace 目录的路径
        # 如果传递 output_dir，它会计算 mailbox_root = os.path.dirname(output_dir)/mailboxes
        # 但实际应该是 output_dir/mailboxes，所以需要传递 workspace_dir
        try:
            from src.tools.message_system import get_message_router
            workspace_dir = os.path.join(output_dir, "workspace")
            router = get_message_router(workspace_dir, cleanup_on_init=False)
            router.register_agent(agent_id)
            print(f"📬 Agent {agent_id} 已注册到消息系统")
        except Exception as e:
            print(f"⚠️ 警告: 注册消息系统失败: {e}")
        
        # 初始消息：让agent进入IDLE状态等待任务
        idle_message = (
            f"你好！你是Agent {agent_id}。"
            "当前没有具体任务需要执行。"
            "请使用idle工具，设置sleep=-1进入无限等待模式，等待新的任务消息。"
            "当收到新的任务消息时（通过inbox），请立即读取并开始执行该任务。"
            "任务完成后，请再次使用idle工具进入等待状态。"
        )
        
        # 运行agent，传入初始需求让agent进入IDLE状态
        # 使用-1表示无限循环，agent会在IDLE状态中持续等待
        success = main_app.run(
            user_requirement=idle_message,
            loops=-1  # 无限循环，agent会持续运行并等待任务
        )
        
        print(f"✅ Agent {agent_id} 执行完成")
        return success
        
    except Exception as e:
        print(f"❌ Agent {agent_id} 执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='启动多个agent进程，让它们进入IDLE状态等待任务，并持续扫描邮箱进行消息传递',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 启动2个agent并等待
  %(prog)s --agent-count 3          # 启动3个agent
  %(prog)s --output-base-dir ./agents_output  # 指定输出目录基础路径
  %(prog)s --scan-interval 1       # 设置消息扫描间隔为1秒
        """
    )
    parser.add_argument(
        '--agent-count', '-n',
        type=int,
        default=2,
        help='要启动的agent数量（默认: 2）'
    )
    parser.add_argument(
        '--output-base-dir', '-d',
        default=None,
        help='输出目录基础路径（默认: 当前目录下的agents_output）'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用DEBUG模式'
    )
    parser.add_argument(
        '--scan-interval',
        type=float,
        default=2.0,
        help='消息扫描间隔（秒，默认: 2.0）'
    )
    
    args = parser.parse_args()
    
    # 确定输出目录基础路径
    if args.output_base_dir:
        base_output_dir = args.output_base_dir
    else:
        base_output_dir = os.path.join(script_dir, "agents_output")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 获取API配置
    api_key = get_api_key()
    model = get_model()
    api_base = get_api_base()
    
    if not api_key or not model or not api_base:
        print("❌ 错误: 请确保config/config.txt中配置了api_key、model和api_base")
        sys.exit(1)
    
    print(f"📁 输出目录基础路径: {base_output_dir}")
    print(f"🤖 模型: {model}")
    print(f"🔢 启动Agent数量: {args.agent_count}")
    print("-" * 60)
    
    # 创建进程列表
    processes = []
    agent_output_dirs = []
    
    # 启动agent进程
    for i in range(1, args.agent_count + 1):
        agent_id = f"agent_{i:03d}"
        output_dir = os.path.join(base_output_dir, agent_id)
        agent_output_dirs.append((agent_id, output_dir))
        
        # 创建进程
        p = multiprocessing.Process(
            target=run_agent,
            args=(agent_id, output_dir, api_key, model, api_base, args.debug),
            name=f"Agent-{agent_id}"
        )
        p.start()
        processes.append((agent_id, p))
        print(f"✅ 已启动进程: {agent_id} (PID: {p.pid})")
        time.sleep(1)  # 稍微延迟，避免同时启动造成资源竞争
    
    print("-" * 60)
    print(f"✅ 已启动 {len(processes)} 个Agent进程")
    print("⏳ 等待Agent初始化并进入IDLE状态...")
    time.sleep(10)  # 等待agent初始化完成并进入IDLE状态
    
    # 启动消息传递服务（在独立线程中运行）
    message_thread = threading.Thread(
        target=process_messages,
        args=(base_output_dir, args.scan_interval),
        name="MessageRouter",
        daemon=True
    )
    message_thread.start()
    print("✅ 消息传递服务已启动")
    
    print("-" * 60)
    print("📋 Agent状态:")
    print("   - Agent进程正在运行")
    print("   - Agent正在初始化并准备进入IDLE状态")
    print("   - 消息传递服务正在运行，持续扫描所有agent的outbox")
    print("   - 消息传递流程:")
    print("     1. Agent将消息放入 outbox (agent_XXX/mailboxes/agent_XXX/outbox)")
    print("     2. 消息传递服务读取消息，获取 receiver_id")
    print("     3. 消息被传递到目标agent的inbox (agent_YYY/mailboxes/agent_XXX/inbox)")
    print("     4. 消息被移动到原agent的sent目录 (agent_XXX/mailboxes/agent_XXX/sent)")
    print("-" * 60)
    print("⏸️  按Ctrl+C停止所有Agent进程和消息传递服务")
    
    try:
        # 等待所有进程完成
        for agent_id, p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在停止所有Agent进程和消息传递服务...")
        for agent_id, p in processes:
            if p.is_alive():
                print(f"   停止 {agent_id} (PID: {p.pid})")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    print(f"   强制停止 {agent_id}")
                    p.kill()
        print("✅ 所有Agent进程已停止")


if __name__ == "__main__":
    # 设置multiprocessing启动方法
    if sys.platform == 'darwin':  # macOS
        multiprocessing.set_start_method('spawn', force=True)
    main()

