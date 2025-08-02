#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIBot Message System
Supports message passing and mailbox functionality between agents
"""

import os
import json
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from .id_manager import generate_message_id
from .print_system import print_system, print_current, streaming_context


class MessageType(Enum):
    """Message type enumeration"""
    STATUS_UPDATE = "status_update"      # Status update
    TASK_REQUEST = "task_request"        # Task request
    TASK_RESPONSE = "task_response"      # Task response
    COLLABORATION = "collaboration"      # Collaboration message
    BROADCAST = "broadcast"              # Broadcast message
    SYSTEM = "system"                    # System message
    ERROR = "error"                      # Error message


class MessagePriority(Enum):
    """Message priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class Message:
    """Message class"""
    def __init__(self, 
                 sender_id: str, 
                 receiver_id: str, 
                 message_type: MessageType,
                 content: Dict[str, Any],
                 priority: MessagePriority = MessagePriority.NORMAL,
                 requires_response: bool = False):
        self.message_id = generate_message_id("msg")
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.requires_response = requires_response
        self.timestamp = datetime.now().isoformat()
        self.delivered = False
        self.read = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "requires_response": self.requires_response,
            "timestamp": self.timestamp,
            "delivered": self.delivered,
            "read": self.read
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message object from dictionary"""
        msg = cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            priority=MessagePriority(data["priority"]),
            requires_response=data.get("requires_response", False)
        )
        msg.message_id = data["message_id"]
        msg.timestamp = data["timestamp"]
        msg.delivered = data.get("delivered", False)
        msg.read = data.get("read", False)
        return msg


class StatusUpdateMessage:
    """Status update message content structure"""
    @staticmethod
    def create_content(round_number: int, 
                      task_completed: bool,
                      llm_response_preview: str,
                      tool_calls_summary: List[str],
                      current_task_description: str = "",
                      error_message: str = None) -> Dict[str, Any]:
        return {
            "round_number": round_number,
            "task_completed": task_completed,
            "llm_response_preview": llm_response_preview,  # 不再截断，显示完整内容
            "tool_calls_summary": tool_calls_summary,
            "current_task_description": current_task_description,
            "error_message": error_message,
            "update_time": datetime.now().isoformat()
        }


class Mailbox:
    """Agent mailbox system"""
    def __init__(self, agent_id: str, mailbox_root: str):
        self.agent_id = agent_id
        self.mailbox_root = mailbox_root or "."
        self.mailbox_dir = os.path.join(self.mailbox_root, agent_id)
        self.inbox_dir = os.path.join(self.mailbox_dir, "inbox")
        self.outbox_dir = os.path.join(self.mailbox_dir, "outbox")
        self.sent_dir = os.path.join(self.mailbox_dir, "sent")
        
        # Create mailbox directories
        for dir_path in [self.inbox_dir, self.outbox_dir, self.sent_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self._lock = threading.Lock()
    
    def send_message(self, message: Message) -> bool:
        """Send message"""
        try:
            with self._lock:
                # Save to outbox
                outbox_file = os.path.join(self.outbox_dir, f"{message.message_id}.json")
                with open(outbox_file, 'w', encoding='utf-8') as f:
                    json.dump(message.to_dict(), f, indent=2, ensure_ascii=False)
                
                # Save to sent
                sent_file = os.path.join(self.sent_dir, f"{message.message_id}.json")
                with open(sent_file, 'w', encoding='utf-8') as f:
                    json.dump(message.to_dict(), f, indent=2, ensure_ascii=False)
                
                print_current(f"📤 Agent {self.agent_id} sent message {message.message_id} to {message.receiver_id}")
                return True
        except Exception as e:
            print_current(f"❌ Failed to send message: {e}")
            return False
    
    def receive_message(self, message: Message) -> bool:
        """Receive message"""
        try:
            with self._lock:
                inbox_file = os.path.join(self.inbox_dir, f"{message.message_id}.json")
                message.delivered = True
                
                with open(inbox_file, 'w', encoding='utf-8') as f:
                    json.dump(message.to_dict(), f, indent=2, ensure_ascii=False)
                
                print_current(f"📥 Agent {self.agent_id} received message {message.message_id} from {message.sender_id}")
                return True
        except Exception as e:
            print_current(f"❌ Failed to receive message: {e}")
            return False
    
    def get_unread_messages(self) -> List[Message]:
        """Get unread messages"""
        messages = []
        try:
            with self._lock:
                if not os.path.exists(self.inbox_dir):
                    return messages
                
                for filename in os.listdir(self.inbox_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.inbox_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if not data.get('read', False):
                                message = Message.from_dict(data)
                                messages.append(message)
                        except Exception as e:
                            print_current(f"⚠️ Failed to read message file {filename}: {e}")
                
                # Sort by priority and time
                messages.sort(key=lambda m: (m.priority.value, m.timestamp), reverse=True)
        except Exception as e:
            print_current(f"❌ Failed to get unread messages: {e}")
        
        return messages
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages (including read and unread)"""
        messages = []
        try:
            with self._lock:
                if not os.path.exists(self.inbox_dir):
                    return messages
                
                for filename in os.listdir(self.inbox_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.inbox_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            message = Message.from_dict(data)
                            messages.append(message)
                        except Exception as e:
                            print_current(f"⚠️ Failed to read message file {filename}: {e}")
                
                # Sort by priority and time
                messages.sort(key=lambda m: (m.priority.value, m.timestamp), reverse=True)
        except Exception as e:
            print_current(f"❌ Failed to get all messages: {e}")
        
        return messages
    
    def mark_as_read(self, message_id: str) -> bool:
        """Mark message as read"""
        try:
            with self._lock:
                inbox_file = os.path.join(self.inbox_dir, f"{message_id}.json")
                if os.path.exists(inbox_file):
                    with open(inbox_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['read'] = True
                    
                    with open(inbox_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    return True
        except Exception as e:
            print_current(f"❌ Failed to mark message as read: {e}")
        
        return False
    
    def get_message_stats(self) -> Dict[str, int]:
        """Get mailbox statistics"""
        stats = {
            "total_received": 0,
            "unread_count": 0,
            "sent_count": 0
        }
        
        try:
            with self._lock:
                # Count inbox
                if os.path.exists(self.inbox_dir):
                    inbox_files = [f for f in os.listdir(self.inbox_dir) if f.endswith('.json')]
                    stats["total_received"] = len(inbox_files)
                    
                    unread_count = 0
                    for filename in inbox_files:
                        filepath = os.path.join(self.inbox_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if not data.get('read', False):
                                unread_count += 1
                        except:
                            pass
                    stats["unread_count"] = unread_count
                
                # Count sent
                if os.path.exists(self.sent_dir):
                    sent_files = [f for f in os.listdir(self.sent_dir) if f.endswith('.json')]
                    stats["sent_count"] = len(sent_files)
        
        except Exception as e:
            print_current(f"❌ Failed to get message stats: {e}")
        
        return stats


class MessageRouter:
    """Message router"""
    def __init__(self, workspace_root: str, mailbox_root: str = None, cleanup_on_init: bool = True):
        """
        Initialize message router
        
        Args:
            workspace_root: Workspace root directory
            mailbox_root: Mailbox root directory (optional)
            cleanup_on_init: Whether to cleanup old mailboxes on initialization
        """
        self.workspace_root = workspace_root
        
        # 🔧 Fix: Ensure mailboxes are created in appropriate output directory
        if mailbox_root is None:
            # If workspace_root ends with 'workspace', place mailboxes in its parent directory
            if os.path.basename(workspace_root) == "workspace":
                outdir = os.path.dirname(workspace_root)
                
                # 🔧 Additional fix: If outdir is the AGIBot project root (contains src/, agibot.py),
                # place mailboxes in the workspace directory instead to avoid cluttering project root
                if (os.path.exists(os.path.join(outdir, "src")) and 
                    os.path.exists(os.path.join(outdir, "src", "main.py")) and 
                    os.path.exists(os.path.join(outdir, "agibot.py"))):
                    # This indicates we're running from AGIBot project root
                    # Place mailboxes in workspace directory to keep project root clean
                    self.mailbox_root = os.path.join(workspace_root, "mailboxes")
                else:
                    # Normal case: place mailboxes in output directory
                    self.mailbox_root = os.path.join(outdir, "mailboxes")
            else:
                # workspace_root is not a 'workspace' subdirectory, place mailboxes in it
                self.mailbox_root = os.path.join(workspace_root, "mailboxes")
        else:
            self.mailbox_root = mailbox_root
            
        self.mailboxes = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._background_thread = None
        
        # Create mailbox root directory
        os.makedirs(self.mailbox_root, exist_ok=True)
        
        # Cleanup old mailboxes if requested
        if cleanup_on_init:
            self._cleanup_all_mailboxes()
        
        # Ensure manager mailbox is always registered
        self.register_agent("manager")
        
        # 🔧 修复：暂时禁用后台处理，避免与手动处理冲突
        # self._start_background_processing()


    def _start_background_processing(self):
        """启动后台消息处理线程"""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._background_thread = threading.Thread(
                target=self._process_messages_continuously,
                daemon=True,
                name="MessageRouter-Background"
            )
            self._background_thread.start()

    def _process_messages_continuously(self):
        """持续处理消息的后台线程"""
        while not self._stop_event.is_set():
            try:
                # 处理所有邮箱的消息
                processed_count = self.process_all_messages_once()
                if processed_count > 0:
                    print_current(f"📬 Processed {processed_count} messages")
                
                # 短暂休眠，避免CPU过度占用
                time.sleep(0.1)
                
            except Exception as e:
                print_current(f"⚠️ Error in background message processing: {e}")
                time.sleep(1)  # 出错时等待更长时间
                
    def register_agent(self, agent_id: str) -> Optional[Mailbox]:
        """
        Register new agent and create mailbox
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Created mailbox
        """
        with self._lock:
            # 验证agent ID格式
            if not self._is_valid_agent_id(agent_id):
                print_current(f"⚠️ Invalid agent ID format: {agent_id}. Expected format: agent_XXX")
                return None
                
            if agent_id not in self.mailboxes:
                mailbox = Mailbox(agent_id, self.mailbox_root)
                self.mailboxes[agent_id] = mailbox
                return mailbox
            else:
                return self.mailboxes[agent_id]

    def route_message(self, message: Message) -> bool:
        """
        Route message to target agent
        
        Args:
            message: Message to route
            
        Returns:
            True if message routed successfully
        """
        try:
            with self._lock:
                # Get target mailbox
                target_mailbox = self.mailboxes.get(message.receiver_id)
                if not target_mailbox:
                    print_current(f"⚠️ Target agent {message.receiver_id} not found, auto-registering")
                    # 自动注册目标agent
                    target_mailbox = self.register_agent(message.receiver_id)
                    if not target_mailbox:
                        print_current(f"❌ Failed to register agent {message.receiver_id} - invalid agent ID format")
                        return False
                
                # Deliver message
                success = target_mailbox.receive_message(message)
                if success:
                    print_current(f"✅ Message {message.message_id} routed from {message.sender_id} to {message.receiver_id}")
                else:
                    print_current(f"❌ Failed to route message {message.message_id} to {message.receiver_id}")
                
                return success
                
        except Exception as e:
            print_current(f"❌ Error routing message {message.message_id}: {e}")
            return False

    def _process_messages(self):
        """Process messages from all mailboxes"""
        try:
            with self._lock:
                processed_count = 0
                for mailbox in self.mailboxes.values():
                    try:
                        count = self._process_outbox(mailbox)
                        processed_count += count
                    except Exception as e:
                        print_current(f"⚠️ Error processing mailbox for {mailbox.agent_id}: {e}")
                
                return processed_count
        except Exception as e:
            print_current(f"❌ Error in message processing: {e}")
            return 0

    def _process_outbox(self, mailbox: Mailbox):
        """
        Process outbox messages for a specific mailbox
        
        Args:
            mailbox: Mailbox to process
            
        Returns:
            Number of messages processed
        """
        processed_count = 0
        try:
            outbox_files = []
            if os.path.exists(mailbox.outbox_dir):
                outbox_files = [f for f in os.listdir(mailbox.outbox_dir) if f.endswith('.json')]
            
            for filename in outbox_files:
                try:
                    filepath = os.path.join(mailbox.outbox_dir, filename)
                    
                    # Read message
                    with open(filepath, 'r', encoding='utf-8') as f:
                        message_data = json.load(f)
                    
                    message = Message.from_dict(message_data)
                    
                    # Route message
                    if self._route_message_direct(message):
                        # Remove from outbox after successful routing
                        os.remove(filepath)
                        processed_count += 1
                    else:
                        print_current(f"⚠️ Failed to route message {message.message_id}, keeping in outbox")
                        
                except Exception as e:
                    print_current(f"⚠️ Error processing outbox file {filename}: {e}")
                    
        except Exception as e:
            print_current(f"❌ Error processing outbox for {mailbox.agent_id}: {e}")
        
        return processed_count

    def _route_message_direct(self, message: Message) -> bool:
        """
        直接路由消息（线程安全版本）
        
        Args:
            message: Message to route
            
        Returns:
            True if message routed successfully
        """
        try:
            # 使用锁来保护mailbox字典的访问和修改
            with self._lock:
                target_mailbox = self.mailboxes.get(message.receiver_id)
                if not target_mailbox:
                    # 🔧 修复：检查是否已经有其他线程注册了这个agent
                    target_mailbox = self.mailboxes.get(message.receiver_id)
                    if not target_mailbox:
                        print_current(f"⚠️ Target agent {message.receiver_id} not found for message {message.message_id}")
                        # 尝试自动注册目标agent
                        target_mailbox = Mailbox(message.receiver_id, self.mailbox_root)
                        self.mailboxes[message.receiver_id] = target_mailbox
                        print_current(f"📬 Auto-registered mailbox for agent {message.receiver_id}")
            
            # Deliver message (在锁外执行，避免死锁)
            success = target_mailbox.receive_message(message)
            return success
            
        except Exception as e:
            print_current(f"❌ Error routing message {message.message_id}: {e}")
            return False

    def stop(self):
        """Stop the message router"""
        self._stop_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=2)

    def process_all_messages_once(self) -> int:
        """
        Process all pending messages once
        
        Returns:
            Number of messages processed
        """
        processed_count = 0
        try:
            # 🔧 修复：创建mailboxes的副本以避免迭代时字典大小变化的问题
            with self._lock:
                # 创建当前mailboxes的副本，避免在迭代过程中字典被修改
                mailboxes_snapshot = list(self.mailboxes.values())
            
            # 在锁外处理消息，避免长时间持锁
            for mailbox in mailboxes_snapshot:
                try:
                    count = self._process_outbox_direct(mailbox)
                    processed_count += count
                except Exception as e:
                    print_current(f"⚠️ Error processing mailbox for {mailbox.agent_id}: {e}")
                
            return processed_count
        except Exception as e:
            print_current(f"❌ Error in process_all_messages_once: {e}")
            return 0

    def _process_outbox_direct(self, mailbox: Mailbox) -> int:
        """
        直接处理outbox消息（不加锁，由调用者负责同步）
        
        Args:
            mailbox: Mailbox to process
            
        Returns:
            Number of messages processed
        """
        processed_count = 0
        try:
            if not os.path.exists(mailbox.outbox_dir):
                return 0
                
            outbox_files = [f for f in os.listdir(mailbox.outbox_dir) if f.endswith('.json')]
            
            for filename in outbox_files:
                try:
                    filepath = os.path.join(mailbox.outbox_dir, filename)
                    
                    # Check if file still exists (could be processed by another thread)
                    if not os.path.exists(filepath):
                        continue
                    
                    # Read message
                    with open(filepath, 'r', encoding='utf-8') as f:
                        message_data = json.load(f)
                    
                    message = Message.from_dict(message_data)
                    
                    # Route message
                    if self._route_message_direct(message):
                        # Remove from outbox after successful routing (with existence check)
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                processed_count += 1
                                print_current(f"📨 Successfully routed and removed message {message.message_id} from outbox")
                        except FileNotFoundError:
                            # File already removed by another thread, this is fine
                            processed_count += 1
                            print_current(f"📨 Message {message.message_id} already processed by another thread")
                    else:
                        print_current(f"⚠️ Failed to route message {message.message_id}, keeping in outbox")
                        
                except FileNotFoundError:
                    # File was already processed by another thread, skip silently
                    continue
                except Exception as e:
                    print_current(f"⚠️ Error processing outbox file {filename}: {e}")
                    
        except Exception as e:
            print_current(f"❌ Error processing outbox for {mailbox.agent_id}: {e}")
        
        return processed_count

    def _cleanup_all_mailboxes(self):
        """Clean up all historical mailbox directories (cleanup before running)"""
        try:
            if not os.path.exists(self.mailbox_root):
                return
            
            cleaned_count = 0
            for agent_dir in os.listdir(self.mailbox_root):
                agent_path = os.path.join(self.mailbox_root, agent_dir)
                
                # Skip non-directories
                if not os.path.isdir(agent_path):
                    continue
                
                # Skip manager mailbox to preserve it
                if agent_dir == "manager":
                    continue
                
                # Delete all other mailbox directories
                try:
                    import shutil
                    shutil.rmtree(agent_path, ignore_errors=True)
                    cleaned_count += 1
                    print_current(f"🧹 Cleaned up mailbox: {agent_dir}")
                except Exception as e:
                    print_current(f"⚠️ Failed to cleanup mailbox {agent_dir}: {e}")
            
            if cleaned_count > 0:
                print_current(f"🧹 Cleaned up {cleaned_count} mailboxes before startup")
                
        except Exception as e:
            print_current(f"⚠️ Failed to cleanup all mailboxes: {e}")
    
    def _cleanup_old_mailboxes(self, max_age_hours: int = 24):
        """Clean up expired mailbox directories"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            if not os.path.exists(self.mailbox_root):
                return
            
            cleaned_count = 0
            for agent_dir in os.listdir(self.mailbox_root):
                agent_path = os.path.join(self.mailbox_root, agent_dir)
                
                # Skip non-directories
                if not os.path.isdir(agent_path):
                    continue
                
                # Check directory modification time
                try:
                    dir_mtime = os.path.getmtime(agent_path)
                    if current_time - dir_mtime > max_age_seconds:
                        # Delete expired mailbox
                        import shutil
                        shutil.rmtree(agent_path, ignore_errors=True)
                        cleaned_count += 1
                        print_current(f"🧹 Cleaned up old mailbox: {agent_dir}")
                except Exception as e:
                    print_current(f"⚠️ Failed to check mailbox {agent_dir}: {e}")
            
            if cleaned_count > 0:
                print_current(f"🧹 Cleaned up {cleaned_count} old mailboxes")
                
        except Exception as e:
            print_current(f"⚠️ Failed to cleanup old mailboxes: {e}")

    def get_mailbox(self, agent_id: str) -> Optional[Mailbox]:
        """Get agent mailbox"""
        with self._lock:
            return self.mailboxes.get(agent_id)

    def get_all_agents(self) -> List[str]:
        """Get all registered agent IDs"""
        try:
            with self._lock:
                return list(self.mailboxes.keys())
        except Exception as e:
            print_current(f"❌ Error in get_all_agents: {e}")
            return []

    def broadcast_message(self, sender_id: str, content: Dict[str, Any], 
                         exclude_agents: Optional[List[str]] = None) -> int:
        """Broadcast message to all agents including sender"""
        exclude_agents = exclude_agents or []
        sent_count = 0
        
        with self._lock:
            # 直接访问 mailboxes 避免递归锁
            sender_mailbox = self.mailboxes.get(sender_id)
            if not sender_mailbox:
                return 0
                
            for agent_id in self.mailboxes.keys():
                # 移除 agent_id != sender_id 条件，让发送者也能收到广播消息
                if agent_id not in exclude_agents:
                    message = Message(
                        sender_id=sender_id,
                        receiver_id=agent_id,
                        message_type=MessageType.BROADCAST,
                        content=content,
                        priority=MessagePriority.NORMAL
                    )
                    
                    if sender_mailbox.send_message(message):
                        sent_count += 1
        
        return sent_count
    
    def _is_valid_agent_id(self, agent_id: str) -> bool:
        """
        Validate if the agent ID format is correct

        Args:
            agent_id: Agent ID to validate

        Returns:
            True if agent ID format is valid
        """
        import re

        # Allowed formats:
        # 1. "manager" (special admin ID)
        # 2. agent_XXX (letters, numbers, and underscores allowed)
        if agent_id == "manager":
            return True

        # Must start with "agent_", followed by letters, numbers, or underscores
        # Examples: agent_001, agent_main, agent_primary, agent_test_1, etc.
        pattern = r'^agent_[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, agent_id))


class MessageFormatter:
    """Message formatter for formatting mailbox messages as LLM context"""
    
    @staticmethod
    def format_messages_for_llm(messages: List[Message]) -> str:
        """
        Format message list as LLM context string
        
        Args:
            messages: Message list
            
        Returns:
            Formatted string
        """
        if not messages:
            return ""
        
        formatted_parts = []
        formatted_parts.append("📬 Inbox Messages:")
        formatted_parts.append("=" * 50)
        
        for i, message in enumerate(messages, 1):
            # Basic information
            sender_name = message.sender_id[:8] if len(message.sender_id) > 8 else message.sender_id
            message_type = message.message_type.value
            priority = message.priority.name
            timestamp = message.timestamp
            
            formatted_parts.append(f"\n📨 Message {i}:")
            formatted_parts.append(f"Sender: {sender_name}")
            formatted_parts.append(f"Type: {message_type}")
            formatted_parts.append(f"Priority: {priority}")
            formatted_parts.append(f"Time: {timestamp}")
            
            # Format message content
            content_str = MessageFormatter._format_message_content(message)
            if content_str:
                formatted_parts.append(f"Content:\n{content_str}")
            
            if message.requires_response:
                formatted_parts.append("⚠️ This message requires a response")
            
            formatted_parts.append("-" * 40)
        
        formatted_parts.append(f"\nTotal {len(messages)} unread messages")
        formatted_parts.append("=" * 50)
        
        return "\n".join(formatted_parts)
    
    @staticmethod
    def _format_message_content(message: Message) -> str:
        """Format content part of a single message"""
        content = message.content
        message_type = message.message_type
        
        if message_type == MessageType.STATUS_UPDATE:
            return MessageFormatter._format_status_update(content)
        elif message_type == MessageType.TASK_REQUEST:
            return MessageFormatter._format_task_request(content)
        elif message_type == MessageType.COLLABORATION:
            return MessageFormatter._format_collaboration(content)
        elif message_type == MessageType.BROADCAST:
            return MessageFormatter._format_broadcast(content)
        elif message_type == MessageType.SYSTEM:
            return MessageFormatter._format_system(content)
        elif message_type == MessageType.ERROR:
            return MessageFormatter._format_error(content)
        else:
            # Generic format
            return MessageFormatter._format_generic(content)
    
    @staticmethod
    def _format_status_update(content: Dict[str, Any]) -> str:
        """Format status update message"""
        parts = []
        parts.append(f"  Round: {content.get('round_number', 'Unknown')}")
        parts.append(f"  Task Status: {'Completed' if content.get('task_completed') else 'In Progress'}")
        
        if content.get('current_task_description'):
            parts.append(f"  Current Task: {content['current_task_description']}")
        
        if content.get('llm_response_preview'):
            # 不再截断 LLM 响应预览，显示完整内容
            parts.append(f"  LLM Response Preview: {content['llm_response_preview']}")
        
        if content.get('tool_calls_summary'):
            tools = ", ".join(content['tool_calls_summary'])
            parts.append(f"  Tools Used: {tools}")
        
        if content.get('error_message'):
            parts.append(f"  ❌ Error: {content['error_message']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_task_request(content: Dict[str, Any]) -> str:
        """Format task request message"""
        parts = []
        
        if content.get('task'):
            parts.append(f"  Task Description: {content['task']}")
        
        if content.get('priority'):
            parts.append(f"  Priority: {content['priority']}")
        
        if content.get('deadline'):
            parts.append(f"  Deadline: {content['deadline']}")
        
        if content.get('requirements'):
            parts.append(f"  Requirements: {content['requirements']}")
        
        if content.get('description'):
            parts.append(f"  Details: {content['description']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_collaboration(content: Dict[str, Any]) -> str:
        """Format collaboration message"""
        parts = []
        
        if content.get('collaboration_type'):
            parts.append(f"  Collaboration Type: {content['collaboration_type']}")
        
        if content.get('proposal'):
            parts.append(f"  Proposal: {content['proposal']}")
        
        if content.get('shared_resources'):
            resources = ", ".join(content['shared_resources'])
            parts.append(f"  Shared Resources: {resources}")
        
        if content.get('message'):
            parts.append(f"  Message: {content['message']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_broadcast(content: Dict[str, Any]) -> str:
        """Format broadcast message"""
        parts = []
        
        # 处理常见的广播消息字段
        if content.get('announcement'):
            parts.append(f"  📢 Announcement: {content['announcement']}")
        
        if content.get('type'):
            parts.append(f"  Broadcast Type: {content['type']}")
        
        if content.get('content'):
            if isinstance(content['content'], dict):
                for key, value in content['content'].items():
                    parts.append(f"  {key}: {value}")
            else:
                parts.append(f"  Content: {content['content']}")
        
        # 处理其他所有字段（不截断内容）
        handled_keys = {'announcement', 'type', 'content'}
        for key, value in content.items():
            if key not in handled_keys and key not in ['timestamp', 'message_id']:
                if isinstance(value, dict):
                    parts.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        parts.append(f"    {sub_key}: {sub_value}")
                else:
                    parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_system(content: Dict[str, Any]) -> str:
        """Format system message"""
        parts = []
        
        if content.get('system_message'):
            parts.append(f"  🔧 System Message: {content['system_message']}")
        
        if content.get('action_required'):
            parts.append(f"  ⚠️ Action Required: {content['action_required']}")
        
        if content.get('announcement'):
            parts.append(f"  📢 System Announcement: {content['announcement']}")
        
        # Handle nested content
        if content.get('content'):
            if isinstance(content['content'], dict):
                for key, value in content['content'].items():
                    parts.append(f"  {key}: {value}")
            else:
                parts.append(f"  Details: {content['content']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_error(content: Dict[str, Any]) -> str:
        """Format error message"""
        parts = []
        
        if content.get('error_message'):
            parts.append(f"  ❌ Error: {content['error_message']}")
        
        if content.get('error_type'):
            parts.append(f"  Error Type: {content['error_type']}")
        
        if content.get('stack_trace'):
            # 不再截断堆栈跟踪信息，显示完整内容
            parts.append(f"  Stack Trace: {content['stack_trace']}")
        
        if content.get('suggested_action'):
            parts.append(f"  Suggested Action: {content['suggested_action']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _format_generic(content: Dict[str, Any]) -> str:
        """Format generic message"""
        parts = []
        
        for key, value in content.items():
            if key in ['timestamp', 'message_id']:  # Skip metadata
                continue
            
            # 不再截断内容，显示完整信息
            if isinstance(value, (dict, list)):
                # 对于字典和列表，使用更好的格式化
                if isinstance(value, dict):
                    # 字典格式化为多行显示
                    parts.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        parts.append(f"    {sub_key}: {sub_value}")
                else:
                    # 列表格式化
                    parts.append(f"  {key}: {str(value)}")
            else:
                parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)


# Global message router instance (singleton)
_global_message_router = None
_router_lock = threading.Lock()

def get_global_message_router(workspace_root: str = None) -> MessageRouter:
    """Get or create global MessageRouter instance"""
    global _global_message_router
    
    if _global_message_router is None:
        if workspace_root is None:
            return None
        _global_message_router = MessageRouter(workspace_root)
    return _global_message_router


def format_inbox_for_llm_context(agent_id: str, workspace_root: str = None, 
                                output_directory: str = None, mark_as_read: bool = True) -> str:
    """
    Format agent inbox messages for LLM context
    
    Args:
        agent_id: Agent ID
        workspace_root: Workspace root directory  
        output_directory: Output directory (alternative to workspace_root)
        mark_as_read: Whether to mark messages as read
        
    Returns:
        Formatted message string for LLM
    """
    try:
        # 确定workspace_root
        if workspace_root is None:
            if output_directory:
                workspace_root = output_directory
            else:
                workspace_root = os.getcwd()
        
        router = get_global_message_router(workspace_root)
        mailbox = router.get_mailbox(agent_id)
        
        if not mailbox:
            return "📭 No mailbox found for this agent."
        
        # Get unread messages
        unread_messages = mailbox.get_unread_messages()
        
        if not unread_messages:
            return "📭 No new messages in inbox."
        
        # Format messages using MessageFormatter
        formatted_text = MessageFormatter.format_messages_for_llm(unread_messages)
        
        # Mark messages as read if requested
        if mark_as_read:
            for message in unread_messages:
                mailbox.mark_as_read(message.message_id)
        
        return formatted_text
        
    except Exception as e:
        return f"❌ Error accessing inbox: {str(e)}"


class MessageSystem:
    """Message System - facade for message routing functionality"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.router = get_global_message_router(self.workspace_root)
    
    def send_message(self, sender_id: str, receiver_id: str, message_type: MessageType, 
                    content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send a message through the system"""
        return self.router.send_message(sender_id, receiver_id, message_type, content, priority)
    
    def get_mailbox(self, agent_id: str) -> Optional['Mailbox']:
        """Get mailbox for an agent"""
        return self.router.get_mailbox(agent_id)
    
    def create_agent_mailbox(self, agent_id: str) -> 'Mailbox':
        """Create mailbox for an agent"""
        return self.router.create_agent_mailbox(agent_id)
    
    def get_agent_messages(self, agent_id: str, mark_as_read: bool = False) -> str:
        """Get formatted messages for an agent"""
        return get_agent_inbox_content(agent_id, self.workspace_root, mark_as_read)


# Backward compatibility alias
def get_message_router(workspace_root: str = None, output_directory: str = None, cleanup_on_init: bool = True) -> MessageRouter:
    """
    Backward compatibility function for get_message_router.
    
    Args:
        workspace_root: Workspace root directory
        output_directory: Output directory (alternative to workspace_root) 
        cleanup_on_init: Whether to cleanup old mailboxes on initialization (ignored)
        
    Returns:
        MessageRouter instance
    """
    if workspace_root is None and output_directory is not None:
        workspace_root = output_directory
    return get_global_message_router(workspace_root) 