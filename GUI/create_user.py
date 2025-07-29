#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Bot GUI User Management Script
Simple command-line tool for creating and managing GUI users
"""

import os
import sys
import argparse
import getpass
from datetime import datetime, timedelta

# Add current directory to path (we are now in GUI folder)
sys.path.append(os.path.dirname(__file__))
from auth_manager import AuthenticationManager


def create_user_interactive():
    """Interactive user creation"""
    print("=== AGI Bot GUI - 用户创建向导 ===\n")
    
    # Initialize auth manager (find config dir relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from GUI/ to project root
    config_dir = os.path.join(project_root, 'config')
    auth_manager = AuthenticationManager(config_dir)
    
    # Get user input
    print("请输入用户信息：")
    username = input("用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        return False
    
    # Check if user already exists
    existing_users = auth_manager.list_authorized_keys()
    for user in existing_users:
        if user['name'] == username:
            print(f"❌ 用户 '{username}' 已存在")
            return False
    
    # Get API key
    print("\nAPI Key输入方式:")
    print("1. 输入自定义API Key")
    print("2. 自动生成安全API Key")
    
    choice = input("选择 (1/2): ").strip()
    
    if choice == "1":
        api_key = getpass.getpass("请输入API Key (不会显示): ").strip()
        if not api_key:
            print("❌ API Key不能为空")
            return False
    elif choice == "2":
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        api_key = ''.join(secrets.choice(alphabet) for _ in range(32))
        print(f"✅ 自动生成的API Key: {api_key}")
        print("⚠️  请务必保存此API Key，系统不会明文存储！")
        input("按Enter键继续...")
    else:
        print("❌ 无效选择")
        return False
    
    # Get description
    description = input("用户描述 (可选): ").strip() or f"{username} user"
    
    # Get permissions
    print("\n权限设置:")
    print("可用权限: read, write, execute, admin")
    print("默认权限: read, write, execute")
    
    permissions_input = input("权限列表 (用空格分隔，回车使用默认): ").strip()
    if permissions_input:
        permissions = permissions_input.split()
        # Validate permissions
        valid_permissions = {"read", "write", "execute", "admin"}
        invalid_perms = set(permissions) - valid_permissions
        if invalid_perms:
            print(f"❌ 无效权限: {', '.join(invalid_perms)}")
            return False
    else:
        permissions = ["read", "write", "execute"]
    
    # Get expiration (optional)
    print("\n过期时间设置:")
    print("1. 永不过期")
    print("2. 设置过期时间")
    
    expire_choice = input("选择 (1/2): ").strip()
    expires_at = None
    
    if expire_choice == "2":
        try:
            days = int(input("多少天后过期: ").strip())
            if days <= 0:
                print("❌ 天数必须大于0")
                return False
            expires_at = (datetime.now() + timedelta(days=days)).isoformat()
        except ValueError:
            print("❌ 请输入有效的天数")
            return False
    
    # Confirm creation
    print(f"\n=== 用户信息确认 ===")
    print(f"用户名: {username}")
    print(f"描述: {description}")
    print(f"权限: {', '.join(permissions)}")
    print(f"过期时间: {'永不过期' if not expires_at else expires_at}")
    print(f"API Key: {'已设置' if api_key else '未设置'}")
    
    confirm = input("\n确认创建用户? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ 用户创建已取消")
        return False
    
    # Create user
    success = auth_manager.add_authorized_key(
        name=username,
        api_key=api_key,
        description=description,
        permissions=permissions,
        expires_at=expires_at
    )
    
    if success:
        print(f"\n✅ 用户 '{username}' 创建成功！")
        print(f"📁 配置文件: {auth_manager.authorized_keys_file}")
        
        # Show API key again if auto-generated
        if choice == "2":
            print(f"\n🔑 API Key: {api_key}")
            print("⚠️  请妥善保存API Key，系统仅存储哈希值！")
        
        return True
    else:
        print(f"❌ 用户创建失败")
        return False


def create_user_command(username, api_key, description=None, permissions=None, expires_days=None):
    """Command-line user creation"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from GUI/ to project root
    config_dir = os.path.join(project_root, 'config')
    auth_manager = AuthenticationManager(config_dir)
    
    # Set defaults
    if not description:
        description = f"{username} user"
    if not permissions:
        permissions = ["read", "write", "execute"]
    
    # Calculate expiration
    expires_at = None
    if expires_days:
        expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
    
    # Create user
    success = auth_manager.add_authorized_key(
        name=username,
        api_key=api_key,
        description=description,
        permissions=permissions,
        expires_at=expires_at
    )
    
    if success:
        print(f"✅ 用户 '{username}' 创建成功")
        return True
    else:
        print(f"❌ 用户 '{username}' 创建失败")
        return False


def list_users():
    """List all users"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from GUI/ to project root
    config_dir = os.path.join(project_root, 'config')
    auth_manager = AuthenticationManager(config_dir)
    users = auth_manager.list_authorized_keys()
    
    if not users:
        print("📋 当前没有授权用户")
        return
    
    print("\n=== 授权用户列表 ===")
    print("-" * 80)
    for user in users:
        status = "✅ 启用" if user["enabled"] else "❌ 禁用"
        expire_info = "永不过期" if not user["expires_at"] else f"过期: {user['expires_at'][:10]}"
        
        print(f"{status} {user['name']:<15} | {user['description']:<30}")
        print(f"   权限: {', '.join(user['permissions']):<20} | {expire_info}")
        print(f"   哈希: {user['hash_preview']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AGI Bot GUI 用户管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式创建用户
  python create_user.py

  # 命令行创建用户
  python create_user.py -u alice -k alice123 -d "Alice用户"

  # 创建管理员用户
  python create_user.py -u admin2 -k admin456 -p read write execute admin

  # 创建临时用户（30天后过期）
  python create_user.py -u temp -k temp123 -e 30

  # 列出所有用户
  python create_user.py --list
        """
    )
    
    parser.add_argument('-u', '--username', help='用户名')
    parser.add_argument('-k', '--api-key', help='API Key')
    parser.add_argument('-d', '--description', help='用户描述')
    parser.add_argument('-p', '--permissions', nargs='+', 
                       choices=['read', 'write', 'execute', 'admin'],
                       help='权限列表')
    parser.add_argument('-e', '--expires', type=int, metavar='DAYS',
                       help='过期天数')
    parser.add_argument('--list', action='store_true', help='列出所有用户')
    
    args = parser.parse_args()
    
    # List users
    if args.list:
        list_users()
        return
    
    # Command-line mode
    if args.username and args.api_key:
        success = create_user_command(
            username=args.username,
            api_key=args.api_key,
            description=args.description,
            permissions=args.permissions,
            expires_days=args.expires
        )
        sys.exit(0 if success else 1)
    
    # Partial arguments provided
    elif args.username or args.api_key:
        print("❌ 命令行模式需要同时提供用户名和API Key")
        print("使用 'python create_user.py -h' 查看帮助")
        sys.exit(1)
    
    # Interactive mode
    else:
        try:
            success = create_user_interactive()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n\n❌ 用户创建已取消")
            sys.exit(1)


if __name__ == "__main__":
    main()