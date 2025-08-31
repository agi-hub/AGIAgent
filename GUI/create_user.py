#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Agent GUI User Management Script
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
    print("=== AGI Agent GUI - User Creation Wizard ===\n")
    
    # Initialize auth manager (find config dir relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from GUI/ to project root
    config_dir = os.path.join(project_root, 'config')
    auth_manager = AuthenticationManager(config_dir)
    
    # Get user input
    print("Please enter user information:")
    username = input("Username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return False
    
    # Check if user already exists
    existing_users = auth_manager.list_authorized_keys()
    for user in existing_users:
        if user['name'] == username:
            print(f"❌ User '{username}' Already exists")
            return False
    
    # Get API key
    print("\nAPI Key input method:")
    print("1. Enter custom API Key")
    print("2. Auto-generate secure API Key")
    
    choice = input("Select (1/2): ").strip()
    
    if choice == "1":
        api_key = getpass.getpass("Please enter API Key (Won't display): ").strip()
        if not api_key:
            print("❌ API Key cannot be empty")
            return False
    elif choice == "2":
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        api_key = ''.join(secrets.choice(alphabet) for _ in range(32))
        print(f"✅ Auto-generated API Key: {api_key}")
        print("⚠️  Please be sure to save this API Key")
        input("Press Enter to continue...")
    else:
        print("❌ Invalid selection")
        return False
    
    # Get description
    description = input("User description (Optional): ").strip() or f"{username} user"
    
    # Get permissions
    print("\nPermission settings:")
    print("Available permissions: read, write, execute, admin")
    print("Default permissions: read, write, execute")
    
    permissions_input = input("Permission list (Separate with spaces): ").strip()
    if permissions_input:
        permissions = permissions_input.split()
        # Validate permissions
        valid_permissions = {"read", "write", "execute", "admin"}
        invalid_perms = set(permissions) - valid_permissions
        if invalid_perms:
            print(f"❌ Invalid permission: {', '.join(invalid_perms)}")
            return False
    else:
        permissions = ["read", "write", "execute"]
    
    # Get expiration (optional)
    print("\nExpiration time settings:")
    print("1. Never expires")
    print("2. Set expiration time")
    
    expire_choice = input("Select (1/2): ").strip()
    expires_at = None
    
    if expire_choice == "2":
        try:
            days = int(input("How many days until expiration: ").strip())
            if days <= 0:
                print("❌ Number of days must be greater than 0")
                return False
            expires_at = (datetime.now() + timedelta(days=days)).isoformat()
        except ValueError:
            print("❌ Please enter a valid number of days")
            return False
    
    # Confirm creation
    print(f"\n=== User Information Confirmation ===")
    print(f"Username: {username}")
    print(f"Description: {description}")
    print(f"Permissions: {', '.join(permissions)}")
    print(f"Expiration time: {'Never expires' if not expires_at else expires_at}")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    
    confirm = input("\nConfirm create user? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ User creation cancelled")
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
        print(f"\n✅ User '{username}' Created successfully!")
        print(f"📁 Configuration file: {auth_manager.authorized_keys_file}")
        
        # Show API key again if auto-generated
        if choice == "2":
            print(f"\n🔑 API Key: {api_key}")
            print("⚠️  Please save the API Key properly")
        
        return True
    else:
        print(f"❌ User creation failed")
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
        print(f"✅ User '{username}' Created successfully")
        return True
    else:
        print(f"❌ User '{username}' Creation failed")
        return False


def list_users():
    """List all users"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from GUI/ to project root
    config_dir = os.path.join(project_root, 'config')
    auth_manager = AuthenticationManager(config_dir)
    users = auth_manager.list_authorized_keys()
    
    if not users:
        print("📋 No authorized users currently")
        return
    
    print("\n=== Authorized User List ===")
    print("-" * 80)
    for user in users:
        status = "✅ Enabled" if user["enabled"] else "❌ Disabled"
        expire_info = "Never expires" if not user["expires_at"] else f"Expired: {user['expires_at'][:10]}"
        
        print(f"{status} {user['name']:<15} | {user['description']:<30}")
        print(f"   Permissions: {', '.join(user['permissions']):<20} | {expire_info}")
        print(f"   Hash: {user['hash_preview']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AGI Agent GUI User Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Interactive user creation
  python create_user.py

  # Command line user creation
  python create_user.py -u alice -k alice123 -d "Alice user"

  # Create administrator user
  python create_user.py -u admin2 -k admin456 -p read write execute admin

  # Create temporary user (expires in 30 days)
  python create_user.py -u temp -k temp123 -e 30

  # List all users
  python create_user.py --list
        """
    )
    
    parser.add_argument('-u', '--username', help='Username')
    parser.add_argument('-k', '--api-key', help='API Key')
    parser.add_argument('-d', '--description', help='User description')
    parser.add_argument('-p', '--permissions', nargs='+', 
                       choices=['read', 'write', 'execute', 'admin'],
                       help='Permission list')
    parser.add_argument('-e', '--expires', type=int, metavar='DAYS',
                       help='Expiration days')
    parser.add_argument('--list', action='store_true', help='List all users')
    
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
        print("❌ Command line mode requires both username and API Key")
        print("Use 'python create_user.py -h' View help")
        sys.exit(1)
    
    # Interactive mode
    else:
        try:
            success = create_user_interactive()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n\n❌ User creation cancelled")
            sys.exit(1)


if __name__ == "__main__":
    main()