#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Authentication Manager for AGI Agent
Provides secure API key authentication with SHA-256 hashing
"""

import os
import hashlib
import json
import logging
import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set

class AuthenticationManager:
    """Secure authentication manager for GUI access"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.authorized_keys_file = os.path.join(config_dir, "authorized_keys.json")
        self.session_timeout = timedelta(hours=24)  # Session timeout
        self.active_sessions: Dict[str, Dict] = {}  # session_id -> session_info
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Initialize logger first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize authorized keys file if not exists
        self._init_authorized_keys()
    
    def _init_authorized_keys(self):
        """Initialize authorized keys file with default structure"""
        if not os.path.exists(self.authorized_keys_file):
            default_config = {
                "description": "Authorized API keys for AGI Agent GUI access",
                "version": "1.0",
                "keys": [
                    {
                        "name": "admin",
                        "description": "Default admin key",
                        "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Empty string hash
                        "permissions": ["read", "write", "execute", "admin"],
                        "created_at": datetime.now().isoformat(),
                        "expires_at": None,
                        "enabled": False
                    }
                ]
            }
            self._save_authorized_keys(default_config)
            self.logger.info(f"Created default authorized keys file: {self.authorized_keys_file}")
    
    def _load_authorized_keys(self) -> Dict:
        """Load authorized keys from file"""
        try:
            with open(self.authorized_keys_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load authorized keys: {e}")
            return {"keys": []}
    
    def _save_authorized_keys(self, config: Dict):
        """Save authorized keys to file"""
        try:
            with open(self.authorized_keys_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save authorized keys: {e}")
            raise
    
    def _hash_api_key(self, api_key: str) -> str:
        """Generate SHA-256 hash of API key"""
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()

    def _generate_api_key(self, length: int = 32) -> str:
        """Generate a secure random API key"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def _generate_deterministic_api_key(self, username: str, length: int = 32) -> str:
        """Generate a deterministic API key based on username"""
        # Use username + salt to generate deterministic key
        salt = "AGI_Agent_2025"  # Fixed salt for consistency
        combined = f"{username}_{salt}"
        
        # Generate hash and convert to alphanumeric string
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to alphanumeric characters
        alphabet = string.ascii_letters + string.digits
        api_key = ""
        for i in range(0, len(hash_hex), 2):
            if len(api_key) >= length:
                break
            # Convert hex pair to integer and map to alphabet
            hex_pair = hash_hex[i:i+2]
            num = int(hex_pair, 16)
            api_key += alphabet[num % len(alphabet)]
        
        return api_key[:length]

    def _validate_phone_number(self, phone: str) -> bool:
        """
        Validate phone number format
        Supports various international formats with more flexible rules
        """
        # Remove all whitespace and common separators
        phone = re.sub(r'[\s\-\(\)\+]', '', phone)

        # More flexible phone number validation
        # Allow 7-20 digits, optionally starting with country code
        phone_pattern = r'^\d{7,20}$'
        
        # Also allow international format with country code
        international_pattern = r'^\d{1,4}\d{6,14}$'

        return bool(re.match(phone_pattern, phone) or re.match(international_pattern, phone))
    
    def authenticate_api_key(self, api_key: Optional[str]) -> Dict[str, any]:
        """
        Authenticate API key against authorized keys
        
        Returns:
            Dict with authentication result:
            {
                "authenticated": bool,
                "user_info": dict or None,
                "error": str or None,
                "is_guest": bool
            }
        """
        # Handle guest access when no API key is provided
        if api_key is None:
            self.logger.info("Guest user access granted")
            return {
                "authenticated": True,
                "user_info": {
                    "name": "guest",
                    "description": "Guest user with limited access",
                    "permissions": ["read", "write", "execute"],  # Guest has basic permissions
                    "authenticated_at": datetime.now().isoformat(),
                    "is_guest": True
                },
                "error": None,
                "is_guest": True
            }
        
        # Hash the provided API key
        api_key_hash = self._hash_api_key(api_key)
        
        # Load authorized keys (fresh read each time for immediate updates)
        auth_config = self._load_authorized_keys()
        self.logger.debug(f"Loaded {len(auth_config.get('keys', []))} authorized keys for authentication")
        
        # Check against authorized keys
        for key_info in auth_config.get("keys", []):
            if not key_info.get("enabled", False):
                continue
                
            if key_info.get("sha256_hash") == api_key_hash:
                # Check expiration
                expires_at = key_info.get("expires_at")
                if expires_at:
                    try:
                        expire_date = datetime.fromisoformat(expires_at)
                        if datetime.now() > expire_date:
                            return {
                                "authenticated": False,
                                "user_info": None,
                                "error": "API key has expired"
                            }
                    except ValueError:
                        pass  # Invalid date format, treat as no expiration
                
                # Authentication successful
                user_info = {
                    "name": key_info.get("name", "unknown"),
                    "permissions": key_info.get("permissions", ["read"]),
                    "authenticated_at": datetime.now().isoformat(),
                    "is_guest": False
                }
                
                self.logger.info(f"Successful authentication for user: {user_info['name']}")
                return {
                    "authenticated": True,
                    "user_info": user_info,
                    "error": None,
                    "is_guest": False
                }
        
        self.logger.warning(f"Failed authentication attempt with hash: {api_key_hash[:16]}...")
        return {
            "authenticated": False,
            "user_info": None,
            "error": "Invalid API key",
            "is_guest": False
        }
    
    def create_session(self, api_key: Optional[str], session_id: str) -> bool:
        """Create authenticated session"""
        auth_result = self.authenticate_api_key(api_key)
        
        if auth_result["authenticated"]:
            # For guest users, use "guest" as the hash
            api_key_hash = "guest" if auth_result.get("is_guest", False) else self._hash_api_key(api_key)
            
            self.active_sessions[session_id] = {
                "api_key_hash": api_key_hash,
                "user_info": auth_result["user_info"],  
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
            return True
        
        return False
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and return user info if valid"""
        if session_id not in self.active_sessions:
            return None
        
        session_info = self.active_sessions[session_id]
        
        # Check session timeout
        if datetime.now() - session_info["last_accessed"] > self.session_timeout:
            del self.active_sessions[session_id]
            return None
        
        # Update last accessed time
        session_info["last_accessed"] = datetime.now()
        
        return session_info["user_info"]
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def register_user(self, username: str, phone_number: str, permissions: List[str] = None) -> Dict[str, any]:
        """
        Register a new user with username and phone number

        Returns:
            Dict with registration result:
            {
                "success": bool,
                "api_key": str or None,
                "user_info": dict or None,
                "error": str or None
            }
        """
        if permissions is None:
            permissions = ["read", "write", "execute"]

        # Validate inputs
        if not username or not username.strip():
            return {
                "success": False,
                "api_key": None,
                "user_info": None,
                "error": "用户名不能为空"
            }

        if not phone_number or not phone_number.strip():
            return {
                "success": False,
                "api_key": None,
                "user_info": None,
                "error": "手机号不能为空"
            }

        # Validate phone number format
        if not self._validate_phone_number(phone_number):
            return {
                "success": False,
                "api_key": None,
                "user_info": None,
                "error": "手机号格式不正确"
            }

        # Check if username already exists
        auth_config = self._load_authorized_keys()
        for key_info in auth_config.get("keys", []):
            if key_info.get("name") == username.strip():
                # User already exists, generate deterministic API key for display (don't update file)
                api_key = self._generate_deterministic_api_key(username.strip())
                return {
                    "success": True,
                    "api_key": api_key,  # Return the deterministic API key
                    "user_info": {
                        "name": key_info.get("name"),
                        "phone_number": key_info.get("phone_number", ""),
                        "permissions": key_info.get("permissions", []),
                        "created_at": key_info.get("created_at"),
                        "is_guest": False,
                        "existing_user": True  # 标记为已存在的用户
                    },
                    "error": None
                }

        # Check if phone number is already registered (compare last 4 digits)
        phone_last4 = phone_number.strip()[-4:] if len(phone_number.strip()) >= 4 else phone_number.strip()
        for key_info in auth_config.get("keys", []):
            stored_phone = key_info.get("phone_number", "")
            if stored_phone == phone_last4:
                # Phone number already exists, generate deterministic API key for display (don't update file)
                api_key = self._generate_deterministic_api_key(key_info.get("name", ""))
                return {
                    "success": True,
                    "api_key": api_key,  # Return the deterministic API key
                    "user_info": {
                        "name": key_info.get("name"),
                        "phone_number": key_info.get("phone_number", ""),
                        "permissions": key_info.get("permissions", []),
                        "created_at": key_info.get("created_at"),
                        "is_guest": False,
                        "existing_user": True  # 标记为已存在的用户
                    },
                    "error": None
                }

        # Generate API key
        api_key = self._generate_deterministic_api_key(username.strip())

        # Create user record (store only last 4 digits of phone number)
        phone_last4 = phone_number.strip()[-4:] if len(phone_number.strip()) >= 4 else phone_number.strip()
        new_user = {
            "name": username.strip(),
            "phone_number": phone_last4,
            "sha256_hash": self._hash_api_key(api_key),
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": None,
            "enabled": True
        }

        auth_config["keys"].append(new_user)

        try:
            self._save_authorized_keys(auth_config)
            self.logger.info(f"Registered new user: {username} with phone ending: {phone_last4}")

            user_info = {
                "name": new_user["name"],
                "phone_number": new_user["phone_number"],
                "permissions": new_user["permissions"],
                "created_at": new_user["created_at"],
                "is_guest": False
            }

            return {
                "success": True,
                "api_key": api_key,
                "user_info": user_info,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Failed to register user: {e}")
            return {
                "success": False,
                "api_key": None,
                "user_info": None,
                "error": f"注册失败: {str(e)}"
            }

    def add_authorized_key(self, name: str, api_key: str, permissions: List[str] = None, expires_at: str = None) -> bool:
        """Add new authorized key"""
        if permissions is None:
            permissions = ["read", "write", "execute"]

        # Always load fresh data
        auth_config = self._load_authorized_keys()

        # Check if name already exists
        for key_info in auth_config.get("keys", []):
            if key_info.get("name") == name:
                self.logger.error(f"Key with name '{name}' already exists")
                return False

        # Add new key
        new_key = {
            "name": name,
            "sha256_hash": self._hash_api_key(api_key),
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
            "enabled": True
        }

        auth_config["keys"].append(new_key)

        try:
            self._save_authorized_keys(auth_config)
            self.logger.info(f"Added authorized key for user: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add authorized key: {e}")
            return False
    
    def remove_authorized_key(self, name: str) -> bool:
        """Remove authorized key by name"""
        # Always load fresh data
        auth_config = self._load_authorized_keys()
        
        # Find and remove key
        original_count = len(auth_config.get("keys", []))
        auth_config["keys"] = [k for k in auth_config.get("keys", []) if k.get("name") != name]
        
        if len(auth_config["keys"]) < original_count:
            try:
                self._save_authorized_keys(auth_config)
                self.logger.info(f"Removed authorized key for user: {name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to remove authorized key: {e}")
                return False
        
        self.logger.warning(f"Key with name '{name}' not found")
        return False
    
    def list_authorized_keys(self) -> List[Dict]:
        """List all authorized keys (without sensitive data)"""
        # Always load fresh data
        auth_config = self._load_authorized_keys()

        result = []
        for key_info in auth_config.get("keys", []):
            safe_info = {
                "name": key_info.get("name"),
                "phone_number": key_info.get("phone_number", ""),
                "permissions": key_info.get("permissions"),
                "created_at": key_info.get("created_at"),
                "expires_at": key_info.get("expires_at"),
                "enabled": key_info.get("enabled"),
                "hash_preview": key_info.get("sha256_hash", "")[:16] + "..." if key_info.get("sha256_hash") else ""
            }
            result.append(safe_info)

        return result
    
    def enable_key(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a key"""
        # Always load fresh data
        auth_config = self._load_authorized_keys()
        
        for key_info in auth_config.get("keys", []):
            if key_info.get("name") == name:
                key_info["enabled"] = enabled
                try:
                    self._save_authorized_keys(auth_config)
                    self.logger.info(f"{'Enabled' if enabled else 'Disabled'} key for user: {name}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to update key status: {e}")
                    return False
        
        self.logger.warning(f"Key with name '{name}' not found")
        return False


# CLI management functions
def main():
    """CLI interface for managing authorized keys"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage AGI Agent GUI authorized keys")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add key command
    add_parser = subparsers.add_parser("add", help="Add new authorized key")
    add_parser.add_argument("name", help="Key name/identifier")
    add_parser.add_argument("api_key", help="API key")
    add_parser.add_argument("--permissions", nargs="+", default=["read", "write", "execute"], 
                           help="Permissions list")
    add_parser.add_argument("--expires", help="Expiration date (ISO format)")
    
    # Remove key command
    remove_parser = subparsers.add_parser("remove", help="Remove authorized key")
    remove_parser.add_argument("name", help="Key name to remove")
    
    # List keys command
    list_parser = subparsers.add_parser("list", help="List authorized keys")
    
    # Enable/disable key command
    enable_parser = subparsers.add_parser("enable", help="Enable authorized key")
    enable_parser.add_argument("name", help="Key name")
    
    disable_parser = subparsers.add_parser("disable", help="Disable authorized key")
    disable_parser.add_argument("name", help="Key name")
    
    # Hash command for testing
    hash_parser = subparsers.add_parser("hash", help="Generate SHA-256 hash of API key")
    hash_parser.add_argument("api_key", help="API key to hash")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    auth_manager = AuthenticationManager(args.config_dir)
    
    if args.command == "add":
        success = auth_manager.add_authorized_key(
            args.name, args.api_key, args.permissions, args.expires
        )
        print(f"Key addition {'successful' if success else 'failed'}")
    
    elif args.command == "remove":
        success = auth_manager.remove_authorized_key(args.name)
        print(f"Key removal {'successful' if success else 'failed'}")
    
    elif args.command == "list":
        keys = auth_manager.list_authorized_keys()
        print("\nAuthorized Keys:")
        print("-" * 80)
        for key in keys:
            status = "✓" if key["enabled"] else "✗"
            print(f"{status} {key['name']:15} | {key['hash_preview']}")
            print(f"   Permissions: {', '.join(key['permissions'])}")
            if key['expires_at']:
                print(f"   Expires: {key['expires_at']}")
            print()
    
    elif args.command == "enable":
        success = auth_manager.enable_key(args.name, True)
        print(f"Key enable {'successful' if success else 'failed'}")
    
    elif args.command == "disable":
        success = auth_manager.enable_key(args.name, False)
        print(f"Key disable {'successful' if success else 'failed'}")
    
    elif args.command == "hash":
        hash_value = hashlib.sha256(args.api_key.encode('utf-8')).hexdigest()
        print(f"SHA-256 hash: {hash_value}")


if __name__ == "__main__":
    main()