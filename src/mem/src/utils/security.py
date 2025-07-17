"""
Security Utility Module
"""

import os
import base64
import hashlib
import secrets
from typing import Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityManager:
    """Security manager"""

    def __init__(self, key_file: str = ".secret_key"):
        """
        Initialize security manager

        Args:
            key_file: Key file path
        """
        self.key_file = key_file
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self) -> bytes:
        """Load or generate key"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_text(self, text: str) -> str:
        """
        Encrypt text

        Args:
            text: Text to encrypt

        Returns:
            Encrypted text
        """
        encrypted = self.cipher.encrypt(text.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt text

        Args:
            encrypted_text: Encrypted text

        Returns:
            Decrypted text
        """
        encrypted = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode('utf-8')

    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash password

        Args:
            password: Password
            salt: Salt value (optional)

        Returns:
            (Hash value, salt value)
        """
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode('utf-8'),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        return key.decode('utf-8'), salt

    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """
        Verify password

        Args:
            password: Password
            hashed: Hash value
            salt: Salt value

        Returns:
            Whether it matches
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode('utf-8'),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(
                kdf.derive(password.encode('utf-8')))
            return key.decode('utf-8') == hashed
        except Exception:
            return False

    def generate_token(self, length: int = 32) -> str:
        """
        Generate security token

        Args:
            length: Token length

        Returns:
            Security token
        """
        return secrets.token_urlsafe(length)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename

        Args:
            filename: Original filename

        Returns:
            Safe filename
        """
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        return filename


class ConfigSecurity:
    """Configuration security management"""

    def __init__(self, security_manager: SecurityManager):
        """
        Initialize configuration security management

        Args:
            security_manager: Security manager instance
        """
        self.security_manager = security_manager

    def encrypt_config_value(self, value: str) -> str:
        """
        Encrypt configuration value

        Args:
            value: Configuration value

        Returns:
            Encrypted value
        """
        return self.security_manager.encrypt_text(value)

    def decrypt_config_value(self, encrypted_value: str) -> str:
        """
        Decrypt configuration value

        Args:
            encrypted_value: Encrypted configuration value

        Returns:
            Decrypted value
        """
        return self.security_manager.decrypt_text(encrypted_value)

    def mask_sensitive_data(self, data: str, mask_char: str = '*') -> str:
        """
        Mask sensitive data

        Args:
            data: Sensitive data
            mask_char: Mask character

        Returns:
            Masked data
        """
        if len(data) <= 4:
            return mask_char * len(data)

        return data[:2] + mask_char * (len(data) - 4) + data[-2:]

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format

        Args:
            api_key: API key

        Returns:
            Whether it is valid
        """
        if not api_key:
            return False

        # Check if it starts with sk-
        if not api_key.startswith('sk-'):
            return False

        # Check length
        if len(api_key) < 20:
            return False

        return True


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """
    Get security manager instance

    Returns:
        Security manager instance
    """
    global _security_manager

    if _security_manager is None:
        _security_manager = SecurityManager()

    return _security_manager


def get_config_security() -> ConfigSecurity:
    """
    Get configuration security management instance

    Returns:
        Configuration security management instance
    """
    return ConfigSecurity(get_security_manager())
