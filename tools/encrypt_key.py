#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGIAgent 密钥加密工具

用法:
  python tools/encrypt_key.py                # 交互式输入
  python tools/encrypt_key.py <your_api_key> # 命令行参数

将加密后的字符串填入 config/config.txt：
  api_key=agi-xxxxxxxxxxxxxxxxxxxxxxxx
"""

import sys
import os

# 将 src 目录加入路径以导入 key_crypto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from key_crypto import encrypt_key, decrypt_key


def main():
    print("=" * 50)
    print("  AGIAgent 密钥加密工具")
    print("=" * 50)

    if len(sys.argv) > 1:
        plaintext = sys.argv[1]
    else:
        print("\n请输入需要加密的密钥（输入后按回车）：")
        try:
            plaintext = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(0)

    if not plaintext:
        print("错误：密钥不能为空。")
        sys.exit(1)

    encrypted = encrypt_key(plaintext)

    print("\n加密成功！请将以下内容填入 config/config.txt 的 api_key= 处：")
    print()
    print(f"  {encrypted}")
    print()
    print("示例 config.txt 写法：")
    print(f"  api_key={encrypted}")
    print()

    # 自我验证：确认解密结果与原始密钥一致
    try:
        verified = decrypt_key(encrypted)
        if verified == plaintext:
            print("[验证通过] 加解密正确，可以放心使用。")
        else:
            print("[警告] 验证失败：解密结果与原始密钥不符！")
            sys.exit(1)
    except Exception as e:
        print(f"[警告] 验证时出错：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
