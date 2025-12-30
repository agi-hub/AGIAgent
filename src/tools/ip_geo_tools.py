#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .print_system import print_system, print_current, print_error
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
"""

import requests
import socket
from typing import Dict, Any, Optional, Tuple
import json


class IPGeoTools:
    """IP地理位置工具类，用于判断IP地址是否在中国"""
    
    def __init__(self):
        """初始化IP地理位置工具"""
        self.timeout = 5  # API请求超时时间（秒）
        self.cache = {}  # 简单的内存缓存，避免重复查询
    
    def _get_public_ip(self) -> Optional[str]:
        """
        获取本机公网IP地址
        
        Returns:
            str: 公网IP地址，失败返回None
        """
        try:
            # 方法1: 使用ipify.org
            response = requests.get('https://api.ipify.org?format=json', timeout=self.timeout)
            if response.status_code == 200:
                return response.json().get('ip')
        except Exception as e:
            print_current(f"⚠️ 获取公网IP失败 (ipify): {e}")
        
        try:
            # 方法2: 使用httpbin.org
            response = requests.get('https://httpbin.org/ip', timeout=self.timeout)
            if response.status_code == 200:
                return response.json().get('origin')
        except Exception as e:
            print_current(f"⚠️ 获取公网IP失败 (httpbin): {e}")
        
        return None
    
    def _is_valid_ip(self, ip: str) -> bool:
        """
        验证IP地址格式是否有效
        
        Args:
            ip: IP地址字符串
            
        Returns:
            bool: 是否为有效的IP地址
        """
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    def check_ip_in_china_api(self, ip: Optional[str] = None) -> Dict[str, Any]:
        """
        通过免费API检查IP是否在中国（方法1：使用API服务）
        
        Args:
            ip: 要检查的IP地址，如果为None则检查本机公网IP
            
        Returns:
            dict: 包含检查结果的字典
                - is_in_china: bool, 是否在中国
                - country: str, 国家代码（如CN）
                - country_name: str, 国家名称
                - ip: str, 查询的IP地址
                - method: str, 使用的查询方法
                - error: str, 错误信息（如果有）
        """
        result = {
            'is_in_china': False,
            'country': None,
            'country_name': None,
            'ip': ip,
            'method': 'api',
            'error': None
        }
        
        # 如果没有提供IP，获取本机公网IP
        if ip is None:
            ip = self._get_public_ip()
            if ip is None:
                result['error'] = '无法获取公网IP地址'
                return result
            result['ip'] = ip
        
        # 验证IP格式
        if not self._is_valid_ip(ip):
            result['error'] = f'无效的IP地址格式: {ip}'
            return result
        
        # 检查缓存
        cache_key = f"api_{ip}"
        if cache_key in self.cache:
            print_current(f"📦 使用缓存结果: {ip}")
            return self.cache[cache_key]
        
        # 方法1: 使用ip-api.com（免费，无需API密钥）
        try:
            url = f'http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,query'
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    country_code = data.get('countryCode', '').upper()
                    country_name = data.get('country', '')
                    is_china = country_code == 'CN'
                    
                    result.update({
                        'is_in_china': is_china,
                        'country': country_code,
                        'country_name': country_name,
                        'error': None
                    })
                    
                    # 缓存结果
                    self.cache[cache_key] = result.copy()
                    return result
        except Exception as e:
            print_current(f"⚠️ ip-api.com查询失败: {e}")
        
        # 方法2: 使用ipapi.co（免费，有速率限制）
        try:
            url = f'https://ipapi.co/{ip}/json/'
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if 'error' not in data:
                    country_code = data.get('country_code', '').upper()
                    country_name = data.get('country_name', '')
                    is_china = country_code == 'CN'
                    
                    result.update({
                        'is_in_china': is_china,
                        'country': country_code,
                        'country_name': country_name,
                        'error': None
                    })
                    
                    # 缓存结果
                    self.cache[cache_key] = result.copy()
                    return result
        except Exception as e:
            print_current(f"⚠️ ipapi.co查询失败: {e}")
        
        # 方法3: 使用ip-api.io（备用）
        try:
            url = f'https://ip-api.io/json/{ip}'
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                country_code = data.get('country_code', '').upper()
                country_name = data.get('country_name', '')
                is_china = country_code == 'CN'
                
                result.update({
                    'is_in_china': is_china,
                    'country': country_code,
                    'country_name': country_name,
                    'error': None
                })
                
                # 缓存结果
                self.cache[cache_key] = result.copy()
                return result
        except Exception as e:
            print_current(f"⚠️ ip-api.io查询失败: {e}")
        
        result['error'] = '所有API查询方法均失败'
        return result
    
    def check_ip_in_china_local(self, ip: Optional[str] = None) -> Dict[str, Any]:
        """
        通过本地数据库检查IP是否在中国（方法2：使用本地GeoIP数据库）
        需要先安装geoip2库和下载数据库文件
        
        Args:
            ip: 要检查的IP地址，如果为None则检查本机公网IP
            
        Returns:
            dict: 包含检查结果的字典
        """
        result = {
            'is_in_china': False,
            'country': None,
            'country_name': None,
            'ip': ip,
            'method': 'local',
            'error': None
        }
        
        try:
            import geoip2.database
            import geoip2.errors
        except ImportError:
            result['error'] = 'geoip2库未安装，请运行: pip install geoip2'
            return result
        
        # 如果没有提供IP，获取本机公网IP
        if ip is None:
            ip = self._get_public_ip()
            if ip is None:
                result['error'] = '无法获取公网IP地址'
                return result
            result['ip'] = ip
        
        # 验证IP格式
        if not self._is_valid_ip(ip):
            result['error'] = f'无效的IP地址格式: {ip}'
            return result
        
        # 检查缓存
        cache_key = f"local_{ip}"
        if cache_key in self.cache:
            print_current(f"📦 使用缓存结果: {ip}")
            return self.cache[cache_key]
        
        # 查找数据库文件（常见位置）
        import os
        db_paths = [
            '/usr/share/GeoIP/GeoLite2-Country.mmdb',
            '/var/lib/GeoIP/GeoLite2-Country.mmdb',
            os.path.expanduser('~/GeoLite2-Country.mmdb'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'GeoLite2-Country.mmdb'),
        ]
        
        db_path = None
        for path in db_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if db_path is None:
            result['error'] = '未找到GeoIP数据库文件。请下载GeoLite2-Country.mmdb并放置在以下位置之一:\n' + '\n'.join(db_paths)
            return result
        
        try:
            with geoip2.database.Reader(db_path) as reader:
                response = reader.country(ip)
                country_code = response.country.iso_code
                country_name = response.country.name
                is_china = country_code == 'CN'
                
                result.update({
                    'is_in_china': is_china,
                    'country': country_code,
                    'country_name': country_name,
                    'error': None
                })
                
                # 缓存结果
                self.cache[cache_key] = result.copy()
                return result
        except geoip2.errors.AddressNotFoundError:
            result['error'] = f'IP地址 {ip} 未在数据库中找到'
        except Exception as e:
            result['error'] = f'查询数据库时出错: {str(e)}'
        
        return result
    
    def is_ip_in_china(self, ip: Optional[str] = None, use_local: bool = False) -> bool:
        """
        判断IP是否在中国的便捷方法
        
        Args:
            ip: 要检查的IP地址，如果为None则检查本机公网IP
            use_local: 是否使用本地数据库（需要安装geoip2和数据库文件）
            
        Returns:
            bool: 是否在中国，查询失败返回False
        """
        if use_local:
            result = self.check_ip_in_china_local(ip)
        else:
            result = self.check_ip_in_china_api(ip)
        
        if result.get('error'):
            print_error(f"❌ IP地理位置查询失败: {result['error']}")
            return False
        
        return result.get('is_in_china', False)
    
    def get_ip_info(self, ip: Optional[str] = None, use_local: bool = False) -> Dict[str, Any]:
        """
        获取IP的详细信息
        
        Args:
            ip: 要查询的IP地址，如果为None则查询本机公网IP
            use_local: 是否使用本地数据库
            
        Returns:
            dict: IP的详细信息
        """
        if use_local:
            return self.check_ip_in_china_local(ip)
        else:
            return self.check_ip_in_china_api(ip)


# 便捷函数
def is_ip_in_china(ip: Optional[str] = None, use_local: bool = False) -> bool:
    """
    便捷函数：判断IP是否在中国
    
    Args:
        ip: IP地址，None表示查询本机公网IP
        use_local: 是否使用本地数据库
        
    Returns:
        bool: 是否在中国
    """
    tools = IPGeoTools()
    return tools.is_ip_in_china(ip, use_local)


def get_ip_country(ip: Optional[str] = None, use_local: bool = False) -> Optional[str]:
    """
    便捷函数：获取IP所在国家代码
    
    Args:
        ip: IP地址，None表示查询本机公网IP
        use_local: 是否使用本地数据库
        
    Returns:
        str: 国家代码（如CN），失败返回None
    """
    tools = IPGeoTools()
    result = tools.get_ip_info(ip, use_local)
    return result.get('country')


if __name__ == '__main__':
    # 测试代码
    tools = IPGeoTools()
    
    print("=" * 50)
    print("IP地理位置查询测试")
    print("=" * 50)
    
    # 测试1: 查询本机IP
    print("\n1. 查询本机公网IP:")
    result = tools.check_ip_in_china_api()
    print(f"   IP: {result['ip']}")
    print(f"   国家: {result['country_name']} ({result['country']})")
    print(f"   是否在中国: {result['is_in_china']}")
    if result['error']:
        print(f"   错误: {result['error']}")
    
    # 测试2: 查询中国IP
    print("\n2. 查询中国IP (114.114.114.114):")
    result = tools.check_ip_in_china_api('114.114.114.114')
    print(f"   IP: {result['ip']}")
    print(f"   国家: {result['country_name']} ({result['country']})")
    print(f"   是否在中国: {result['is_in_china']}")
    if result['error']:
        print(f"   错误: {result['error']}")
    
    # 测试3: 查询美国IP
    print("\n3. 查询美国IP (8.8.8.8):")
    result = tools.check_ip_in_china_api('8.8.8.8')
    print(f"   IP: {result['ip']}")
    print(f"   国家: {result['country_name']} ({result['country']})")
    print(f"   是否在中国: {result['is_in_china']}")
    if result['error']:
        print(f"   错误: {result['error']}")
    
    # 测试4: 使用便捷函数
    print("\n4. 使用便捷函数:")
    print(f"   本机是否在中国: {is_ip_in_china()}")
    print(f"   114.114.114.114是否在中国: {is_ip_in_china('114.114.114.114')}")
    print(f"   8.8.8.8是否在中国: {is_ip_in_china('8.8.8.8')}")

