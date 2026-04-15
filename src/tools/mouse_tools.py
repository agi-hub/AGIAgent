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
"""

import os
import platform
import subprocess
import time
from typing import Dict, Any, Optional
from .print_system import print_current, print_debug


class MouseTools:
    """鼠标操作工具类，支持跨平台的鼠标控制操作"""
    
    def __init__(self):
        """初始化鼠标工具"""
        self.system = platform.system()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查并尝试导入鼠标控制库"""
        self.pyautogui_available = False
        self.pynput_available = False
        
        # 尝试导入 pyautogui
        try:
            import pyautogui
            self.pyautogui = pyautogui
            self.pyautogui_available = True

            # 在 macOS 上测试权限
            if self.system == 'Darwin':
                try:
                    # 尝试获取屏幕尺寸来测试权限
                    _ = pyautogui.size()
                except Exception as e:
                    if 'permission' in str(e).lower() or 'accessibility' in str(e).lower():
                        print_current("⚠️ PyAutoGUI 需要辅助功能权限。请在系统设置中授予权限。")
        except Exception:
            # Catches ImportError as well as KeyError('DISPLAY') raised by mouseinfo
            # when no display server is available (headless environment)
            pass

        # 尝试导入 pynput
        if not self.pyautogui_available:
            try:
                from pynput.mouse import Button, Controller
                self.mouse_controller = Controller()
                self.Button = Button
                self.pynput_available = True
                print_current("✅ PyNput 可用，将使用 PyNput 进行鼠标操作")

                # 在 macOS 上测试权限
                if self.system == 'Darwin':
                    try:
                        # 尝试获取当前位置来测试权限
                        _ = self.mouse_controller.position
                    except Exception as e:
                        if 'permission' in str(e).lower() or 'accessibility' in str(e).lower():
                            print_current("⚠️ PyNput 需要辅助功能权限。请在系统设置中授予权限。")
            except Exception:
                # Catches ImportError as well as display-related errors in headless environments
                pass
    
    def mouse_control(self, action: str, x: Optional[int] = None, y: Optional[int] = None, 
                     button: Optional[str] = None, clicks: Optional[int] = None, 
                     scroll_delta: Optional[int] = None) -> Dict[str, Any]:
        """
        控制鼠标操作
        
        Args:
            action: 操作类型，可选值: 'move', 'click', 'double_click', 'right_click', 'scroll'
            x: X坐标（移动、点击、滚轮时使用）
            y: Y坐标（移动、点击、滚轮时使用）
            button: 按钮类型，'left' 或 'right'（仅用于点击操作）
            clicks: 点击次数（默认1次，双击时为2次）
            scroll_delta: 滚轮滚动量，正数向上，负数向下
        
        Returns:
            操作结果字典
        """
        try:
            print_current(f"🖱️ 执行鼠标操作: action={action}, x={x}, y={y}, button={button}, clicks={clicks}, scroll_delta={scroll_delta}")
            
            if action == 'move':
                return self._move_mouse(x, y)
            elif action == 'click':
                return self._click_mouse(x, y, button or 'left', clicks or 1)
            elif action == 'double_click':
                return self._double_click_mouse(x, y)
            elif action == 'right_click':
                return self._right_click_mouse(x, y)
            elif action == 'scroll':
                return self._scroll_mouse(x, y, scroll_delta or 0)
            else:
                return {
                    'status': 'error',
                    'message': f'不支持的操作类型: {action}。支持的操作: move, click, double_click, right_click, scroll'
                }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print_current(f"❌ 鼠标操作异常: {str(e)}")
            print_current(f"详细错误信息: {error_details}")
            
            # 检查是否是 macOS 权限问题
            if self.system == 'Darwin' and 'permission' in str(e).lower():
                return {
                    'status': 'error',
                    'message': f'鼠标操作失败: {str(e)}。在 macOS 上，请确保已授予终端或 Python 辅助功能权限。前往"系统设置" > "隐私与安全性" > "辅助功能"中添加权限。',
                    'error_type': 'permission_error',
                    'system': 'macOS'
                }
            
            return {
                'status': 'error',
                'message': f'鼠标操作失败: {str(e)}',
                'error_details': error_details
            }
    
    def _move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """移动鼠标到指定坐标"""
        if x is None or y is None:
            return {'status': 'error', 'message': '移动鼠标需要提供 x 和 y 坐标'}
        
        try:
            if self.pyautogui_available:
                try:
                    self.pyautogui.moveTo(x, y)
                    print_current(f"🖱️ 鼠标已移动到坐标 ({x}, {y})")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            elif self.pynput_available:
                try:
                    self.mouse_controller.position = (x, y)
                    print_current(f"🖱️ 鼠标已移动到坐标 ({x}, {y})")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            else:
                return self._move_mouse_system_command(x, y)
            
            return {
                'status': 'success',
                'action': 'move',
                'x': x,
                'y': y,
                'message': f'鼠标已移动到坐标 ({x}, {y})'
            }
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ 移动鼠标失败: {error_msg}")
            return {
                'status': 'error', 
                'message': f'移动鼠标失败: {error_msg}',
                'x': x,
                'y': y
            }
    
    def _click_mouse(self, x: int, y: int, button: str, clicks: int) -> Dict[str, Any]:
        """单击鼠标"""
        if x is None or y is None:
            return {'status': 'error', 'message': '点击鼠标需要提供 x 和 y 坐标'}
        
        try:
            if self.pyautogui_available:
                try:
                    if button == 'right':
                        self.pyautogui.rightClick(x, y, clicks=clicks)
                    else:
                        self.pyautogui.click(x, y, clicks=clicks)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            elif self.pynput_available:
                try:
                    # 先移动鼠标
                    self.mouse_controller.position = (x, y)
                    time.sleep(0.05)  # 短暂延迟确保移动完成
                    # 执行点击
                    btn = self.Button.right if button == 'right' else self.Button.left
                    for _ in range(clicks):
                        self.mouse_controller.click(btn, 1)
                        if clicks > 1:
                            time.sleep(0.1)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            else:
                return self._click_mouse_system_command(x, y, button, clicks)
            
            return {
                'status': 'success',
                'action': 'click',
                'x': x,
                'y': y,
                'button': button,
                'clicks': clicks,
                'message': f'在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击'
            }
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ 点击鼠标失败: {error_msg}")
            return {
                'status': 'error', 
                'message': f'点击鼠标失败: {error_msg}',
                'x': x,
                'y': y,
                'button': button
            }
    
    def _double_click_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """双击鼠标左键"""
        if x is None or y is None:
            return {'status': 'error', 'message': '双击鼠标需要提供 x 和 y 坐标'}
        
        try:
            if self.pyautogui_available:
                try:
                    self.pyautogui.doubleClick(x, y)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了双击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            elif self.pynput_available:
                try:
                    self.mouse_controller.position = (x, y)
                    time.sleep(0.05)
                    self.mouse_controller.click(self.Button.left, 2)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了双击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            else:
                return self._click_mouse_system_command(x, y, 'left', 2)
            
            return {
                'status': 'success',
                'action': 'double_click',
                'x': x,
                'y': y,
                'message': f'在坐标 ({x}, {y}) 执行了双击'
            }
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ 双击鼠标失败: {error_msg}")
            return {
                'status': 'error', 
                'message': f'双击鼠标失败: {error_msg}',
                'x': x,
                'y': y
            }
    
    def _right_click_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """右键点击鼠标"""
        if x is None or y is None:
            return {'status': 'error', 'message': '右键点击需要提供 x 和 y 坐标'}
        
        try:
            if self.pyautogui_available:
                try:
                    self.pyautogui.rightClick(x, y)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了右键点击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            elif self.pynput_available:
                try:
                    self.mouse_controller.position = (x, y)
                    time.sleep(0.05)
                    self.mouse_controller.click(self.Button.right, 1)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了右键点击")
                except Exception as e:
                    error_msg = str(e)
                    if 'permission' in error_msg.lower() or 'accessibility' in error_msg.lower():
                        raise Exception(f"权限错误: {error_msg}。在 macOS 上需要辅助功能权限。")
                    raise
            else:
                return self._click_mouse_system_command(x, y, 'right', 1)
            
            return {
                'status': 'success',
                'action': 'right_click',
                'x': x,
                'y': y,
                'message': f'在坐标 ({x}, {y}) 执行了右键点击'
            }
        except Exception as e:
            error_msg = str(e)
            print_current(f"❌ 右键点击失败: {error_msg}")
            return {
                'status': 'error', 
                'message': f'右键点击失败: {error_msg}',
                'x': x,
                'y': y
            }
    
    def _scroll_mouse(self, x: int, y: int, delta: int) -> Dict[str, Any]:
        """滚动鼠标滚轮"""
        if x is None or y is None:
            return {'status': 'error', 'message': '滚动鼠标需要提供 x 和 y 坐标'}
        
        if delta == 0:
            return {'status': 'error', 'message': '滚动量不能为0'}
        
        try:
            if self.pyautogui_available:
                # 先移动鼠标到指定位置
                self.pyautogui.moveTo(x, y)
                # PyAutoGUI 的 scroll 函数，正数向上，负数向下
                self.pyautogui.scroll(delta, x=x, y=y)
                direction = "向上" if delta > 0 else "向下"
                print_current(f"🖱️ 在坐标 ({x}, {y}) 滚轮{direction}滚动 {abs(delta)} 个单位")
            elif self.pynput_available:
                self.mouse_controller.position = (x, y)
                time.sleep(0.05)
                # pynput 的 scroll 函数，正数向上，负数向下
                self.mouse_controller.scroll(0, delta)
                direction = "向上" if delta > 0 else "向下"
                print_current(f"🖱️ 在坐标 ({x}, {y}) 滚轮{direction}滚动 {abs(delta)} 个单位")
            else:
                return self._scroll_mouse_system_command(x, y, delta)
            
            return {
                'status': 'success',
                'action': 'scroll',
                'x': x,
                'y': y,
                'delta': delta,
                'message': f'在坐标 ({x}, {y}) 滚轮{"向上" if delta > 0 else "向下"}滚动 {abs(delta)} 个单位'
            }
        except Exception as e:
            return {'status': 'error', 'message': f'滚动鼠标失败: {str(e)}'}
    
    def _move_mouse_system_command(self, x: int, y: int) -> Dict[str, Any]:
        """使用系统命令移动鼠标（备用方案）"""
        try:
            if self.system == 'Darwin':  # macOS
                # 使用 AppleScript
                script = f'''
                tell application "System Events"
                    set mouse position to {{{x}, {y}}}
                end tell
                '''
                result = subprocess.run(['osascript', '-e', script], 
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print_current(f"🖱️ 鼠标已移动到坐标 ({x}, {y})")
                    return {
                        'status': 'success',
                        'action': 'move',
                        'x': x,
                        'y': y,
                        'message': f'鼠标已移动到坐标 ({x}, {y})'
                    }
            elif self.system == 'Linux':
                # 尝试使用 xdotool
                try:
                    result = subprocess.run(['xdotool', 'mousemove', str(x), str(y)],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print_current(f"🖱️ 鼠标已移动到坐标 ({x}, {y})")
                        return {
                            'status': 'success',
                            'action': 'move',
                            'x': x,
                            'y': y,
                            'message': f'鼠标已移动到坐标 ({x}, {y})'
                        }
                except FileNotFoundError:
                    pass
            elif self.system == 'Windows':
                # Windows 可以使用 PowerShell 或 AutoIt
                # 这里使用 PowerShell
                script = f'''
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({x}, {y})
                '''
                result = subprocess.run(['powershell', '-Command', script],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print_current(f"🖱️ 鼠标已移动到坐标 ({x}, {y})")
                    return {
                        'status': 'success',
                        'action': 'move',
                        'x': x,
                        'y': y,
                        'message': f'鼠标已移动到坐标 ({x}, {y})'
                    }
            
            return {'status': 'error', 'message': '系统命令移动鼠标失败，请安装 PyAutoGUI 或 PyNput'}
        except Exception as e:
            return {'status': 'error', 'message': f'系统命令移动鼠标失败: {str(e)}'}
    
    def _click_mouse_system_command(self, x: int, y: int, button: str, clicks: int) -> Dict[str, Any]:
        """使用系统命令点击鼠标（备用方案）"""
        try:
            if self.system == 'Darwin':  # macOS
                # 先移动鼠标
                move_result = self._move_mouse_system_command(x, y)
                if move_result['status'] != 'success':
                    return move_result
                
                # 使用 AppleScript 点击
                button_name = 'right' if button == 'right' else 'left'
                script = f'''
                tell application "System Events"
                    {clicks} times
                        click {button_name} button
                    end repeat
                end tell
                '''
                result = subprocess.run(['osascript', '-e', script],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击")
                    return {
                        'status': 'success',
                        'action': 'click',
                        'x': x,
                        'y': y,
                        'button': button,
                        'clicks': clicks,
                        'message': f'在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击'
                    }
            elif self.system == 'Linux':
                # 使用 xdotool
                try:
                    # 先移动
                    subprocess.run(['xdotool', 'mousemove', str(x), str(y)],
                                 capture_output=True, timeout=5)
                    time.sleep(0.1)
                    # 点击
                    click_type = '3' if button == 'right' else '1'  # 3=右键, 1=左键
                    for _ in range(clicks):
                        subprocess.run(['xdotool', 'click', click_type],
                                     capture_output=True, timeout=5)
                        if clicks > 1:
                            time.sleep(0.1)
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击")
                    return {
                        'status': 'success',
                        'action': 'click',
                        'x': x,
                        'y': y,
                        'button': button,
                        'clicks': clicks,
                        'message': f'在坐标 ({x}, {y}) 执行了 {clicks} 次{button}键点击'
                    }
                except FileNotFoundError:
                    pass
            elif self.system == 'Windows':
                # 使用 PowerShell
                move_result = self._move_mouse_system_command(x, y)
                if move_result['status'] != 'success':
                    return move_result
                
                # Windows 点击需要使用不同的方法
                # 这里简化处理，建议安装 PyAutoGUI
                return {'status': 'error', 'message': 'Windows 系统命令点击功能受限，请安装 PyAutoGUI 或 PyNput'}
            
            return {'status': 'error', 'message': '系统命令点击鼠标失败，请安装 PyAutoGUI 或 PyNput'}
        except Exception as e:
            return {'status': 'error', 'message': f'系统命令点击鼠标失败: {str(e)}'}
    
    def _scroll_mouse_system_command(self, x: int, y: int, delta: int) -> Dict[str, Any]:
        """使用系统命令滚动鼠标（备用方案）"""
        try:
            if self.system == 'Darwin':  # macOS
                # 先移动鼠标
                move_result = self._move_mouse_system_command(x, y)
                if move_result['status'] != 'success':
                    return move_result
                
                # macOS 滚动需要使用 CGEvent
                # 这里简化处理，建议安装 PyAutoGUI
                return {'status': 'error', 'message': 'macOS 系统命令滚动功能受限，请安装 PyAutoGUI 或 PyNput'}
            elif self.system == 'Linux':
                # 使用 xdotool
                try:
                    subprocess.run(['xdotool', 'mousemove', str(x), str(y)],
                                 capture_output=True, timeout=5)
                    time.sleep(0.1)
                    # xdotool 滚动，正数向上，负数向下
                    subprocess.run(['xdotool', 'click', '--repeat', str(abs(delta)), 
                                  '--delay', '10', '4' if delta > 0 else '5'],
                                 capture_output=True, timeout=5)
                    direction = "向上" if delta > 0 else "向下"
                    print_current(f"🖱️ 在坐标 ({x}, {y}) 滚轮{direction}滚动 {abs(delta)} 个单位")
                    return {
                        'status': 'success',
                        'action': 'scroll',
                        'x': x,
                        'y': y,
                        'delta': delta,
                        'message': f'在坐标 ({x}, {y}) 滚轮{direction}滚动 {abs(delta)} 个单位'
                    }
                except FileNotFoundError:
                    pass
            elif self.system == 'Windows':
                # Windows 滚动需要使用不同的方法
                return {'status': 'error', 'message': 'Windows 系统命令滚动功能受限，请安装 PyAutoGUI 或 PyNput'}
            
            return {'status': 'error', 'message': '系统命令滚动鼠标失败，请安装 PyAutoGUI 或 PyNput'}
        except Exception as e:
            return {'status': 'error', 'message': f'系统命令滚动鼠标失败: {str(e)}'}

