#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据采集工具使用示例
Sensor Data Acquisition Tool Usage Examples

该文件演示了如何使用 get_sensor_data 工具获取物理世界信息。
This file demonstrates how to use the get_sensor_data tool to acquire physical world information.
"""

from tools.sensor_tools import SensorDataCollector

def main():
    """主函数，演示各种传感器数据采集功能"""
    
    # 初始化传感器数据采集器
    collector = SensorDataCollector()
    
    print("=== 传感器数据采集工具使用示例 ===")
    print("=== Sensor Data Acquisition Tool Usage Examples ===\n")
    
    # 示例1：从摄像头采集图像
    print("1. 从摄像头采集图像 (Capture image from camera)")
    result = collector.get_sensor_data(
        type=1,  # 图像类型
        source="0",  # 默认摄像头
        para={"resolution": "640x320"}
    )
    print(f"   结果 (Result): {result['success']}")
    if result['success']:
        print(f"   数据格式 (Data format): {result['dataformat']}")
        print(f"   文件大小 (File size): {result.get('file_size', 'N/A')} bytes")
    else:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    # 示例2：从文件加载图像
    print("2. 从文件加载图像 (Load image from file)")
    result = collector.get_sensor_data(
        type=1,  # 图像类型
        source="example_image.jpg",  # 示例图像文件
        para={}
    )
    print(f"   结果 (Result): {result['success']}")
    if not result['success']:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    # 示例3：从摄像头录制视频
    print("3. 从摄像头录制视频 (Record video from camera)")
    result = collector.get_sensor_data(
        type=2,  # 视频类型
        source="0",  # 默认摄像头
        para={"resolution": "640x320", "duration": 5}
    )
    print(f"   结果 (Result): {result['success']}")
    if result['success']:
        print(f"   数据格式 (Data format): {result['dataformat']}")
        print(f"   文件路径 (File path): {result.get('file_path', 'N/A')}")
        print(f"   文件大小 (File size): {result.get('file_size', 'N/A')} bytes")
    else:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    # 示例4：从麦克风录制音频
    print("4. 从麦克风录制音频 (Record audio from microphone)")
    result = collector.get_sensor_data(
        type=3,  # 音频类型
        source="default",  # 默认麦克风
        para={"sampling_rate": 16000, "duration": 5}
    )
    print(f"   结果 (Result): {result['success']}")
    if result['success']:
        print(f"   数据格式 (Data format): {result['dataformat']}")
        print(f"   文件路径 (File path): {result.get('file_path', 'N/A')}")
        print(f"   采样率 (Sampling rate): {result.get('sampling_rate', 'N/A')} Hz")
    else:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    # 示例5：读取传感器数据
    print("5. 读取传感器数据 (Read sensor data)")
    result = collector.get_sensor_data(
        type=4,  # 传感器类型
        source="/sys/class/thermal/thermal_zone0/temp",  # CPU温度传感器
        para={}
    )
    print(f"   结果 (Result): {result['success']}")
    if result['success']:
        print(f"   数据格式 (Data format): {result['dataformat']}")
        print(f"   数据值 (Data value): {result.get('data', 'N/A')}")
    else:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    # 示例6：从JSON文件读取传感器数据
    print("6. 从JSON文件读取传感器数据 (Read sensor data from JSON file)")
    result = collector.get_sensor_data(
        type=4,  # 传感器类型
        source="sensor_data.json",  # 传感器数据文件
        para={}
    )
    print(f"   结果 (Result): {result['success']}")
    if not result['success']:
        print(f"   错误 (Error): {result['error']}")
    print()
    
    print("=== 示例完成 ===")
    print("=== Examples completed ===")


# 工具调用示例（Tool calling examples）
def example_tool_calls():
    """演示如何在工具调用中使用传感器数据采集功能"""
    
    print("\n=== 工具调用示例 ===")
    print("=== Tool Calling Examples ===\n")
    
    # 这些是可以在AGI系统中使用的工具调用格式
    # These are the tool calling formats that can be used in AGI systems
    
    examples = [
        {
            "name": "get_sensor_data",
            "description": "拍摄一张照片 (Take a photo)",
            "arguments": {
                "type": 1,
                "source": "0",
                "para": {"resolution": "1920x1080"}
            }
        },
        {
            "name": "get_sensor_data", 
            "description": "录制5秒视频 (Record 5-second video)",
            "arguments": {
                "type": 2,
                "source": "/dev/video0",
                "para": {"resolution": "640x320", "duration": 5}
            }
        },
        {
            "name": "get_sensor_data",
            "description": "录制3秒音频 (Record 3-second audio)",
            "arguments": {
                "type": 3,
                "source": "default",
                "para": {"sampling_rate": 44100, "duration": 3}
            }
        },
        {
            "name": "get_sensor_data",
            "description": "读取CPU温度 (Read CPU temperature)",
            "arguments": {
                "type": 4,
                "source": "/sys/class/thermal/thermal_zone0/temp",
                "para": {}
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   Tool: {example['name']}")
        print(f"   Arguments: {example['arguments']}")
        print()


if __name__ == "__main__":
    main()
    example_tool_calls() 