#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像输入功能演示
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AGIBotMain
except ImportError:
    print("❌ 无法导入AGIBotMain，请确保在AGIBot项目目录中运行")
    sys.exit(1)


def demo_image_input():
    """演示图像输入功能"""
    print("🎯 AGIBot 图像输入功能演示")
    print("=" * 50)
    
    # 检查是否有可用的图像文件
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    
    # 寻找workspace目录中的图像文件
    image_files = []
    for file in os.listdir(workspace_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print("⚠️ 在workspace目录中没有找到图像文件")
        print("请将一个图像文件放在workspace目录中，然后重新运行演示")
        print("支持的格式：PNG, JPEG, GIF, BMP")
        return
    
    image_file = image_files[0]
    print(f"📸 使用图像文件: {image_file}")
    
    # 创建AGIBot实例
    try:
        agibot = AGIBotMain(
            debug_mode=True,
            detailed_summary=True,
            single_task_mode=True,
            interactive_mode=False
        )
        
        # 使用图像输入的需求
        requirement_with_image = f"""
        请分析这张图像并描述其内容。[img={image_file}]
        然后告诉我图像的主要特征。
        """
        
        print("📸 执行带图像的任务...")
        print(f"任务描述: {requirement_with_image}")
        
        # 执行任务
        success = agibot.run(requirement_with_image)
        
        if success:
            print("✅ 任务执行成功！")
        else:
            print("❌ 任务执行失败")
            
    except Exception as e:
        print(f"❌ 演示执行失败: {e}")


def demo_multi_image_input():
    """演示多图像输入功能"""
    print("\n🖼️ 多图像输入功能演示")
    print("=" * 50)
    
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    
    # 寻找workspace目录中的图像文件
    image_files = []
    for file in os.listdir(workspace_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(file)
    
    if len(image_files) < 2:
        print("⚠️ 需要至少2个图像文件来演示多图像功能")
        print("请将多个图像文件放在workspace目录中")
        return
    
    # 使用前两个图像文件
    image1, image2 = image_files[:2]
    print(f"📸 使用图像文件: {image1} 和 {image2}")
    
    try:
        agibot = AGIBotMain(
            debug_mode=True,
            detailed_summary=True,
            single_task_mode=True,
            interactive_mode=False
        )
        
        # 使用多图像输入的需求
        requirement_with_images = f"""
        请分析这两张图像并比较它们的内容。
        第一张图像：[img={image1}]
        第二张图像：[img={image2}]
        告诉我它们的相似点和不同点。
        """
        
        print("📸 执行带多图像的任务...")
        print(f"任务描述: {requirement_with_images}")
        
        # 执行任务
        success = agibot.run(requirement_with_images)
        
        if success:
            print("✅ 多图像任务执行成功！")
        else:
            print("❌ 多图像任务执行失败")
            
    except Exception as e:
        print(f"❌ 多图像演示失败: {e}")


def demo_usage_guide():
    """显示使用指南"""
    print("\n📚 图像输入功能使用指南")
    print("=" * 50)
    
    print("1. 图像标签格式：")
    print("   [img=image_file.png]")
    print("   [img=path/to/image.jpg]")
    print("   [img=/absolute/path/to/image.jpeg]")
    
    print("\n2. 支持的图像格式：")
    print("   PNG, JPEG, JPG, GIF, BMP")
    
    print("\n3. 路径说明：")
    print("   - 相对路径：相对于workspace目录")
    print("   - 绝对路径：系统完整路径")
    
    print("\n4. 多图像支持：")
    print("   在一个需求中可以包含多个图像")
    print("   例如：请分析这些图像 [img=img1.png] [img=img2.jpg]")
    
    print("\n5. 重要特性：")
    print("   - 图像只在第一次迭代时发送给大模型")
    print("   - 后续迭代不会重复发送图像数据")
    print("   - 支持Claude和OpenAI视觉模型")
    
    print("\n6. 示例需求：")
    print("   '请分析这张图表 [img=chart.png] 并提取数据'")
    print("   '基于这张设计图 [img=design.jpg] 生成HTML代码'")
    print("   '比较这两张图片 [img=before.png] [img=after.png]'")


if __name__ == "__main__":
    # 显示使用指南
    demo_usage_guide()
    
    # 基本演示
    demo_image_input()
    
    # 多图像演示
    demo_multi_image_input()
    
    print("\n🎉 演示完成！")
    print("💡 提示：如果演示没有运行，请确保:")
    print("   1. 在workspace目录中放置图像文件")
    print("   2. 使用支持视觉功能的大模型")
    print("   3. 确保API配置正确") 