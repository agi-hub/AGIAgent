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

Image input function demonstration
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AGIAgentMain
except ImportError:
    print("❌ Unable to import AGIAgentMain")
    sys.exit(1)


def demo_image_input():
    """Demonstrate image input function"""
    print("🎯 AGIAgent image input function demonstration")
    print("=" * 50)
    
    # Check if there are available image files
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    
    # Look for image files in workspace directory
    image_files = []
    for file in os.listdir(workspace_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print("⚠️ No image files found in workspace directory")
        print("Please place an image file in the workspace directory")
        print("Supported formats: PNG, JPEG, GIF, BMP")
        return
    
    image_file = image_files[0]
    print(f"📸 Using image file: {image_file}")
    
    # Create AGIAgent instance
    try:
        agia = AGIAgentMain(
            debug_mode=True,
            detailed_summary=True,
            single_task_mode=True,
            interactive_mode=False
        )
        
        # Requirements using image input
        requirement_with_image = f"""
        Please analyze this image and describe its content. [img={image_file}]
        Then tell me the main features of the image.
        """
        
        print("📸 Execute task with image...")
        print(f"Task description: {requirement_with_image}")
        
        # Execute task
        success = agia.run(requirement_with_image)
        
        if success:
            print("✅ Task execution successful!")
        else:
            print("❌ Task execution failed")
            
    except Exception as e:
        print(f"❌ Demonstration execution failed: {e}")


def demo_multi_image_input():
    """Demonstrate multi-image input function"""
    print("\n🖼️ Multi-image input function demonstration")
    print("=" * 50)
    
    workspace_dir = "workspace"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    
    # Look for image files in workspace directory
    image_files = []
    for file in os.listdir(workspace_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(file)
    
    if len(image_files) < 2:
        print("⚠️ Need at least 2 image files to demonstrate multi-image function")
        print("Please place multiple image files in the workspace directory")
        return
    
    # Use the first two image files
    image1, image2 = image_files[:2]
    print(f"📸 Using image file: {image1} and {image2}")
    
    try:
        agia = AGIAgentMain(
            debug_mode=True,
            detailed_summary=True,
            single_task_mode=True,
            interactive_mode=False
        )
        
        # Requirements using multi-image input
        requirement_with_images = f"""
        Please analyze these two images and compare their content.
        First image: [img={image1}]
        Second image: [img={image2}]
        Tell me their similarities and differences.
        """
        
        print("📸 Execute task with multiple images...")
        print(f"Task description: {requirement_with_images}")
        
        # Execute task
        success = agia.run(requirement_with_images)
        
        if success:
            print("✅ Multi-image task execution successful!")
        else:
            print("❌ Multi-image task execution failed")
            
    except Exception as e:
        print(f"❌ Multi-image demonstration failed: {e}")


def demo_usage_guide():
    """Display usage guide"""
    print("\n📚 Image input function usage guide")
    print("=" * 50)
    
    print("1. Image tag format:")
    print("   [img=image_file.png]")
    print("   [img=path/to/image.jpg]")
    print("   [img=/absolute/path/to/image.jpeg]")
    
    print("\n2. Supported image formats:")
    print("   PNG, JPEG, JPG, GIF, BMP")
    
    print("\n3. Path description:")
    print("   - Relative path: relative to workspace directory")
    print("   - Absolute path: complete system path")
    
    print("\n4. Multi-image support:")
    print("   Multiple images can be included in one requirement")
    print("   For example: please analyze these images [img=img1.png] [img=img2.jpg]")
    
    print("\n5. Important features:")
    print("   - Images are only sent to the large model during the first iteration")
    print("   - Subsequent iterations will not repeatedly send image data")
    print("   - Supports Claude and OpenAI vision models")
    
    print("\n6. Example requirements:")
    print("   'Please analyze this chart [img=chart.png] and extract data'")
    print("   'Based on this design image [img=design.jpg] Generate HTML code'")
    print("   'Compare these two images [img=before.png] [img=after.png]'")


if __name__ == "__main__":
    # Display usage guide
    demo_usage_guide()
    
    # Basic demonstration
    demo_image_input()
    
    # Multi-image demonstration
    demo_multi_image_input()
    
    print("\n🎉 Demonstration completed!")
    print("💡 Tip: If the demonstration doesn't run:")
    print("   1. Place image files in the workspace directory")
    print("   2. Use large models that support vision functions")
    print("   3. Ensure API configuration is correct") 