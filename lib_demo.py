#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Bot Python Library Usage Examples

This file demonstrates how to use AGI Bot as a Python library instead of 
command-line tool. The library provides an OpenAI-like chat interface.
"""

import os
import sys

# Import the AGI Bot client
from main import AGIBotClient, create_client

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize the client with your API credentials
    client = AGIBotClient(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG",  # Replace with your actual API key
        model="claude-sonnet-4-0",  # or "gpt-4", "gpt-3.5-turbo", etc.
        api_base="https://api.openai-proxy.org/anthropic",  # Optional: API base URL
        debug_mode=False,  # Enable debug logging
        single_task_mode=True  # Use single task mode (recommended)
    )
    
    # Send a chat message (similar to OpenAI API)
    response = client.chat(
        messages=[
            {"role": "user", "content": "Create a simple Python calculator that can add, subtract, multiply and divide"}
        ],
        dir="calculator_project",  # Output directory
        loops=10  # Maximum execution rounds
    )
    
    # Check the result
    if response["success"]:
        print(f"✅ Task completed successfully!")
        print(f"📁 Output directory: {response['output_dir']}")
        print(f"💻 Workspace directory: {response['workspace_dir']}")
        print(f"⏱️ Execution time: {response['execution_time']:.2f} seconds")
    else:
        print(f"❌ Task failed: {response['message']}")
        print(f"📝 Error details: {response['details']}")

def example_with_continue_mode():
    """Example using continue mode to build upon previous work"""
    print("\n=== Continue Mode Example ===")
    
    client = AGIBotClient(
        api_key="your_api_key_here",
        model="claude-3-sonnet-20240229"
    )
    
    # First task: Create basic web app
    print("Step 1: Creating basic web app...")
    response1 = client.chat(
        messages=[
            {"role": "user", "content": "Create a simple Flask web application with a homepage"}
        ],
        dir="my_web_app"
    )
    
    if response1["success"]:
        print("✅ Basic web app created!")
        
        # Second task: Add features to the existing app
        print("Step 2: Adding features to existing app...")
        response2 = client.chat(
            messages=[
                {"role": "user", "content": "Add a contact form and an about page to the existing Flask app"}
            ],
            dir="my_web_app",  # Same directory
            continue_mode=True  # Continue from previous work
        )
        
        if response2["success"]:
            print("✅ Features added successfully!")
            print(f"📁 Final project: {response2['output_dir']}")
        else:
            print(f"❌ Failed to add features: {response2['message']}")
    else:
        print(f"❌ Failed to create basic app: {response1['message']}")

def example_multi_task_mode():
    """Example using multi-task mode for complex projects"""
    print("\n=== Multi-Task Mode Example ===")
    
    client = AGIBotClient(
        api_key="your_api_key_here",
        model="gpt-4",
        single_task_mode=False  # Enable multi-task mode
    )
    
    response = client.chat(
        messages=[
            {"role": "user", "content": "Create a complete e-commerce website with user authentication, product catalog, shopping cart, and payment integration"}
        ],
        dir="ecommerce_project",
        loops=15
    )
    
    if response["success"]:
        print("✅ E-commerce project completed!")
        print(f"📁 Project files: {response['output_dir']}")
    else:
        print(f"❌ Project failed: {response['message']}")

def example_with_custom_config():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Using the convenience function
    client = create_client(
        api_key="your_api_key_here",
        model="claude-3-haiku-20240307",  # Faster, cheaper model
        debug_mode=True,  # Enable detailed logging
        detailed_summary=True,  # Generate detailed reports
        interactive_mode=False  # Non-interactive execution
    )
    
    # Check current configuration
    config = client.get_config()
    print(f"Client configuration: {config}")
    
    # Get supported models
    models = client.get_models()
    print(f"Supported models: {models}")
    
    # Execute task
    response = client.chat(
        messages=[
            {"role": "user", "content": "Write a Python script that analyzes CSV files and generates charts"}
        ],
        dir="data_analysis_tool"
    )
    
    print(f"Task result: {'Success' if response['success'] else 'Failed'}")

def example_error_handling():
    """Example demonstrating error handling"""
    print("\n=== Error Handling Example ===")
    
    try:
        # This will raise an error due to missing API key
        client = AGIBotClient(
            api_key="",  # Empty API key
            model="gpt-4"
        )
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Valid client but invalid messages
    client = AGIBotClient(
        api_key="test_key",
        model="gpt-4"
    )
    
    # Test with invalid messages format
    response = client.chat(messages=[])  # Empty messages
    print(f"Empty messages result: {response['message']}")
    
    # Test with missing user message
    response = client.chat(messages=[{"role": "system", "content": "You are helpful"}])
    print(f"No user message result: {response['message']}")

def example_batch_processing():
    """Example of processing multiple tasks in batch"""
    print("\n=== Batch Processing Example ===")
    
    client = AGIBotClient(
        api_key="your_api_key_here",
        model="claude-3-sonnet-20240229"
    )
    
    tasks = [
        "Create a Python TODO list application",
        "Write a simple weather app using an API",
        "Build a password generator with GUI",
        "Create a file backup utility script"
    ]
    
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"Processing task {i}/{len(tasks)}: {task}")
        
        response = client.chat(
            messages=[{"role": "user", "content": task}],
            dir=f"batch_task_{i}"
        )
        
        results.append({
            "task": task,
            "success": response["success"],
            "output_dir": response["output_dir"] if response["success"] else None,
            "error": response["message"] if not response["success"] else None
        })
        
        print(f"Task {i} {'✅ completed' if response['success'] else '❌ failed'}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n📊 Batch processing complete: {successful}/{len(tasks)} tasks successful")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{status} Task {i}: {result['task'][:50]}...")

if __name__ == "__main__":
    print("AGI Bot Python Library Examples")
    print("=" * 50)
    
    # Note: Before running these examples, make sure to:
    # 1. Replace "your_api_key_here" with your actual API key
    # 2. Install required dependencies
    # 3. Have the main.py file in the same directory
    
    print("⚠️  Note: These examples use placeholder API keys.")
    print("   Please replace 'your_api_key_here' with your actual API key before running.")
    print()
    
    # Uncomment the examples you want to run:
    
    example_basic_usage()
    # example_with_continue_mode()
    # example_multi_task_mode()
    # example_with_custom_config()
    # example_error_handling()
    # example_batch_processing()
    
    print("Examples ready to run! Uncomment the function calls above to test.") 