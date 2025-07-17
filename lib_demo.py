#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Bot Python Library Usage Examples

This file demonstrates how to use AGI Bot as a Python library instead of 
command-line tool. The library provides an OpenAI-like chat interface.
"""

# Import the AGI Bot client
from src.main import AGIBotClient, create_client

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize the client (will automatically read from config/config.txt)
    client = AGIBotClient(
        # api_key and model will be read from config/config.txt automatically
        # You can also specify them explicitly: api_key="your_api_key", model="claude-sonnet-4-0"
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
        # Will read from config/config.txt automatically
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

def example_with_custom_config():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Using the convenience function (will read api_key and model from config/config.txt)
    client = create_client(
        # api_key and model will be read from config/config.txt automatically
        # You can override with: model="claude-3-haiku-20240307"  # Faster, cheaper model
        debug_mode=True,  # Enable detailed logging
        detailed_summary=True,  # Generate detailed reports
        interactive_mode=False,  # Non-interactive execution
        MCP_config_file="config/custom_mcp_servers.json",  # Custom MCP configuration
        prompts_folder="custom_prompts"  # Custom prompts folder
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

def example_with_custom_mcp_and_prompts():
    """Example using custom MCP configuration and prompts folder"""
    print("\n=== Custom MCP and Prompts Example ===")
    
    client = AGIBotClient(
        # api_key and model will be read from config/config.txt automatically
        debug_mode=False,
        MCP_config_file="config/specialized_mcp_servers.json",  # Use specialized MCP tools
        prompts_folder="specialized_prompts"  # Use specialized prompts for different domains
    )
    
    # This allows you to:
    # 1. Use different MCP server configurations for different projects
    # 2. Use different prompt templates and tool interfaces
    # 3. Create domain-specific AGIBot instances (e.g., for data science, web development, etc.)
    
    response = client.chat(
        messages=[
            {"role": "user", "content": "Create a machine learning pipeline for time series forecasting"}
        ],
        dir="ml_pipeline_project"
    )
    
    if response["success"]:
        print("✅ ML pipeline created with specialized tools and prompts!")
        print(f"📁 Output: {response['output_dir']}")
    else:
        print(f"❌ Failed: {response['message']}")

def example_batch_processing():
    """Example of processing multiple tasks in batch"""
    print("\n=== Batch Processing Example ===")
    
    client = AGIBotClient(
        # Will read from config/config.txt automatically
    )
    
    tasks = [
        "Create a Python TODO list application",
        "Write a simple weather app using an API"
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
    # 1. Set your API key and model in config/config.txt
    # 2. Install required dependencies
    # 3. Have the src/ directory with the AGIBot source code
    
    print("ℹ️  Note: These examples will automatically read API key and model from config/config.txt")
    print("   Make sure your config/config.txt file contains valid API_KEY and MODEL settings.")
    print()
    
    # Uncomment the examples you want to run:
    
    example_basic_usage()
    example_with_continue_mode()
    example_with_custom_config()
    example_with_custom_mcp_and_prompts()
    example_batch_processing()
    
    print("Examples ready to run! Uncomment the function calls above to test.") 