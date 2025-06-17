#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

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

"""
AGI Bot startup script
"""

import sys

# Application name macro definition
APP_NAME = "AGI Bot"

def main():
    print(f"🚀 Starting {APP_NAME}...")
    print("📍 Access URLs:")
    print("   Main interface: http://localhost:5001")
    print("🔧 Features:")
    print("   📤 Direct execution (blue button) - default mode")
    print("   📋 Plan mode (orange button) - task decomposition")
    print("   ➕ New directory (green button)")

    
    # Import and run app
    from app import app, socketio
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print(f"\n👋 {APP_NAME} stopped")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 