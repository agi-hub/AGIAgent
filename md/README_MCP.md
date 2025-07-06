# AGIBot MCP (Model Context Protocol) Support

AGIBot now supports the MCP (Model Context Protocol) protocol, enabling communication with external tool servers and greatly expanding the system's tool ecosystem.

## 🌟 Features

### MCP Protocol Support
- **Standardized Tool Calls**: Supports standard MCP JSON-RPC protocol
- **STDIO Transport**: Communicates with MCP servers through standard input/output
- **Dynamic Tool Discovery**: Automatically discovers and registers external tools at runtime
- **Error Handling**: Comprehensive error handling and retry mechanisms

### Tool Ecosystem Integration
- **Official MCP Servers**: Supports official servers for filesystem, GitHub, Slack, etc.
- **Third-party Tools**: Supports various MCP tools developed by the community
- **Custom Tools**: Supports development of custom MCP servers

## 📋 Configuration

### Configuration File Location
```
config/mcp_servers.json
```

### Configuration Format
```json
{
  "mcp_servers": {
    "server_name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "env": {
        "ENV_VAR": "value"
      },
      "transport": "stdio",
      "timeout": 30,
      "enabled": true,
      "capabilities": ["tool1", "tool2"]
    }
  }
}
```

### Configuration Parameters
- **command**: Command to start the MCP server
- **args**: Command line arguments
- **env**: Environment variable settings
- **transport**: Transport protocol (currently only supports "stdio")
- **timeout**: Tool call timeout in seconds
- **enabled**: Whether to enable this server
- **capabilities**: List of capabilities provided by the server (for documentation purposes)

## 🚀 Quick Start

### 1. Enable Filesystem Tools
```json
{
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "env": {
        "HOME": "/path/to/your/home"
      },
      "transport": "stdio",
      "timeout": 30,
      "enabled": true,
      "capabilities": ["read_file", "write_file", "list_directory"]
    }
  }
}
```

### 2. Using MCP Tools
In AGIBot, MCP tools are called using the format `server_name:tool_name`:

```python
# Call filesystem tool
result = executor.execute_tool({
    "name": "filesystem:read_file",
    "arguments": {
        "path": "/path/to/file.txt"
    }
})
```

## 🛠️ Supported MCP Servers

### Official Servers

#### 1. Filesystem Server
```bash
npx -y @modelcontextprotocol/server-filesystem
```
**Features**: File read/write, directory listing, file search

#### 2. GitHub Server
```bash
npx -y @modelcontextprotocol/server-github
```
**Features**: Repository operations, issue management, PR management
**Environment Variables**: `GITHUB_PERSONAL_ACCESS_TOKEN`

#### 3. Slack Server
```bash
npx -y @modelcontextprotocol/server-slack
```
**Features**: Message sending, channel management, user management
**Environment Variables**: `SLACK_BOT_TOKEN`, `SLACK_TEAM_ID`

#### 4. Search Server (Brave)
```bash
npx -y @modelcontextprotocol/server-brave-search
```
**Features**: Web search
**Environment Variables**: `BRAVE_API_KEY`

#### 5. Database Server (PostgreSQL)
```bash
npx -y @modelcontextprotocol/server-postgres
```
**Features**: Database queries, table management
**Environment Variables**: `POSTGRES_CONNECTION_STRING`

### Third-party Servers

#### Puppeteer Server
```bash
npx -y @modelcontextprotocol/server-puppeteer
```
**Features**: Web automation, screenshots, data scraping

#### SQLite Server
```bash
npx -y @modelcontextprotocol/server-sqlite
```
**Features**: SQLite database operations

## 🔧 Developing Custom MCP Servers

### Python Example
```python
#!/usr/bin/env python3
import json
import sys
from typing import Dict, Any

class CustomMCPServer:
    def __init__(self):
        self.tools = {
            "custom_tool": {
                "name": "custom_tool",
                "description": "A custom tool example",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input text"
                        }
                    },
                    "required": ["input"]
                }
            }
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "tools": list(self.tools.values())
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "custom_tool":
                result = f"Processing input: {arguments.get('input', '')}"
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32601,
                "message": "Method not found"
            }
        }

def main():
    server = CustomMCPServer()
    
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
```

### Configure Custom Server
```json
{
  "mcp_servers": {
    "custom": {
      "command": "python",
      "args": ["path/to/custom_mcp_server.py"],
      "env": {},
      "transport": "stdio",
      "timeout": 30,
      "enabled": true,
      "capabilities": ["custom_operations"]
    }
  }
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. MCP Server Startup Failure
- Check if `command` and `args` are correct
- Ensure required environment variables are set
- Verify that MCP server packages are installed

#### 2. Tool Call Timeout
- Increase the `timeout` value
- Check network connectivity
- Verify MCP server response is normal

#### 3. Permission Errors
- Ensure authentication information in environment variables is correct
- Check API key permissions
- Verify file path access permissions

### Debug Mode
AGIBot supports detailed debug logging to help diagnose MCP-related issues:

```python
# Enable debug mode
executor = ToolExecutor(debug_mode=True)
```

## 📚 Additional Resources

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Server List](https://github.com/modelcontextprotocol/servers)
- [MCP Specification](https://spec.modelcontextprotocol.io/)

## 🤝 Contributing

Contributions of new MCP server integrations or improvements to existing functionality are welcome! Please check the contribution guidelines for more information.

## 📄 License

This feature follows the license terms of the AGIBot project. 