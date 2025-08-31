# AGIAgent MCP (Model Context Protocol) 支持

AGIAgent 现在支持 MCP (Model Context Protocol) 协议，可以与外部工具服务器进行通信，大大扩展了系统的工具生态。

## 🌟 功能特性

### MCP 协议支持
- **标准化工具调用**: 支持标准的 MCP JSON-RPC 协议
- **STDIO 传输**: 通过标准输入输出与 MCP 服务器通信
- **动态工具发现**: 运行时自动发现和注册外部工具
- **错误处理**: 完善的错误处理和重试机制

### 工具生态集成
- **官方 MCP 服务器**: 支持文件系统、GitHub、Slack 等官方服务器
- **第三方工具**: 支持社区开发的各种 MCP 工具
- **自定义工具**: 支持开发自定义 MCP 服务器

## 📋 配置说明

### 配置文件位置
```
config/mcp_servers.json
```

### 配置格式
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

### 配置参数说明
- **command**: 启动 MCP 服务器的命令
- **args**: 命令行参数
- **env**: 环境变量设置
- **transport**: 传输协议（目前仅支持 "stdio"）
- **timeout**: 工具调用超时时间（秒）
- **enabled**: 是否启用此服务器
- **capabilities**: 服务器提供的能力列表（文档用途）

## 🚀 快速开始

### 1. 启用文件系统工具
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

### 2. 使用 MCP 工具
在 AGIAgent 中，MCP 工具使用 `服务器名:工具名` 的格式调用：

```python
# 调用文件系统工具
result = executor.execute_tool({
    "name": "filesystem:read_file",
    "arguments": {
        "path": "/path/to/file.txt"
    }
})
```

## 🛠️ 支持的 MCP 服务器

### 官方服务器

#### 1. 文件系统服务器
```bash
npx -y @modelcontextprotocol/server-filesystem
```
**功能**: 文件读写、目录列表、文件搜索

#### 2. GitHub 服务器
```bash
npx -y @modelcontextprotocol/server-github
```
**功能**: 仓库操作、问题管理、PR 管理
**环境变量**: `GITHUB_PERSONAL_ACCESS_TOKEN`

#### 3. Slack 服务器
```bash
npx -y @modelcontextprotocol/server-slack
```
**功能**: 消息发送、频道管理、用户管理
**环境变量**: `SLACK_BOT_TOKEN`, `SLACK_TEAM_ID`

#### 4. 搜索服务器 (Brave)
```bash
npx -y @modelcontextprotocol/server-brave-search
```
**功能**: 网络搜索
**环境变量**: `BRAVE_API_KEY`

#### 5. 数据库服务器 (PostgreSQL)
```bash
npx -y @modelcontextprotocol/server-postgres
```
**功能**: 数据库查询、表管理
**环境变量**: `POSTGRES_CONNECTION_STRING`

### 第三方服务器

#### Puppeteer 服务器
```bash
npx -y @modelcontextprotocol/server-puppeteer
```
**功能**: 网页自动化、截图、数据抓取

#### SQLite 服务器
```bash
npx -y @modelcontextprotocol/server-sqlite
```
**功能**: SQLite 数据库操作

## 🔧 开发自定义 MCP 服务器

### Python 示例
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
                result = f"处理输入: {arguments.get('input', '')}"
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

### 配置自定义服务器
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

## 🐛 故障排除

### 常见问题

#### 1. MCP 服务器启动失败
- 检查 `command` 和 `args` 是否正确
- 确认所需的环境变量已设置
- 验证 MCP 服务器包是否已安装

#### 2. 工具调用超时
- 增加 `timeout` 值
- 检查网络连接
- 验证 MCP 服务器响应是否正常

#### 3. 权限错误
- 确认环境变量中的认证信息正确
- 检查 API 密钥权限
- 验证文件路径访问权限

### 调试模式
AGIAgent 支持详细的调试日志，可以帮助诊断 MCP 相关问题：

```python
# 启用调试模式
executor = ToolExecutor(debug_mode=True)
```

## 📚 更多资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)
- [MCP 规范](https://spec.modelcontextprotocol.io/)

## 🤝 贡献

欢迎贡献新的 MCP 服务器集成或改进现有功能！请查看贡献指南了解更多信息。

## 📄 许可证

此功能遵循 AGIAgent 项目的许可证条款。 