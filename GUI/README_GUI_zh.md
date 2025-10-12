# AGI Agent GUI

AGI Agent 的图形用户界面，提供直观便捷的任务执行和文件管理功能。

## 🚀 快速开始

### 启动GUI, 在工程根目录下执行:
```bash
python GUI/app.py
```

启动后访问：`http://localhost:5002`

## 界面示例
<div align="center">
      <img src="../fig/AGIAgent_GUI.png" alt="AGI Agent GUI"/>
</div>

## 主要用法

您可以新建或选择一个工作区, 并将需要处理的数据文件通过工作区的上传按钮上传, 写入需求, 并按执行按钮运行, 程序会执行最多50轮迭代, 运行结束后, 可以从工作区看到生成的文件, 此时可以点击下载工作区的按钮进行下载. 运行过程中及结束后,您都可以预览已经产生的文件. 

当选择一个工作区时, 务必将这个工作区点击为蓝色高亮状态.

当任务执行完毕或被中断后, 您可以通过选择工作区并输入提示词继续任务, 但需要注意上一轮的需求及执行过程并没有带入到本次运行.


## 🔐 用户认证与多用户管理

### 登录方式
AGI Agent GUI 采用 API Key 认证方式：
1. 启动 GUI 后，首次访问会要求输入 API Key
2. 输入有效的 API Key 后即可登录使用
3. API Key 在浏览器会话中保持有效，关闭浏览器后需重新输入

### 默认账户
系统预置了以下测试账户：
- **用户名**: `agiagenttest`，**API Key**（请你输入这个登录）: `agiatest`
- **用户名**: `guest`，**API Key**（请你输入这个登录）: ``（无内容）
注：guest用户无法修改内容，也无法启动新的任务，仅供预览。
> ⚠️ **安全提醒**: 生产环境中请及时修改或删除默认账户，创建专属的安全账户。

### 创建新账户

#### 方法一：交互式创建（推荐）
```bash
cd GUI
python create_user.py
```
按照提示输入用户信息：
- 用户名
- API Key（可手动输入或自动生成）
- 用户描述
- 权限设置（read, write, execute, admin）
- 过期时间（可选）

#### 方法二：命令行创建
```bash
cd GUI
# 创建普通用户
python create_user.py -u alice -k alice123 -d "Alice用户"

```

#### 方法三：查看现有用户
```bash
cd GUI
python create_user.py --list
```

### 权限说明
- **read**: 读取权限，可以查看工作区和文件
- **write**: 写入权限，可以上传文件和修改内容  
- **execute**: 执行权限，可以运行任务和执行命令
- **admin**: 管理员权限，拥有所有权限

### 账户管理文件
用户认证信息存储在：`config/authorized_keys.json`

该文件包含：
- 用户名和描述信息
- API Key 的 SHA256 哈希值（不存储明文）
- 用户权限列表
- 创建时间和过期时间
- 账户启用状态

> 🔒 **安全特性**: 系统仅存储 API Key 的哈希值，不保存明文密码，确保账户安全。

## 🔧 配置说明

### 环境要求
- Python 3.8+
- Flask
- Flask-SocketIO
- 其他依赖见 requirements.txt

### 配置文件
GUI会读取主目录的 `config/config.txt` 配置：
- `language`: 界面语言 (zh/en)
- `gui_default_data_directory`: GUI数据目录路径

