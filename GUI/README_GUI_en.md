# AGI Bot GUI

Graphical user interface for AGI Bot, providing intuitive and convenient task execution and file management functions.

## 🚀 Quick Start

### Launch GUI, execute in the project root directory:
```bash
python GUI/app.py
```

After startup, visit: `http://localhost:5001`

## Interface Example
<div align="center">
      <img src="../fig/AGIBot_GUI.png" alt="AGI Bot GUI"/>
</div>

## Usage

You can create a new workspace or select an existing one, upload data files that need to be processed through the workspace upload button, write your requirements, and click the execute button to run. The program will execute up to 50 iterations. After completion, you can view the generated files in the workspace and download the workspace by clicking the download button. During execution and after completion, you can preview the files that have been generated.

When selecting a workspace, make sure to click it to highlight it in blue.

After task execution is completed or interrupted, you can continue the task by selecting the workspace and entering a prompt. However, note that the previous round's requirements and execution process are not carried over to the current run.

## 🔐 User Authentication & Multi-User Management

### Login Method
AGI Bot GUI uses API Key authentication:
1. After starting the GUI, first-time access will require entering an API Key
2. Enter a valid API Key to log in and use the system
3. API Key remains valid during the browser session, requires re-entry after closing browser

### Default Accounts
The system comes with the following test accounts:
- **Username**: `testuser`, **API Key**: `testuser123` (Regular user permissions)
- **Username**: `admin`, **API Key**: `admin123` (Administrator permissions)

> ⚠️ **Security Notice**: In production environments, please modify or delete default accounts promptly and create dedicated secure accounts.

### Creating New Accounts

#### Method 1: Interactive Creation (Recommended)
```bash
# From project root directory
python GUI/create_user.py
```
Follow the prompts to enter user information:
- Username
- API Key (manual input or auto-generated)
- User description
- Permission settings (read, write, execute, admin)
- Expiration time (optional)

#### Method 2: Command Line Creation
```bash
# From project root directory
# Create regular user
python GUI/create_user.py -u alice -k alice123 -d "Alice user"

# Create administrator user
python GUI/create_user.py -u admin2 -k admin456 -p read write execute admin

# Create temporary user (expires in 30 days)
python GUI/create_user.py -u temp -k temp123 -e 30
```

#### Method 3: List Existing Users
```bash
# From project root directory
python GUI/create_user.py --list
```

### Permission Description
- **read**: Read permission, can view workspaces and files
- **write**: Write permission, can upload files and modify content
- **execute**: Execute permission, can run tasks and execute commands
- **admin**: Administrator permission, has all permissions

### Account Management File
User authentication information is stored in: `config/authorized_keys.json`

This file contains:
- Username and description information
- SHA256 hash values of API Keys (no plaintext storage)
- User permission lists
- Creation time and expiration time
- Account enabled status

> 🔒 **Security Feature**: The system only stores hash values of API Keys, not plaintext passwords, ensuring account security.

## 🔧 Configuration

### Environment Requirements
- Python 3.8+
- Flask
- Flask-SocketIO
- Other dependencies see requirements.txt

### Configuration File
The GUI reads the `config/config.txt` configuration from the main directory:
- `language`: Interface language (zh/en)
- `gui_default_data_directory`: GUI data directory path 