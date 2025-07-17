# AGI Bot GUI

AGI Bot 的图形用户界面，提供直观便捷的任务执行和文件管理功能。

## 🚀 快速开始

### 启动GUI, 在工程根目录下执行:
```bash
python GUI/app.py
```

启动后访问：`http://localhost:5001`

## 界面示例
<div align="center">
      <img src="../fig/AGIBot_GUI.png" alt="AGI Bot GUI"/>
</div>

## 主要用法

您可以新建或选择一个工作区, 并将需要处理的数据文件通过工作区的上传按钮上传, 写入需求, 并按执行按钮运行, 程序会执行最多50轮迭代, 运行结束后, 可以从工作区看到生成的文件, 此时可以点击下载工作区的按钮进行下载. 运行过程中及结束后,您都可以预览已经产生的文件. 

当选择一个工作区时, 务必将这个工作区点击为蓝色高亮状态.

当任务执行完毕或被中断后, 您可以通过选择工作区并输入提示词继续任务, 但需要注意上一轮的需求及执行过程并没有带入到本次运行.


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

