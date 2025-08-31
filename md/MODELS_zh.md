# AGI Agent 模型选择指南

**中文** | [**English**](MODELS.md)

AGI Agent 支持多种AI模型，本指南帮助您根据需求和预算选择最适合的模型。

## 🌟 推荐模型

### Claude Sonnet 4 (⭐ 强烈推荐)
**适合：需要高准确性和详细回答的复杂任务**

- ✅ **优点**：
  - 智能程度极高，理解能力强
  - 代码生成质量优秀
  - 回复详细，分析深入
  - 工具调用能力突出
- ❌ **缺点**：
  - 价格相对较高
  - 响应速度中等
  - 偶尔会有过度谨慎的表现
- 💰 **价格等级**：$$$$
- 🎯 **最佳使用场景**：
  - 复杂代码架构设计
  - 详细技术分析
  - 高级问题解决
  - 多轮复杂任务

**配置示例：**
```bash
python agia.py --model claude-3-5-sonnet-20241022 --api-key your_key -r "您的任务"
```

### OpenAI GPT-4 Turbo
**适合：需要快速可靠性能的用户**

- ✅ **优点**：
  - 响应速度快
  - 准确性高
  - 工具调用稳定
  - 生态完善
- ❌ **缺点**：
  - 价格较高（但比Claude便宜）
  - 有时回复较简洁
- 💰 **价格等级**：$$$
- 🎯 **最佳使用场景**：
  - 通用开发任务
  - 快速迭代开发
  - 实时交互场景
  - 平衡性能需求

**配置示例：**
```bash
python agia.py --model gpt-4-turbo --api-key your_key -r "您的任务"
```

### DeepSeek V3 (💰 性价比之选)
**适合：注重成本效益和准确性的用户**

- ✅ **优点**：
  - 价格极其经济
  - 代码生成准确
  - 较少幻觉问题
  - 思维过程清晰
- ❌ **缺点**：
  - 输出相对简洁
  - 创意性稍逊
  - 部分高级任务表现一般
- 💰 **价格等级**：$$
- 🎯 **最佳使用场景**：
  - 代码优化和重构
  - Bug修复
  - 直接实现任务
  - 预算有限的项目

**配置示例：**
```bash
python agia.py --model deepseek-chat --api-base https://api.deepseek.com --api-key your_key -r "您的任务"
```

### Kimi K2 (🚀 国产优选)
**适合：需要中文优化和长上下文的用户**

- ✅ **优点**：
  - 中文理解能力强
  - 超长上下文支持
  - 价格相对合理
  - 对中文开发场景优化
- ❌ **缺点**：
  - 国际化支持相对较弱
  - 部分英文任务表现一般
- 💰 **价格等级**：$$$
- 🎯 **最佳使用场景**：
  - 中文项目开发
  - 大型文档处理
  - 长对话任务
  - 本土化需求

**配置示例：**
```bash
python agia.py --model kimi --api-base https://api.moonshot.cn/v1 --api-key your_key -r "您的任务"
```

### Qwen2.5-7B-Instruct (🆓 免费试用)
**适合：学习试用和简单任务**

- ✅ **优点**：
  - 完全免费使用
  - 中文支持良好
  - 基础任务处理能力
  - 快速响应
- ❌ **缺点**：
  - 智能程度有限
  - 复杂任务表现一般
  - 工具调用能力较弱
- 💰 **价格等级**：FREE
- 🎯 **最佳使用场景**：
  - 学习和实验
  - 简单代码生成
  - 基础任务处理
  - 预算为零的场景

**配置示例：**
```bash
python agia.py --model Qwen/Qwen2.5-7B-Instruct --api-base https://api.siliconflow.cn/v1 --api-key your_free_key -r "您的任务"
```

## 📊 模型对比表

| 模型 | 智能程度 | 响应速度 | 中文支持 | 成本 | 最佳用途 |
|------|---------|---------|---------|------|----------|
| Claude Sonnet 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰💰💰 | 复杂项目 |
| GPT-4 Turbo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰 | 通用开发 |
| DeepSeek V3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰 | 预算项目 |
| Kimi K2 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 | 中文项目 |
| Qwen2.5-7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 免费 | 简单任务 |

## 🎯 选择建议

### 根据项目类型选择

#### 🏢 企业级项目
**推荐：Claude Sonnet 4 或 GPT-4 Turbo**
- 需要高质量代码生成
- 要求详细的技术分析
- 预算相对充足

#### 💼 商业项目
**推荐：DeepSeek V3 或 Kimi K2**
- 平衡成本和性能
- 适合中等复杂度任务
- 性价比优秀

#### 🎓 学习和实验
**推荐：Qwen2.5-7B 或 DeepSeek V3**
- 预算有限或免费
- 适合学习编程
- 简单任务处理

#### 🇨🇳 中文项目
**推荐：Kimi K2 或 DeepSeek V3**
- 中文理解优秀
- 本土化支持好
- 符合国内使用习惯

### 根据预算选择

#### 高预算 (>$100/月)
1. **Claude Sonnet 4** - 最高质量
2. **GPT-4 Turbo** - 速度与质量平衡

#### 中等预算 ($20-100/月)
1. **DeepSeek V3** - 性价比最高
2. **Kimi K2** - 中文项目首选

#### 低预算/免费
1. **Qwen2.5-7B** - 完全免费
2. **DeepSeek V3** - 极低成本

## ⚙️ 配置指南

### 配置文件设置

在 `config/config.txt` 中配置您选择的模型：

```ini
# Claude Sonnet 4
api_key=your_anthropic_key
api_base=https://api.anthropic.com
model=claude-3-5-sonnet-20241022

# GPT-4 Turbo  
api_key=your_openai_key
api_base=https://api.openai.com/v1
model=gpt-4-turbo

# DeepSeek V3
api_key=your_deepseek_key
api_base=https://api.deepseek.com
model=deepseek-chat

# Kimi K2
api_key=your_kimi_key
api_base=https://api.moonshot.cn/v1
model=kimi

# Qwen2.5-7B (免费)
api_key=your_siliconflow_key
api_base=https://api.siliconflow.cn/v1
model=Qwen/Qwen2.5-7B-Instruct
```

### 命令行配置

也可以通过命令行直接指定模型：

```bash
# 临时使用不同模型
python agia.py --model MODEL_NAME --api-key YOUR_KEY --api-base API_BASE -r "任务描述"
```

## 🔧 优化建议

### 性能优化

#### 高端模型 (Claude/GPT-4)
```ini
truncation_length=15000
summary_trigger_length=120000
summary_max_length=8000
```

#### 经济型模型 (DeepSeek/Kimi)
```ini
truncation_length=10000
summary_trigger_length=80000
summary_max_length=5000
```

#### 免费模型 (Qwen)
```ini
truncation_length=6000
summary_trigger_length=50000
summary_max_length=3000
```

### 工具调用优化

**支持原生工具调用的模型：**
- Claude Sonnet 4
- GPT-4 Turbo
- DeepSeek V3

**需要聊天式工具调用的模型：**
- 部分本地模型
- 早期版本模型

```ini
# 自动检测或手动设置
Tool_calling_format=True  # 推荐保持默认
```

## 🚨 常见问题

### 1. 模型选择困难
**建议流程：**
1. 明确预算范围
2. 确定项目复杂度
3. 考虑语言偏好（中文/英文）
4. 从推荐模型开始试用

### 2. API配置问题
- 确保API密钥有效
- 检查api_base地址
- 验证模型名称正确

### 3. 性能不满意
- 尝试调整truncation参数
- 检查任务描述是否清晰
- 考虑升级到更高端模型

### 4. 成本控制
- 设置合理的截断长度
- 启用摘要功能
- 选择经济型模型

## 🔄 模型切换

AGI Agent 支持随时切换模型，无需重新开始任务：

```bash
# 当前任务使用 DeepSeek
python agia.py --model deepseek-chat -r "开始任务"

# 需要更高质量时切换到 Claude
python agia.py --model claude-3-5-sonnet-20241022 -c  # 继续之前的任务
```

选择合适的模型是成功使用AGI Agent的关键。建议从性价比高的模型开始，根据实际需求逐步调整。 