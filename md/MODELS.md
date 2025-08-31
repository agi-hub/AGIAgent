# AGI Agent Model Selection Guide

[**中文**](MODELS_zh.md) | **English**

AGI Agent supports multiple AI models. This guide helps you choose the most suitable model based on your needs and budget.

## 🌟 Recommended Models

### Claude Sonnet 4 (⭐ Highly Recommended)
**Suitable for: Complex tasks requiring high accuracy and detailed responses**

- ✅ **Advantages**:
  - Extremely high intelligence and understanding capability
  - Excellent code generation quality
  - Detailed responses with deep analysis
  - Outstanding tool calling capabilities
- ❌ **Disadvantages**:
  - Relatively high price
  - Medium response speed
  - Occasionally overly cautious behavior
- 💰 **Price Level**: $$$$
- 🎯 **Best Use Cases**:
  - Complex code architecture design
  - Detailed technical analysis
  - Advanced problem solving
  - Multi-round complex tasks

**Configuration Example:**
```bash
python agia.py --model claude-3-5-sonnet-20241022 --api-key your_key -r "Your task"
```

### OpenAI GPT-4 Turbo
**Suitable for: Users needing fast and reliable performance**

- ✅ **Advantages**:
  - Fast response speed
  - High accuracy
  - Stable tool calling
  - Complete ecosystem
- ❌ **Disadvantages**:
  - High price (but cheaper than Claude)
  - Sometimes brief responses
- 💰 **Price Level**: $$$
- 🎯 **Best Use Cases**:
  - General development tasks
  - Rapid iterative development
  - Real-time interactive scenarios
  - Balanced performance needs

**Configuration Example:**
```bash
python agia.py --model gpt-4-turbo --api-key your_key -r "Your task"
```

### DeepSeek V3 (💰 Best Value)
**Suitable for: Users focusing on cost-effectiveness and accuracy**

- ✅ **Advantages**:
  - Extremely economical pricing
  - Accurate code generation
  - Fewer hallucination issues
  - Clear thinking process
- ❌ **Disadvantages**:
  - Relatively concise output
  - Lower creativity
  - Average performance on some advanced tasks
- 💰 **Price Level**: $$
- 🎯 **Best Use Cases**:
  - Code optimization and refactoring
  - Bug fixes
  - Direct implementation tasks
  - Budget-limited projects

**Configuration Example:**
```bash
python agia.py --model deepseek-chat --api-base https://api.deepseek.com --api-key your_key -r "Your task"
```

### Kimi K2 (🚀 Domestic Excellence)
**Suitable for: Users needing Chinese optimization and long context**

- ✅ **Advantages**:
  - Strong Chinese understanding capability
  - Ultra-long context support
  - Reasonable pricing
  - Optimized for Chinese development scenarios
- ❌ **Disadvantages**:
  - Relatively weak international support
  - Average performance on some English tasks
- 💰 **Price Level**: $$$
- 🎯 **Best Use Cases**:
  - Chinese project development
  - Large document processing
  - Long conversation tasks
  - Localization needs

**Configuration Example:**
```bash
python agia.py --model kimi --api-base https://api.moonshot.cn/v1 --api-key your_key -r "Your task"
```

### Qwen2.5-7B-Instruct (🆓 Free Trial)
**Suitable for: Learning trials and simple tasks**

- ✅ **Advantages**:
  - Completely free to use
  - Good Chinese support
  - Basic task processing capability
  - Quick response
- ❌ **Disadvantages**:
  - Limited intelligence level
  - Average performance on complex tasks
  - Weak tool calling capability
- 💰 **Price Level**: FREE
- 🎯 **Best Use Cases**:
  - Learning and experimentation
  - Simple code generation
  - Basic task processing
  - Zero budget scenarios

**Configuration Example:**
```bash
python agia.py --model Qwen/Qwen2.5-7B-Instruct --api-base https://api.siliconflow.cn/v1 --api-key your_free_key -r "Your task"
```

## 📊 Model Comparison Table

| Model | Intelligence | Speed | Chinese Support | Cost | Best Use |
|-------|-------------|-------|----------------|------|----------|
| Claude Sonnet 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰💰💰 | Complex projects |
| GPT-4 Turbo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰 | General development |
| DeepSeek V3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰 | Budget projects |
| Kimi K2 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Chinese projects |
| Qwen2.5-7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Simple tasks |

## 🎯 Selection Recommendations

### Choose by Project Type

#### 🏢 Enterprise Projects
**Recommended: Claude Sonnet 4 or GPT-4 Turbo**
- Need high-quality code generation
- Require detailed technical analysis
- Relatively sufficient budget

#### 💼 Commercial Projects
**Recommended: DeepSeek V3 or Kimi K2**
- Balance cost and performance
- Suitable for medium complexity tasks
- Excellent cost-effectiveness

#### 🎓 Learning and Experimentation
**Recommended: Qwen2.5-7B or DeepSeek V3**
- Limited budget or free
- Suitable for learning programming
- Simple task processing

#### 🇨🇳 Chinese Projects
**Recommended: Kimi K2 or DeepSeek V3**
- Excellent Chinese understanding
- Good localization support
- Fits domestic usage habits

### Choose by Budget

#### High Budget (>$100/month)
1. **Claude Sonnet 4** - Highest quality
2. **GPT-4 Turbo** - Speed and quality balance

#### Medium Budget ($20-100/month)
1. **DeepSeek V3** - Best cost-effectiveness
2. **Kimi K2** - First choice for Chinese projects

#### Low Budget/Free
1. **Qwen2.5-7B** - Completely free
2. **DeepSeek V3** - Extremely low cost

## ⚙️ Configuration Guide

### Configuration File Settings

Configure your chosen model in `config/config.txt`:

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

# Qwen2.5-7B (Free)
api_key=your_siliconflow_key
api_base=https://api.siliconflow.cn/v1
model=Qwen/Qwen2.5-7B-Instruct
```

### Command Line Configuration

You can also specify models directly via command line:

```bash
# Temporarily use different models
python agia.py --model MODEL_NAME --api-key YOUR_KEY --api-base API_BASE -r "Task description"
```

## 🔧 Optimization Recommendations

### Performance Optimization

#### High-end Models (Claude/GPT-4)
```ini
truncation_length=15000
summary_trigger_length=120000
summary_max_length=8000
```

#### Budget Models (DeepSeek/Kimi)
```ini
truncation_length=10000
summary_trigger_length=80000
summary_max_length=5000
```

#### Free Models (Qwen)
```ini
truncation_length=6000
summary_trigger_length=50000
summary_max_length=3000
```

### Tool Calling Optimization

**Models supporting native tool calling:**
- Claude Sonnet 4
- GPT-4 Turbo
- DeepSeek V3

**Models requiring chat-based tool calling:**
- Some local models
- Early version models

```ini
# Auto-detect or manual setting
Tool_calling_format=True  # Recommended to keep default
```

## 🚨 Common Issues

### 1. Difficulty Choosing Models
**Recommended Process:**
1. Clarify budget range
2. Determine project complexity
3. Consider language preference (Chinese/English)
4. Start with recommended models for trial

### 2. API Configuration Issues
- Ensure API key is valid
- Check api_base address
- Verify model name is correct

### 3. Unsatisfactory Performance
- Try adjusting truncation parameters
- Check if task description is clear
- Consider upgrading to higher-end models

### 4. Cost Control
- Set reasonable truncation length
- Enable summary features
- Choose budget models

## 🔄 Model Switching

AGI Agent supports switching models anytime without restarting tasks:

```bash
# Current task using DeepSeek
python agia.py --model deepseek-chat -r "Start task"

# Switch to Claude when needing higher quality
python agia.py --model claude-3-5-sonnet-20241022 -c  # Continue previous task
```

Choosing the right model is key to successfully using AGI Agent. It's recommended to start with cost-effective models and adjust gradually based on actual needs. 