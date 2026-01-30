# AppManager 简化重构方案

## 当前问题分析

### 问题1: 每次调用都创建新实例
```python
def get_user_app_manager(self, session_id: Optional[str] = None) -> AppManager:
    if session_id and session_id in self.user_sessions:
        user_session = self.user_sessions[session_id]
        if user_session.current_app_name is not None:
            return AppManager(app_name=user_session.current_app_name)  # ❌ 每次都创建新实例
    return self.app_manager
```

**问题**：
- 性能问题：每次请求都创建新对象
- 状态不一致：无法保持状态
- 内存浪费：创建大量临时对象

### 问题2: 切换逻辑分散
- `switch_app()` 只更新 `current_app_name`
- 实际的 `AppManager` 在 `get_user_app_manager()` 时才创建
- 存在延迟，可能导致状态不一致

### 问题3: 逻辑复杂
- 需要判断 `current_app_name` 是否为 None
- 需要判断 session 是否存在
- 多个地方需要处理这些逻辑

## 重构方案

### 核心思路
**在 `UserSession` 中直接存储 `AppManager` 实例，而不是只存储 `app_name`**

### 方案优势
1. ✅ **简单直接**：逻辑清晰，易于理解
2. ✅ **性能优化**：避免重复创建对象
3. ✅ **状态一致**：每个用户有独立的实例，状态保持一致
4. ✅ **互不影响**：不同用户完全隔离
5. ✅ **易于维护**：代码更简洁

## 具体实现

### 1. 修改 UserSession 类

```python
class UserSession:
    def __init__(self, session_id, api_key=None, user_info=None):
        # ... 现有代码 ...
        
        # 修改：直接存储 AppManager 实例，而不是只存储 app_name
        self.app_manager = AppManager(app_name=None)  # 默认使用全局默认模式
        
        # 保留 current_app_name 用于向后兼容和日志记录
        self.current_app_name = None
```

### 2. 修改 switch_app 方法

```python
def switch_app(self, app_name: Optional[str], session_id: Optional[str] = None):
    """
    动态切换应用平台
    
    Args:
        app_name: 应用名称（如 'patent'），如果为None则重置为默认模式
        session_id: 会话ID，如果提供则切换指定用户的app，否则切换全局默认app
    """
    if session_id:
        # 会话级切换：直接更新用户的 AppManager 实例
        if session_id in self.user_sessions:
            user_session = self.user_sessions[session_id]
            # 直接创建并更新 AppManager 实例
            user_session.app_manager = AppManager(app_name=app_name)
            user_session.current_app_name = app_name  # 保留用于日志
    else:
        # 全局切换（向后兼容）
        self.app_manager = AppManager(app_name=app_name)
        
        # 更新全局 APP_NAME
        global APP_NAME
        if self.app_manager.is_app_mode():
            APP_NAME = self.app_manager.get_app_name()
        else:
            APP_NAME = "AGI Agent"
        
        # 更新环境变量（向后兼容）
        if app_name:
            os.environ['AGIA_APP_NAME'] = app_name
        else:
            if 'AGIA_APP_NAME' in os.environ:
                del os.environ['AGIA_APP_NAME']
```

### 3. 简化 get_user_app_manager 方法

```python
def get_user_app_manager(self, session_id: Optional[str] = None) -> AppManager:
    """
    根据session_id获取用户专属的AppManager实例
    
    Args:
        session_id: 会话ID，如果为None则返回全局默认AppManager
    
    Returns:
        AppManager实例
    """
    if session_id and session_id in self.user_sessions:
        # 直接返回用户 session 中存储的 AppManager 实例
        return self.user_sessions[session_id].app_manager
    
    # 返回全局默认AppManager（向后兼容）
    return self.app_manager
```

### 4. 修改 GUIInstance 初始化

```python
def __init__(self, app_name: Optional[str] = None):
    # ... 现有代码 ...
    
    # 初始化全局 app manager
    self.app_manager = AppManager(app_name=app_name)
    
    # 更新全局 APP_NAME
    global APP_NAME
    if self.app_manager.is_app_mode():
        APP_NAME = self.app_manager.get_app_name()
```

### 5. 修改 get_user_session 方法（可选优化）

在创建新 session 时，可以继承全局的 app_manager：

```python
def get_user_session(self, session_id, api_key=None):
    """Get or create user session with authentication"""
    # ... 现有认证代码 ...
    
    if session_id not in self.user_sessions:
        # 创建新 session
        user_session = UserSession(session_id, api_key, user_info)
        
        # 可选：继承全局 app_manager（如果需要）
        # user_session.app_manager = AppManager(app_name=self.app_manager.app_name)
        # 或者保持默认的 None（默认模式）
        
        self.user_sessions[session_id] = user_session
        session_type = "guest" if is_guest else "authenticated"
    
    return self.user_sessions[session_id]
```

## 代码对比

### 重构前
```python
# 每次调用都创建新实例
def get_user_app_manager(self, session_id):
    if session_id in self.user_sessions:
        user_session = self.user_sessions[session_id]
        if user_session.current_app_name is not None:
            return AppManager(app_name=user_session.current_app_name)  # 新实例
    return self.app_manager

# 切换时只更新名称
def switch_app(self, app_name, session_id):
    if session_id:
        self.user_sessions[session_id].current_app_name = app_name  # 只更新名称
```

### 重构后
```python
# 直接返回存储的实例
def get_user_app_manager(self, session_id):
    if session_id in self.user_sessions:
        return self.user_sessions[session_id].app_manager  # 直接返回
    return self.app_manager

# 切换时直接更新实例
def switch_app(self, app_name, session_id):
    if session_id:
        user_session = self.user_sessions[session_id]
        user_session.app_manager = AppManager(app_name=app_name)  # 直接更新实例
        user_session.current_app_name = app_name  # 保留用于日志
```

## 迁移步骤

1. **第一步**：修改 `UserSession.__init__()`，添加 `self.app_manager`
2. **第二步**：修改 `switch_app()`，直接更新 `app_manager` 实例
3. **第三步**：简化 `get_user_app_manager()`，直接返回存储的实例
4. **第四步**：测试验证，确保功能正常
5. **第五步**：清理不再需要的代码（可选）

## 向后兼容性

- ✅ 保留 `current_app_name` 属性用于日志和调试
- ✅ 全局 `app_manager` 保持不变
- ✅ API 接口不变
- ✅ 所有现有调用 `get_user_app_manager()` 的地方无需修改

## 性能提升

- **减少对象创建**：从每次请求创建 → 只在切换时创建
- **内存优化**：每个用户只有一个 AppManager 实例
- **响应速度**：直接返回实例，无需判断和创建

## 测试要点

1. ✅ 不同用户可以使用不同的 app，互不影响
2. ✅ 用户切换 app 后，立即生效
3. ✅ 默认模式（app_name=None）正常工作
4. ✅ 全局切换（无 session_id）正常工作
5. ✅ 并发访问时，各用户状态独立
