/**
 * Inline Scripts - Extracted from index.html
 * This file contains JavaScript code that was previously embedded inline in the HTML template
 */

// ============================================
// Theme Initialization
// ============================================
(function() {
    // 从localStorage读取保存的主题，默认为light
    const savedTheme = localStorage.getItem('theme') || 'light';
    // 立即设置主题属性，避免页面加载时的闪烁
    document.documentElement.setAttribute('data-theme', savedTheme);
    // 立即设置html和body的背景色，避免闪屏（在CSS加载前就应用）
    const bgColor = savedTheme === 'light' ? '#ffffff' : '#1e1e1e';
    document.documentElement.style.backgroundColor = bgColor;
    // 如果body已存在，立即设置；否则在DOMContentLoaded时设置
    if (document.body) {
        document.body.style.backgroundColor = bgColor;
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            document.body.style.backgroundColor = bgColor;
        });
    }
})();

// ============================================
// MathJax Configuration
// ============================================

// MathJax配置
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
    },
    startup: {
        ready() {
            MathJax.startup.defaultReady();
            window.mathJaxAvailable = true;
        }
    }
};

// MathJax加载状态
window.mathJaxAvailable = false;
window.mathJaxLoadAttempted = false;

// 保护单独的$符号，避免被误识别为数学公式
function protectLoneDollarSigns(htmlContent) {
    if (!htmlContent) return htmlContent;

    // 创建一个临时标记来保护已存在的HTML标签和代码块
    const placeholders = [];
    let placeholderIndex = 0;

    // 保护代码块和pre标签中的内容
    htmlContent = htmlContent.replace(/<pre[^>]*>[\s\S]*?<\/pre>/gi, function(match) {
        const placeholder = `__CODE_BLOCK_${placeholderIndex}__`;
        placeholders[placeholderIndex] = match;
        placeholderIndex++;
        return placeholder;
    });

    // 保护行内代码
    htmlContent = htmlContent.replace(/<code[^>]*>[\s\S]*?<\/code>/gi, function(match) {
        const placeholder = `__CODE_INLINE_${placeholderIndex}__`;
        placeholders[placeholderIndex] = match;
        placeholderIndex++;
        return placeholder;
    });

    // 保护已存在的数学公式标记（MathJax处理后的）
    htmlContent = htmlContent.replace(/<span[^>]*class="[^"]*math[^"]*"[^>]*>[\s\S]*?<\/span>/gi, function(match) {
        const placeholder = `__MATH_SPAN_${placeholderIndex}__`;
        placeholders[placeholderIndex] = match;
        placeholderIndex++;
        return placeholder;
    });

    // 保护块级数学公式
    htmlContent = htmlContent.replace(/<div[^>]*class="[^"]*math[^"]*"[^>]*>[\s\S]*?<\/div>/gi, function(match) {
        const placeholder = `__MATH_DIV_${placeholderIndex}__`;
        placeholders[placeholderIndex] = match;
        placeholderIndex++;
        return placeholder;
    });

    // 先保护货币符号模式：$后面直接跟数字（如 $94,189, $1, $100等）
    // 使用span包裹并添加tex2jax_ignore类，防止MathJax处理
    const protectedCurrency = [];
    let currencyIndex = 0;

    // 匹配货币符号：$后面跟数字，可能包含逗号、小数点等
    // 模式：$数字（可能包含逗号、小数点），后面不能是$、数字或字母（避免匹配到公式或变量）
    // 使用负向前瞻确保后面是空格、标点符号或结束
    // 注意：要匹配完整的金额，包括可能的小数部分
    htmlContent = htmlContent.replace(/\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)(?![$\d\w])/g, function(match, amount) {
        const placeholder = `__CURRENCY_${currencyIndex}__`;
        // 使用span包裹并添加tex2jax_ignore类，防止MathJax处理
        // 在tex2jax_ignore类中，可以直接使用$符号，MathJax不会处理
        protectedCurrency[currencyIndex] = `<span class="tex2jax_ignore">$` + amount + `</span>`;
        currencyIndex++;
        return placeholder;
    });

    // 保护块级公式 $$...$$
    const protectedFormulas = [];
    let formulaIndex = 0;

    htmlContent = htmlContent.replace(/\$\$[\s\S]*?\$\$/g, function(match) {
        const placeholder = `__FORMULA_BLOCK_${formulaIndex}__`;
        protectedFormulas[formulaIndex] = match;
        formulaIndex++;
        return placeholder;
    });

    // 保护行内公式 $...$，使用更严格的匹配规则
    // 要求：$后面不能是空格，公式内容必须在一行内，且包含数学相关字符
    htmlContent = htmlContent.replace(/\$([^\s$\n][^$\n]{0,200}?)\$/g, function(match, formula) {
        // 检查是否是有效的数学公式
        // 必须包含数学符号、希腊字母、或常见的数学表达式模式
        const mathPattern = /[a-zA-Zα-ωΑ-Ω+\-*/=<>≤≥≠≈∑∏∫∞±√^_()\[\]{}]|\\[a-zA-Z]+|frac|sqrt|sum|prod|int|lim|sin|cos|tan|log|ln|exp/;
        if (mathPattern.test(formula) && formula.trim().length > 0) {
            const placeholder = `__FORMULA_INLINE_${formulaIndex}__`;
            protectedFormulas[formulaIndex] = match;
            formulaIndex++;
            return placeholder;
        }
        // 如果不是有效的数学公式，转义两个$符号
        return '<span class="tex2jax_ignore">&#36;' + formula + '&#36;</span>';
    });

    // 转义所有剩余的单独$符号（不在公式中的），并用span包裹防止MathJax处理
    // 注意：此时货币符号已经被占位符替换，所以不会影响货币符号
    htmlContent = htmlContent.replace(/\$/g, '<span class="tex2jax_ignore">&#36;</span>');

    // 恢复被保护的货币符号
    for (let i = currencyIndex - 1; i >= 0; i--) {
        htmlContent = htmlContent.replace(`__CURRENCY_${i}__`, protectedCurrency[i]);
    }

    // 恢复被保护的公式
    for (let i = formulaIndex - 1; i >= 0; i--) {
        htmlContent = htmlContent.replace(`__FORMULA_INLINE_${i}__`, protectedFormulas[i]);
        htmlContent = htmlContent.replace(`__FORMULA_BLOCK_${i}__`, protectedFormulas[i]);
    }

    // 恢复被保护的其他内容
    for (let i = placeholderIndex - 1; i >= 0; i--) {
        htmlContent = htmlContent.replace(`__CODE_BLOCK_${i}__`, placeholders[i]);
        htmlContent = htmlContent.replace(`__CODE_INLINE_${i}__`, placeholders[i]);
        htmlContent = htmlContent.replace(`__MATH_SPAN_${i}__`, placeholders[i]);
        htmlContent = htmlContent.replace(`__MATH_DIV_${i}__`, placeholders[i]);
    }

    return htmlContent;
}

// Fallback: 简单的数学公式文本替换
function fallbackMathRendering(element) {
    if (!element) return;

    // 先保护单独的$符号
    let content = protectLoneDollarSigns(element.innerHTML);

    // 简单的LaTeX到文本的转换
    const mathReplacements = {
        // 希腊字母
        '\\\\alpha': 'α', '\\\\beta': 'β', '\\\\gamma': 'γ', '\\\\delta': 'δ',
        '\\\\epsilon': 'ε', '\\\\theta': 'θ', '\\\\lambda': 'λ', '\\\\mu': 'μ',
        '\\\\pi': 'π', '\\\\sigma': 'σ', '\\\\phi': 'φ', '\\\\omega': 'ω',

        // 数学符号
        '\\\\sum': '∑', '\\\\prod': '∏', '\\\\int': '∫',
        '\\\\infty': '∞', '\\\\pm': '±', '\\\\mp': '∓',
        '\\\\leq': '≤', '\\\\geq': '≥', '\\\\neq': '≠',
        '\\\\approx': '≈', '\\\\equiv': '≡',

        // 上下标简化处理
        '\\\\^\\{([^}]+)\\}': '^($1)',
        '_\\{([^}]+)\\}': '_($1)',

        // 分数简化
        '\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}': '($1)/($2)',

        // 平方根
        '\\\\sqrt\\{([^}]+)\\}': '√($1)',

        // 移除多余的反斜杠和花括号
        '\\\\\\\\': '\\\\',
        '\\{': '',
        '\\}': ''
    };

    // 处理行内数学公式 $...$
    content = content.replace(/\$([^$]+)\$/g, function(match, formula) {
        let processed = formula;
        for (let pattern in mathReplacements) {
            processed = processed.replace(new RegExp(pattern, 'g'), mathReplacements[pattern]);
        }
        return '<span style="font-style: italic; color: #0066cc; background: #f0f8ff; padding: 1px 3px; border-radius: 2px;" title="">' + processed + '</span>';
    });

    // 处理块级数学公式 $$...$$
    content = content.replace(/\$\$([^$]+)\$\$/g, function(match, formula) {
        let processed = formula;
        for (let pattern in mathReplacements) {
            processed = processed.replace(new RegExp(pattern, 'g'), mathReplacements[pattern]);
        }
        return '<div style="text-align: center; font-style: italic; color: #0066cc; background: #f0f8ff; padding: 8px; margin: 10px 0; border-radius: 4px; border-left: 3px solid #0066cc;" title="">' + processed + '</div>';
    });

    element.innerHTML = content;
}

// 智能数学渲染函数
function renderMath(element) {
    if (!element) return Promise.resolve();

    // 在MathJax处理前，先保护单独的$符号
    element.innerHTML = protectLoneDollarSigns(element.innerHTML);

    if (window.mathJaxAvailable && typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        // 使用MathJax渲染
        return MathJax.typesetPromise([element]).catch(function (err) {
            fallbackMathRendering(element);
        });
    } else {
        // 使用fallback渲染
        fallbackMathRendering(element);
        return Promise.resolve();
    }
}

// 检测MathJax加载状态
function checkMathJaxStatus() {
    setTimeout(function() {
        if (!window.mathJaxAvailable && window.mathJaxLoadAttempted) {
            // 静默处理，不显示用户提示
        }
    }, 3000); // 3秒后检查
}

// ============================================
// Model Configuration Functions
// ============================================

let allModelConfigs = [];
let customModelConfig = null;

// 加载模型配置
function loadModelConfigs() {
    // 获取I18N对象用于多语言支持
    const I18N = window.I18N || {};

    // 从服务器获取模型配置列表
    // 添加超时控制（30秒）
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    
    fetch('/api/gui-configs', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success && data.configs) {
            allModelConfigs = data.configs;

            // 获取模型选择框
            const modelSelect = document.getElementById('modelSelect');
            if (!modelSelect) {
                console.error('Model select element not found');
                return;
            }

            // 清空现有选项，保留加载中选项作为第一个
            modelSelect.innerHTML = '';

            // 添加"自定义"选项
            const customOption = document.createElement('option');
            customOption.value = 'custom';
            customOption.textContent = I18N.custom_config_option || '自定义配置';
            modelSelect.appendChild(customOption);

            // 确定默认选择的配置ID
            let defaultConfigId = null;
            if (data.current_model && data.current_api_base) {
                // 查找匹配当前激活配置的选项
                const currentConfigId = `${data.current_model}__${data.current_api_base}`;
                const matchingConfig = data.configs.find(c => c.value === currentConfigId);
                if (matchingConfig) {
                    defaultConfigId = currentConfigId;
                }
            }

            // 如果没有找到匹配的配置，使用第一个非custom配置
            if (!defaultConfigId) {
                const firstNonCustom = data.configs.find(c => c.value !== 'custom');
                if (firstNonCustom) {
                    defaultConfigId = firstNonCustom.value;
                } else if (data.configs.length > 0) {
                    defaultConfigId = data.configs[0].value;
                }
            }

            // 添加所有配置选项
            data.configs.forEach((config, index) => {
                const option = document.createElement('option');
                option.value = config.value;
                option.textContent = config.label;

                // 设置匹配的配置为默认选择
                if (config.value === defaultConfigId) {
                    option.selected = true;
                }

                modelSelect.appendChild(option);
            });

            // 移除之前的事件监听器（如果存在）
            modelSelect.removeEventListener('change', handleModelSelectionChange);
            modelSelect.removeEventListener('click', handleModelSelectClick);

            // 添加模型选择变更事件监听器
            modelSelect.addEventListener('change', handleModelSelectionChange);

            // 添加点击事件监听器来处理重复选择自定义选项的情况
            modelSelect.addEventListener('click', handleModelSelectClick);
        } else {
            // 如果没有配置或加载失败，显示默认选项
            allModelConfigs = [];
            modelSelect.innerHTML = '<option value="" disabled selected>' + (I18N && I18N.lang === 'zh' ? '无可用配置' : 'No configurations available') + '</option>';
            console.warn('No model configurations found or loading failed');
        }
    })
    .catch(error => {
        clearTimeout(timeoutId);
        console.error('Error loading model configurations:', error);
        allModelConfigs = [];
        
        // 获取模型选择框
        const modelSelect = document.getElementById('modelSelect');
        if (!modelSelect) {
            return;
        }
        
        // 根据错误类型显示不同的错误信息
        let errorMessage = I18N && I18N.lang === 'zh' ? '配置加载失败' : 'Failed to load configurations';
        if (error.name === 'AbortError') {
            errorMessage = I18N && I18N.lang === 'zh' ? '配置加载超时，请刷新页面重试' : 'Configuration loading timeout, please refresh and try again';
        } else if (error.message && error.message.includes('HTTP error')) {
            errorMessage = I18N && I18N.lang === 'zh' ? '配置加载失败，请检查服务器连接' : 'Failed to load configurations, please check server connection';
        }
        
        // 显示错误状态
        modelSelect.innerHTML = '<option value="" disabled selected>' + errorMessage + '</option>';
    });
}

// 处理模型选择变更事件
function handleModelSelectionChange() {
    const modelSelect = document.getElementById('modelSelect');
    const selectedValue = modelSelect.value;

    if (selectedValue === 'custom') {
        // 如果选择了自定义，显示自定义配置对话框
        showCustomConfigDialog();
    }
}

// 处理模型选择框点击事件（用于处理重复选择自定义的情况）
function handleModelSelectClick() {
    // 延迟检查，确保选择框的值已经更新
    setTimeout(() => {
        const modelSelect = document.getElementById('modelSelect');
        const selectedValue = modelSelect.value;

        console.log('Model select clicked, value:', selectedValue);

        if (selectedValue === 'custom') {
            // 如果点击的是自定义选项，显示配置对话框
            // 这样即使用户重复选择自定义选项，也会弹出对话框
            console.log('Showing custom config dialog from click event');
            showCustomConfigDialog();
        }
    }, 50);
}

// 显示自定义配置对话框
function showCustomConfigDialog() {
    const modal = document.getElementById('customConfigModal');
    if (modal) {
        // 清空错误信息
        hideCustomConfigError();

        // 更新对话框标题
        const title = modal.querySelector('.modal-header h3');
        if (title) {
            const hasExistingConfig = customModelConfig && customModelConfig.api_key;
            if (hasExistingConfig) {
                title.innerHTML = '<i class="fas fa-edit"></i> ' + (I18N.custom_config_title || '自定义模型配置') + ' (编辑)';
            } else {
                title.innerHTML = '<i class="fas fa-cog"></i> ' + (I18N.custom_config_title || '自定义模型配置');
            }
        }

        // 如果之前有自定义配置，填充表单
        if (customModelConfig) {
            document.getElementById('customApiKey').value = customModelConfig.api_key || '';
            document.getElementById('customApiBase').value = customModelConfig.api_base || '';
            document.getElementById('customModel').value = customModelConfig.model || '';
            document.getElementById('customMaxTokens').value = customModelConfig.max_tokens || 8192;
        } else {
            // 清空表单
            document.getElementById('customApiKey').value = '';
            document.getElementById('customApiBase').value = '';
            document.getElementById('customModel').value = '';
            document.getElementById('customMaxTokens').value = 8192;
        }

        modal.style.display = 'block';

        // 聚焦到第一个输入框
        const firstInput = customModelConfig ?
            document.getElementById('customApiKey') :
            document.getElementById('customApiKey');
        if (firstInput) {
            setTimeout(() => firstInput.focus(), 100);
        }
    }
}

// 隐藏自定义配置对话框
function hideCustomConfigDialog() {
    const modal = document.getElementById('customConfigModal');
    if (modal) {
        modal.style.display = 'none';

        // 如果用户取消了自定义配置，且没有之前的自定义配置，则重置为第一个选项
        if (!customModelConfig) {
            const modelSelect = document.getElementById('modelSelect');
            if (modelSelect && allModelConfigs.length > 0) {
                modelSelect.value = allModelConfigs[0].value;
            }
        }
    }
}

// 保存自定义配置
async function saveCustomConfig() {
    const apiKey = document.getElementById('customApiKey').value.trim();
    const apiBase = document.getElementById('customApiBase').value.trim();
    const model = document.getElementById('customModel').value.trim();
    let maxTokens = document.getElementById('customMaxTokens').value.trim();

    // 验证必填字段
    if (!apiKey || !apiBase || !model) {
        showCustomConfigError(I18N.custom_config_required || '所有字段都是必填的');
        return;
    }

    // 验证并处理max_tokens
    if (!maxTokens) {
        maxTokens = 8192;
    } else {
        maxTokens = parseInt(maxTokens);
        if (isNaN(maxTokens) || maxTokens <= 0) {
            showCustomConfigError('Max Output Tokens必须是大于0的数字');
            return;
        }
    }

    // 保存自定义配置到JavaScript变量
    customModelConfig = {
        api_key: apiKey,
        api_base: apiBase,
        model: model,
        max_tokens: maxTokens,
        display_name: '自定义'
    };

    console.log('已保存自定义模型配置:', customModelConfig);
    hideCustomConfigDialog();

    // 弹出确认对话框询问是否保存为长期配置
    const confirmMessage = I18N.save_to_config_confirm ||
        '是否将此配置保存到 config/config.txt 作为长期配置？\n\n这将更新配置文件中的默认模型设置。';

    if (confirm(confirmMessage)) {
        try {
            // 调用API保存到config.txt
            const response = await fetch('/api/save-to-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    api_base: apiBase,
                    model: model,
                    max_tokens: maxTokens
                })
            });

            const result = await response.json();

            if (result.success) {
                addMessage('✅ ' + (I18N.save_to_config_success || '配置已成功保存到 config.txt'), 'success');
            } else {
                addMessage('❌ ' + (I18N.save_to_config_failed || '保存到 config.txt 失败') + ': ' + result.error, 'error');
            }
        } catch (error) {
            console.error('保存配置到config.txt时出错:', error);
            addMessage('❌ ' + (I18N.save_to_config_error || '保存到 config.txt 时发生错误') + ': ' + error.message, 'error');
        }
    }
}

// 显示自定义配置错误信息
function showCustomConfigError(message) {
    const errorDiv = document.getElementById('customConfigError');
    const errorText = document.getElementById('customConfigErrorText');
    if (errorDiv && errorText) {
        errorText.textContent = message;
        errorDiv.style.display = 'block';
    }
}

// 隐藏自定义配置错误信息
function hideCustomConfigError() {
    const errorDiv = document.getElementById('customConfigError');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// 根据选择的配置ID获取配置信息（异步，需要从服务器获取敏感信息）
async function getSelectedModelConfig() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect || !modelSelect.value) {
        return null;
    }

    const selectedValue = modelSelect.value;

    // 如果选择的是自定义配置
    if (selectedValue === 'custom') {
        return customModelConfig;
    }

    // 查找内置配置的基本信息
    const config = allModelConfigs.find(c => c.value === selectedValue);

    if (!config) {
        console.error('Selected model configuration not found:', selectedValue);
        return null;
    }

    // 对于内置配置，需要从服务器获取api_key和api_base
    try {
        const response = await fetch('/api/get-model-config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                config_id: selectedValue
            })
        });

        const result = await response.json();

        if (result.success && result.config) {
            return {
                value: result.config.value,
                model: result.config.model,
                api_key: result.config.api_key,
                api_base: result.config.api_base,
                max_tokens: result.config.max_tokens || 8192,
                display_name: result.config.display_name || config.display_name
            };
        } else {
            console.error('Failed to get model config details:', result.error);
            // 返回基本信息（不包含敏感信息）
            return {
                value: config.value,
                model: config.model,
                max_tokens: config.max_tokens || 8192,
                display_name: config.display_name
            };
        }
    } catch (error) {
        console.error('Error fetching model config details:', error);
        // 返回基本信息（不包含敏感信息）
        return {
            value: config.value,
            model: config.model,
            max_tokens: config.max_tokens || 8192,
            display_name: config.display_name
        };
    }
}

// 切换密码输入框的可见性
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    const toggleButton = document.getElementById(inputId + 'Toggle') ||
                         document.querySelector(`button[onclick*="${inputId}"]`);

    if (!input || !toggleButton) {
        console.error('Password input or toggle button not found:', inputId);
        return;
    }

    const icon = toggleButton.querySelector('i');

    if (input.type === 'password') {
        // 显示密码
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
        toggleButton.title = (I18N && I18N.lang === 'zh') ? '隐藏密码' : 'Hide password';
    } else {
        // 隐藏密码
        input.type = 'password';
        icon.className = 'fas fa-eye';
        toggleButton.title = (I18N && I18N.lang === 'zh') ? '显示密码' : 'Show password';
    }
}

// 将函数声明为全局函数，确保可以在控制台访问
window.loadModelConfigs = loadModelConfigs;
window.getSelectedModelConfig = getSelectedModelConfig;
window.showCustomConfigDialog = showCustomConfigDialog;
window.hideCustomConfigDialog = hideCustomConfigDialog;
window.saveCustomConfig = saveCustomConfig;
window.togglePasswordVisibility = togglePasswordVisibility;

// 设置自定义配置对话框事件监听器
document.addEventListener('DOMContentLoaded', function() {
    // 保存按钮
    const saveBtn = document.getElementById('customConfigSave');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveCustomConfig);
    }

    // 取消按钮
    const cancelBtn = document.getElementById('customConfigCancel');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', hideCustomConfigDialog);
    }

    // 关闭按钮
    const closeBtn = document.getElementById('customConfigClose');
    if (closeBtn) {
        closeBtn.addEventListener('click', hideCustomConfigDialog);
    }

    // 点击模态框外部关闭
    const modal = document.getElementById('customConfigModal');
    if (modal) {
        window.addEventListener('click', function(event) {
            if (event.target === modal) {
                hideCustomConfigDialog();
            }
        });
    }

    // ESC键关闭
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal && modal.style.display === 'block') {
            hideCustomConfigDialog();
        }
    });
});
