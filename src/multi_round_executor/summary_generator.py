#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

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
Summary generator for creating intelligent task summaries
"""

from typing import List, Dict, Any
from tools.print_system import print_system, print_current
from config_loader import get_summary_max_length
class SummaryGenerator:
    """Summary generator for creating intelligent task summaries using LLM"""
    
    def __init__(self, executor, detailed_summary: bool = True):
        """
        Initialize summary generator
        
        Args:
            executor: Tool executor instance
            detailed_summary: Whether to generate detailed summary
        """
        self.executor = executor
        self.detailed_summary = detailed_summary
        
        # Get language configuration from executor
        if hasattr(executor, 'language'):
            self.language = executor.language
        else:
            # Fallback to default language
            try:
                from config_loader import get_language
                self.language = get_language()
            except ImportError:
                self.language = 'en'  # Default to English
        
    def generate_smart_summary(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Use large model to generate intelligent summary of prerequisite tasks
        
        Args:
            completed_tasks: List of completed tasks
            
        Returns:
            Detailed intelligent summary text
        """
        if not completed_tasks:
            return ""
        
        # Choose summary mode based on configuration
        if not self.detailed_summary:
            return self._generate_simple_summary(completed_tasks)
        
        # Build detailed summary prompt
        summary_prompt = self._build_detailed_prompt(completed_tasks)
        
        try:
            print_current("🧠 Using large model to generate detailed summary of prerequisite tasks...")
            
            # Use large model to generate summary
            if self.executor.is_claude:
                smart_summary = self._generate_with_claude(summary_prompt)
            else:
                smart_summary = self._generate_with_openai(summary_prompt)
            
            # print_current(f"✅ Detailed summary generation completed (length: {len(smart_summary)} characters)")
            return smart_summary
            
        except Exception as e:
            print_current(f"⚠️ Failed to generate intelligent summary: {e}, using enhanced basic summary")
            return self._generate_enhanced_basic_summary(completed_tasks)
    
    def _build_detailed_prompt(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Build detailed summary prompt
        
        Args:
            completed_tasks: List of completed tasks
            
        Returns:
            Summary prompt text
        """
        # Determine language instruction based on configuration
        if self.language == 'zh':
            print('language is Chinese')
            language_instruction = "**重要语言要求：请使用中文生成总结，保持专业性和准确性**"
        else:
            language_instruction = "**Important Language Requirement: Please generate the summary in English, maintaining professionalism and accuracy**"
        
        summary_prompt = f"""{language_instruction}

Please generate a detailed and comprehensive summary for the following completed tasks, focusing on retaining the following information:

**Information types that must be retained:**
1. Core objectives and implementation plans of tasks
2. Tools used and technical decisions (such as command line tools, API calls, etc.)
3. Technical problems solved and solutions adopted
4. Files created or modified and their functions
5. Important configuration information and parameter settings
6. Errors encountered and solutions
7. Workflows and steps
8. Key learnings and discoveries
9. Context information valuable for subsequent tasks
10. Project structure and architectural decisions

**Please do not include:**
- Complete code content (key function names and class names can be mentioned)
- Repetitive execution details

Detailed information of completed tasks:

"""
        
        for i, task_result in enumerate(completed_tasks, 1):
            task_info = self._extract_task_info(task_result, i)
            summary_prompt += f"""
=== Task {task_info['id']}: {task_info['name']} ===
Task Description: {task_info['description']}

Execution Rounds: {task_info['rounds_count']}

Tools and Operations Used:
{chr(10).join([f"- {tool}" for tool in task_info['tools_used']]) if task_info['tools_used'] else "- No special tools"}

Technical Decisions and Solutions:
{chr(10).join([f"- {decision}" for decision in task_info['technical_decisions']]) if task_info['technical_decisions'] else "- No special decisions"}

File Operations:
{chr(10).join([f"- {file_op}" for file_op in task_info['files_created']]) if task_info['files_created'] else "- No file operations"}

Key Results:
{chr(10).join([f"- {result}" for result in task_info['key_results']]) if task_info['key_results'] else "- No special results"}

Problems Encountered:
{chr(10).join([f"- {error}" for error in task_info['errors_encountered']]) if task_info['errors_encountered'] else "- No significant problems"}

"""
        
        # Add language-specific ending
        if self.language == 'zh':
            summary_prompt += f"""
请基于上述信息生成{get_summary_max_length()//2}-{get_summary_max_length()}字的详细总结，确保：
1. 保留对后续任务有价值的所有信息
2. 突出技术选择、工作流程和解决方案
3. 包含重要的上下文信息和经验总结
4. 组织清晰，便于后续任务的理解和参考
5. 避免冗余，同时保持信息完整性
6. **重要：请使用中文生成总结**
"""
        else:
            summary_prompt += f"""
Please generate a detailed summary of {get_summary_max_length()//2}-{get_summary_max_length()} words based on the above information, ensuring:
1. Retain all information valuable for reference in subsequent tasks
2. Highlight technical choices, workflows, and solutions
3. Include important context information and experience summaries
4. Organize clearly for easy understanding and reference in subsequent tasks
5. Avoid redundancy while maintaining information completeness
6. **Important: Please generate the summary in English**
"""
        
        return summary_prompt
    
    def _extract_task_info(self, task_result: Dict[str, Any], task_index: int) -> Dict[str, Any]:
        """
        Extract structured information from task result
        
        Args:
            task_result: Task execution result
            task_index: Task index for fallback
            
        Returns:
            Structured task information
        """
        task_id = task_result.get('task_id', task_index)
        task_name = task_result.get('task_name', f'Task{task_index}')
        task_desc = task_result.get('task_description', 'No description')
        history = task_result.get('history', [])
        
        # Extract various information types
        tools_used = []
        technical_decisions = []
        errors_encountered = []
        files_created = []
        key_results = []
        
        for record in history:
            if 'result' in record and not record.get('error'):
                result = record['result']
                
                # Extract tool usage information
                if 'Tool Execution Result' in result or 'Tool Execution Result' in result:
                    separator = 'Tool Execution Result' if 'Tool Execution Result' in result else 'Tool Execution Result'
                    parts = result.split(separator)
                    if len(parts) > 1:
                        tool_part = parts[1][:500]
                        tools_used.append(tool_part.strip())
                
                # Extract file operation information
                if 'edit_file' in result or 'create' in result.lower() or 'Create' in result:
                    if len(result) < 800:
                        files_created.append(result[:300])
                
                # Extract technical decisions
                if any(keyword in result.lower() for keyword in ['decide', 'choose', 'adopt', 'use', 'config', 'setup', 'solve', 'implement', 'Decide', 'Select', 'Adopt', 'Use', 'Configure', 'Set', 'Solve', 'Implement']):
                    if len(result) < 1000:
                        technical_decisions.append(result[:400])
                
                # Extract key results
                if len(result) < 600 and not any(code_indicator in result for code_indicator in ['```', 'def ', 'class ', 'import ', 'function']):
                    key_results.append(result[:300])
            
            elif record.get('error'):
                errors_encountered.append(record['error'][:200])
        
        # Remove duplicates and limit quantity
        return {
            'id': task_id,
            'name': task_name,
            'description': task_desc,
            'rounds_count': len([h for h in history if 'round' in h]),
            'tools_used': list(dict.fromkeys(tools_used))[:3],
            'technical_decisions': list(dict.fromkeys(technical_decisions))[:3],
            'files_created': list(dict.fromkeys(files_created))[:5],
            'key_results': list(dict.fromkeys(key_results))[:3],
            'errors_encountered': list(dict.fromkeys(errors_encountered))[:2]
        }
    
    def _generate_with_claude(self, prompt: str) -> str:
        """Generate summary using Claude API"""
        import datetime
        
        current_date = datetime.datetime.now()
        
        system_prompt = f"""You are a professional technical project summary assistant, skilled at extracting and retaining key technical information, workflows, and experience summaries to provide valuable references for subsequent tasks.

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}"""
        
        # print_current("🔄 Starting summary generation...")
        response = self.executor.client.messages.create(
            model=self.executor.model,
            max_tokens=self.executor._get_max_tokens_for_model(self.executor.model),
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        smart_summary = response.content[0].text
        # print_current("📋 Intelligent summary generated")
        
        return smart_summary
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate summary using OpenAI API"""
        import datetime
        
        current_date = datetime.datetime.now()
        
        system_prompt = f"""You are a professional technical project summary assistant, skilled at extracting and retaining key technical information, workflows, and experience summaries to provide valuable references for subsequent tasks.

**Current Date Information**:
- Current Date: {current_date.strftime('%Y-%m-%d')}
- Current Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}"""
        
        # print_current("🔄 Starting summary generation...")
        response = self.executor.client.chat.completions.create(
            model=self.executor.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.executor._get_max_tokens_for_model(self.executor.model),
            temperature=0.2,
            top_p=0.9
        )
        
        if response.choices and len(response.choices) > 0:
            smart_summary = response.choices[0].message.content
        else:
            smart_summary = "Summary generation failed: API returned empty response"
            print_current("⚠️ Warning: OpenAI API returned empty choices list")
        # print_current("📋 Intelligent summary generated")
        
        return smart_summary
    
    def _generate_simple_summary(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Generate simplified task summary
        
        Args:
            completed_tasks: List of completed tasks
            
        Returns:
            Simplified summary text
        """
        if not completed_tasks:
            return ""
        
        simple_summary = f"Basic summary of {len(completed_tasks)} completed tasks:\n\n"
        
        for i, task_result in enumerate(completed_tasks, 1):
            task_id = task_result.get('task_id', i)
            task_name = task_result.get('task_name', f'Task{i}')
            task_desc = task_result.get('task_description', 'No description')
            
            # Extract basic information
            history = task_result.get('history', [])
            rounds_count = len([h for h in history if 'round' in h])
            
            # Check if successfully completed
            has_success = any('Success' in h.get('result', '') or 'Complete' in h.get('result', '') 
                            for h in history if 'result' in h)
            
            # Extract key files
            files_mentioned = []
            for h in history:
                if 'result' in h:
                    import re
                    file_matches = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{1,4}', h['result'])
                    files_mentioned.extend(file_matches[:3])
            
            files_mentioned = list(dict.fromkeys(files_mentioned))[:5]
            
            simple_summary += f"Task{task_id} - {task_name}:\n"
            simple_summary += f"  Description: {task_desc[:100]}{'...' if len(task_desc) > 100 else ''}\n"
            simple_summary += f"  Execution rounds: {rounds_count}\n"
            simple_summary += f"  Status: {'Success' if has_success else 'Completed'}\n"
            if files_mentioned:
                simple_summary += f"  Related files: {', '.join(files_mentioned)}\n"
            simple_summary += "\n"
        
        return simple_summary
    
    def _generate_enhanced_basic_summary(self, completed_tasks: List[Dict[str, Any]]) -> str:
        """Generate enhanced basic summary as fallback"""
        enhanced_summary = f"Completed {len(completed_tasks)} tasks:\n\n"
        
        for i, task_result in enumerate(completed_tasks, 1):
            task_id = task_result.get('task_id', i)
            task_name = task_result.get('task_name', f'Task{i}')
            history = task_result.get('history', [])
            tools_count = len([h for h in history if 'Tool Execution Result' in h.get('result', '') or 'Tool Execution Result' in h.get('result', '')])
            rounds_count = len([h for h in history if 'round' in h])
            
            enhanced_summary += f"Task{task_id} - {task_name}:\n"
            enhanced_summary += f"  Execution rounds: {rounds_count}, Tool calls: {tools_count}\n\n"
        
        return enhanced_summary
    
    def generate_conversation_history_summary(self, conversation_history: List[Dict[str, Any]], latest_tool_result: str = None) -> str:
        """
        Generate a summary of conversation history for context control
        
        Args:
            conversation_history: List of conversation records
            latest_tool_result: The latest tool execution result (should not be summarized)
            
        Returns:
            Summarized conversation history text
        """
        if not conversation_history:
            return ""
        
        # Build summary prompt for conversation history
        summary_prompt = self._build_conversation_summary_prompt(conversation_history, latest_tool_result)
        
        try:
            print_current("🧠 Using large model to generate conversation history summary...")
            
            # Use large model to generate summary
            if self.executor.is_claude:
                conversation_summary = self._generate_with_claude(summary_prompt)
            else:
                conversation_summary = self._generate_with_openai(summary_prompt)
            
            print_current(f"✅ Conversation history summary completed (length: {len(conversation_summary)} characters)")
            return conversation_summary
            
        except Exception as e:
            print_current(f"⚠️ Failed to generate conversation history summary: {e}, using basic summary")
            return self._generate_basic_conversation_summary(conversation_history, latest_tool_result)
    
    def _build_conversation_summary_prompt(self, conversation_history: List[Dict[str, Any]], latest_tool_result: str = None) -> str:
        """
        Build conversation summary prompt
        
        Args:
            conversation_history: List of conversation records
            latest_tool_result: The latest tool execution result
            
        Returns:
            Summary prompt text
        """
        # Get max summary length from executor configuration
        max_summary_length = self.executor.summary_max_length
        
        # Determine language instruction based on configuration
        if self.language == 'zh':
            language_instruction = "**重要语言要求：请使用中文生成总结，保持专业性和准确性**"
        else:
            language_instruction = "**Important Language Requirement: Please generate the summary in English, maintaining professionalism and accuracy**"
        
        # Build language-specific prompt
        if self.language == 'zh':
            summary_prompt = f"""请为以下早期对话历史创建全面总结，以保留对未来交互的重要上下文。此总结将与最近的详细对话轮次结合，因此请重点关注提供背景和基础的早期上下文。

{language_instruction}

**关键 - 必须完全保留：**
1. 用户的原始请求和主要目标
2. 关键技术决策及其推理
3. 错误消息及其已验证的解决方案
4. 重要文件路径、函数名称和配置值
5. 项目结构和架构决策
6. 已建立的成功方法和方法论

**重要 - 详细保留：**
7. 成功使用的工具和命令
8. 重要发现和见解
9. 已识别的技术约束和要求
10. 已建立的工作流程模式

**中等 - 简洁总结：**
11. 中间调试步骤（仅保留关键学习内容）
12. 工具执行详情（仅保留成功结果）
13. 探索性对话（仅保留结论）

**注意：** 最近的对话轮次将单独完整保留，因此请重点关注此总结中提供早期交互基本背景的上下文。

**总结目标长度：** {max_summary_length//2} 到 {max_summary_length} 字符

**需要总结的早期对话历史：**

"""
        else:
            summary_prompt = f"""Please create a comprehensive summary of the following EARLIER conversation history to preserve important context for future interactions. This summary will be combined with recent detailed conversation rounds, so focus on earlier context that provides background and foundation.

{language_instruction}

**CRITICAL - Must preserve completely:**
1. User's original request and main objectives
2. Key technical decisions and their reasoning
3. Error messages and their verified solutions
4. Important file paths, function names, and configuration values
5. Project structure and architectural decisions
6. Successful approaches and methodologies that were established

**IMPORTANT - Preserve with detail:**
7. Tools and commands that worked successfully
8. Important discoveries and insights
9. Technical constraints and requirements identified
10. Workflow patterns that were established

**MODERATE - Summarize concisely:**
11. Intermediate debugging steps (keep only key learnings)
12. Tool execution details (keep only successful outcomes)
13. Exploratory conversations (keep only conclusions)

**Note:** Recent conversation rounds will be preserved separately in full detail, so focus this summary on providing essential background context from earlier interactions.

**Summary target length:** {max_summary_length//2} to {max_summary_length} characters

**Earlier Conversation History to Summarize:**

"""
        
        # Add conversation records, excluding the latest tool result
        for i, record in enumerate(conversation_history):
            if record.get("role") == "user":
                if self.language == 'zh':
                    summary_prompt += f"**用户请求 {i+1}：**\n{record.get('content', '')}\n\n"
                else:
                    summary_prompt += f"**User Request {i+1}:**\n{record.get('content', '')}\n\n"
            elif record.get("role") == "assistant":
                content = record.get('content', '')
                # Skip if this is the latest tool result that shouldn't be summarized
                if latest_tool_result and content == latest_tool_result:
                    continue
                if self.language == 'zh':
                    summary_prompt += f"**助手回复 {i+1}：**\n{record.get('content', '')}\n\n"
                else:
                    summary_prompt += f"**Assistant Response {i+1}:**\n{record.get('content', '')}\n\n"
        
        # Add language-specific ending
        if self.language == 'zh':
            summary_prompt += f"""
请生成大约{max_summary_length}字符的全面总结，捕捉上述对话历史中的所有重要信息。重点关注对继续工作有价值的上下文信息。

**重要：请使用中文生成总结**
"""
        else:
            summary_prompt += f"""
Please generate a comprehensive summary of approximately {max_summary_length} characters that captures all essential information from the conversation history above. Focus on preserving context that would be valuable for continuing the work.

**Important: Please generate the summary in English**
"""
        
        return summary_prompt
    
    def _generate_basic_conversation_summary(self, conversation_history: List[Dict[str, Any]], latest_tool_result: str = None) -> str:
        """
        Generate basic conversation summary as fallback
        
        Args:
            conversation_history: List of conversation records
            latest_tool_result: The latest tool execution result
            
        Returns:
            Basic summary text
        """
        # Use language-appropriate headers
        if self.language == 'zh':
            summary_parts = ["对话总结："]
            user_header = f"\n用户请求 ({len([r for r in conversation_history if r.get('role') == 'user'])}):"
            assistant_header = f"\n助手操作 ({len([r for r in conversation_history if r.get('role') == 'assistant'])}):"
        else:
            summary_parts = ["Conversation Summary:"]
            user_header = f"\nUser Requests ({len([r for r in conversation_history if r.get('role') == 'user'])}):"
            assistant_header = f"\nAssistant Actions ({len([r for r in conversation_history if r.get('role') == 'assistant'])}):"
        
        user_requests = []
        assistant_actions = []
        
        for record in conversation_history:
            if record.get("role") == "user":
                content = record.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                user_requests.append(content)
            elif record.get("role") == "assistant":
                content = record.get('content', '')
                # Skip if this is the latest tool result
                if latest_tool_result and content == latest_tool_result:
                    continue
                if len(content) > 300:
                    content = content[:300] + "..."
                assistant_actions.append(content)
        
        if user_requests:
            summary_parts.append(user_header)
            for i, req in enumerate(user_requests, 1):
                summary_parts.append(f"{i}. {req}")
        
        if assistant_actions:
            summary_parts.append(assistant_header)
            for i, action in enumerate(assistant_actions, 1):
                summary_parts.append(f"{i}. {action}")
        
        return "\n".join(summary_parts)