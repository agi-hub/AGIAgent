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
自定义工具：通用命令执行工具
支持通过 type 参数选择不同的工具类型（game 或 echo）
"""

import random
from typing import Dict, Any, Optional, Tuple
from .print_system import print_current, print_error


class CustomGameTool:
    """
    自定义工具类：通用命令执行工具
    默认实现为12x12棋类游戏
    """
    
    BOARD_SIZE = 12  # 棋盘大小
    WIN_COUNT = 4  # 获胜所需连子数
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        初始化自定义工具
        
        Args:
            workspace_root: 工作空间根目录
        """
        self.workspace_root = workspace_root or ""
        # 游戏状态：棋盘（12x12），'X'表示大模型，'O'表示环境，''表示空
        self.board = [['' for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        # 当前轮到谁：'X'表示大模型，'O'表示环境
        self.current_player = 'X'
        # 游戏是否结束
        self.game_over = False
        # 获胜者：'X'、'O'或None（平局）
        self.winner = None
    
    def _check_winner(self) -> Optional[str]:
        """
        检查是否有获胜者（连成指定数量即获胜）
        
        Returns:
            获胜者：'X'、'O'或None（无获胜者）
        """
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == '':
                    continue
                
                player = self.board[row][col]
                
                # 检查每个方向
                for dr, dc in directions:
                    count = 1  # 当前棋子本身
                    
                    # 检查正方向
                    for i in range(1, self.WIN_COUNT):
                        r, c = row + dr * i, col + dc * i
                        if (0 <= r < self.BOARD_SIZE and 
                            0 <= c < self.BOARD_SIZE and 
                            self.board[r][c] == player):
                            count += 1
                        else:
                            break
                    
                    # 检查负方向
                    for i in range(1, self.WIN_COUNT):
                        r, c = row - dr * i, col - dc * i
                        if (0 <= r < self.BOARD_SIZE and 
                            0 <= c < self.BOARD_SIZE and 
                            self.board[r][c] == player):
                            count += 1
                        else:
                            break
                    
                    # 如果已经连成指定数量，返回获胜者
                    if count >= self.WIN_COUNT:
                        return player
        
        return None
    
    def _check_draw(self) -> bool:
        """
        检查是否平局（棋盘已满且无获胜者）
        
        Returns:
            True表示平局，False表示未平局
        """
        if self.winner:
            return False
        for row in self.board:
            if '' in row:
                return False
        return True
    
    def _get_available_moves(self) -> list:
        """
        获取所有可用的移动位置
        
        Returns:
            可用位置列表，每个位置是(row, col)元组
        """
        moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == '':
                    moves.append((row, col))
        return moves
    
    def _make_move(self, row: int, col: int, player: str) -> bool:
        """
        在指定位置下棋
        
        Args:
            row: 行号（0-11）
            col: 列号（0-11）
            player: 玩家标识（'X'或'O'）
            
        Returns:
            True表示成功，False表示失败
        """
        if row < 0 or row >= self.BOARD_SIZE or col < 0 or col >= self.BOARD_SIZE:
            return False
        if self.board[row][col] != '':
            return False
        if self.game_over:
            return False
        
        self.board[row][col] = player
        self.winner = self._check_winner()
        
        if self.winner:
            self.game_over = True
        elif self._check_draw():
            self.game_over = True
            self.winner = None
        
        return True
    
    def _evaluate_position(self, row: int, col: int, player: str) -> int:
        """
        评估某个位置对指定玩家的价值
        
        Args:
            row: 行号
            col: 列号
            player: 玩家标识（'X'或'O'）
            
        Returns:
            位置得分（分数越高越好）
        """
        if self.board[row][col] != '':
            return 0
        
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        total_score = 0
        
        for dr, dc in directions:
            # 检查这个方向上的连子情况
            # 正方向
            count_forward = 0
            blocked_forward = False
            for i in range(1, self.WIN_COUNT):
                r, c = row + dr * i, col + dc * i
                if (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE):
                    if self.board[r][c] == player:
                        count_forward += 1
                    elif self.board[r][c] != '':
                        blocked_forward = True
                        break
                    else:
                        break
                else:
                    blocked_forward = True
                    break
            
            # 负方向
            count_backward = 0
            blocked_backward = False
            for i in range(1, self.WIN_COUNT):
                r, c = row - dr * i, col - dc * i
                if (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE):
                    if self.board[r][c] == player:
                        count_backward += 1
                    elif self.board[r][c] != '':
                        blocked_backward = True
                        break
                    else:
                        break
                else:
                    blocked_backward = True
                    break
            
            # 计算这个方向的总连子数
            total_count = count_forward + count_backward + 1  # +1是当前位置
            
            # 根据连子数和是否被阻挡来评分
            if total_count >= self.WIN_COUNT:
                # 可以获胜，给最高分
                return 10000
            elif total_count == self.WIN_COUNT - 1:
                # 差一个就获胜，给高分
                if not (blocked_forward and blocked_backward):
                    total_score += 1000
            elif total_count == self.WIN_COUNT - 2:
                # 差两个就获胜，给中高分
                if not (blocked_forward and blocked_backward):
                    total_score += 100
            elif total_count >= 2:
                # 有一定连子，给中等分
                if not (blocked_forward and blocked_backward):
                    total_score += 10
            elif total_count == 1:
                # 只有当前位置，给低分
                if not (blocked_forward and blocked_backward):
                    total_score += 1
        
        return total_score
    
    def _environment_move(self) -> Tuple[int, int]:
        """
        环境（AI）自动走棋（使用评分策略）
        
        Returns:
            移动位置(row, col)
        """
        available_moves = self._get_available_moves()
        if not available_moves:
            return None, None
        
        # 策略1：如果对手下一步能获胜，必须阻止
        for row, col in available_moves:
            self.board[row][col] = 'X'
            if self._check_winner() == 'X':
                self.board[row][col] = ''
                return row, col
            self.board[row][col] = ''
        
        # 策略2：如果自己能获胜，立即获胜
        for row, col in available_moves:
            self.board[row][col] = 'O'
            if self._check_winner() == 'O':
                self.board[row][col] = ''
                return row, col
            self.board[row][col] = ''
        
        # 策略3：使用评分系统选择最佳位置
        best_moves = []
        best_score = -1
        
        for row, col in available_moves:
            # 计算这个位置的进攻得分（自己的得分）
            attack_score = self._evaluate_position(row, col, 'O')
            # 计算这个位置的防守得分（阻止对手的得分）
            defense_score = self._evaluate_position(row, col, 'X')
            
            # 综合得分：防守和进攻都很重要，但防守稍微优先
            total_score = attack_score * 1.2 + defense_score * 1.5
            
            if total_score > best_score:
                best_score = total_score
                best_moves = [(row, col)]
            elif total_score == best_score:
                best_moves.append((row, col))
        
        # 如果有多个相同得分的位置，随机选择一个
        if best_moves:
            return random.choice(best_moves)
        
        # 如果所有位置得分都是0，随机选择
        return random.choice(available_moves)
    
    def _format_board(self) -> str:
        """
        格式化棋盘为字符串（12x12棋盘，使用紧凑格式）
        
        Returns:
            格式化的棋盘字符串
        """
        lines = []
        lines.append("当前棋盘状态（12x12棋盘）：")
        
        # 列号标题（只显示个位数，两位数显示最后一位）
        header = "   "
        for j in range(self.BOARD_SIZE):
            header += f"{j % 10} "
        lines.append(header)
        lines.append("   " + "-" * (self.BOARD_SIZE * 2 - 1))
        
        # 棋盘内容
        for i, row in enumerate(self.board):
            # 行号（右对齐，两位数显示最后一位）
            row_str = f"{i % 10:2} "
            for j, cell in enumerate(row):
                if cell == '':
                    row_str += ". "
                elif cell == 'X':
                    row_str += "X "
                elif cell == 'O':
                    row_str += "O "
            lines.append(row_str)
        
        lines.append("\n说明：行号和列号范围是0-11，X表示大模型，O表示环境")
        return "\n".join(lines)
    
    def _reset_game(self):
        """重置游戏"""
        self.board = [['' for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
    
    def execute_command(self, command: str, arg: Optional[str] = None) -> Dict[str, Any]:
        """
        执行命令
        
        Args:
            command: 命令字符串
            arg: 可选参数字符串
            
        Returns:
            执行结果字典
        """
        command = command.strip().lower()
        
        # help命令：返回工具使用说明
        if command == "help":
            help_text = """
自定义工具使用说明
==================

这是一个通用命令执行工具，默认实现为12x12棋类游戏。

参数说明：
- command: 命令字符串（必需）
- arg: 可选参数字符串（可选）

支持的命令：
1. help - 显示此帮助信息
2. reset - 重置游戏
3. status - 查看当前游戏状态
4. move - 大模型走棋（需要arg参数指定位置）

走棋格式：
- 使用move命令，arg参数格式为 "(行,列)" 或 "行,列"
- 例如：command="move", arg="(0,0)" 或 arg="0,0"
- 行号和列号范围：0-11

游戏基本信息：
- 大模型使用 'X'，环境使用 'O'
- 大模型先手
- 这是一个棋类游戏，请通过实际对局来探索游戏规则和获胜条件
- 游戏会自动判断输赢并及时反馈结果

使用案例：
1. 查看帮助：
   command="help"

2. 重置游戏：
   command="reset"

3. 查看状态：
   command="status"

4. 大模型走棋（在位置(0,0)）：
   command="move", arg="(0,0)"
   或
   command="move", arg="0,0"

5. 大模型走棋（在位置(5,6)）：
   command="move", arg="(5,6)"
   或
   command="move", arg="5,6"

返回结果：
- 执行成功时返回当前棋盘布局和游戏状态
- 如果游戏结束，会明确反馈输赢信息
- 执行失败时返回错误信息
"""
            return {
                "status": "success",
                "message": help_text,
                "command": command,
                "arg": arg
            }
        
        # reset命令：重置游戏
        if command == "reset":
            self._reset_game()
            board_str = self._format_board()
            print_current("🎮 游戏已重置")
            print_current(board_str)
            return {
                "status": "success",
                "message": "游戏已重置",
                "board": board_str,
                "current_player": "X (大模型)",
                "game_over": False,
                "winner": None
            }
        
        # status命令：查看当前状态
        if command == "status":
            board_str = self._format_board()
            status_msg = board_str
            status_msg += f"\n\n当前轮到: {self.current_player} ({'大模型' if self.current_player == 'X' else '环境'})"
            status_msg += f"\n游戏状态: {'已结束' if self.game_over else '进行中'}"
            if self.game_over:
                if self.winner == 'X':
                    status_msg += "\n获胜者: 大模型 (X)"
                elif self.winner == 'O':
                    status_msg += "\n获胜者: 环境 (O)"
                else:
                    status_msg += "\n结果: 平局"
            else:
                status_msg += "\n获胜者: 未决出"
            
            print_current("📊 当前游戏状态：")
            print_current(board_str)
            
            return {
                "status": "success",
                "message": status_msg,
                "board": board_str,
                "current_player": self.current_player,
                "game_over": self.game_over,
                "winner": self.winner
            }
        
        # move命令：大模型走棋
        if command == "move":
            if self.game_over:
                return {
                    "status": "error",
                    "message": "游戏已结束，无法继续走棋。请使用reset命令重置游戏。",
                    "board": self._format_board(),
                    "game_over": True,
                    "winner": self.winner
                }
            
            if self.current_player != 'X':
                return {
                    "status": "error",
                    "message": f"当前轮到环境(O)走棋，不是大模型(X)的回合。",
                    "board": self._format_board(),
                    "current_player": self.current_player
                }
            
            # 解析位置参数
            if not arg:
                return {
                    "status": "error",
                    "message": "move命令需要arg参数指定位置，格式为 '(行,列)' 或 '行,列'，例如 '(0,0)' 或 '0,0'"
                }
            
            # 清理参数：去除括号和空格
            arg_clean = arg.strip().strip('()').replace(' ', '')
            
            try:
                # 解析行号和列号
                parts = arg_clean.split(',')
                if len(parts) != 2:
                    raise ValueError("位置格式错误")
                
                row = int(parts[0])
                col = int(parts[1])
                
                # 验证范围
                if row < 0 or row >= self.BOARD_SIZE or col < 0 or col >= self.BOARD_SIZE:
                    return {
                        "status": "error",
                        "message": f"位置超出范围。行号和列号必须在0-{self.BOARD_SIZE-1}之间，您输入的是 ({row},{col})"
                    }
                
                # 尝试走棋
                if not self._make_move(row, col, 'X'):
                    return {
                        "status": "error",
                        "message": f"位置 ({row},{col}) 已被占用或无效，请选择其他位置。",
                        "board": self._format_board()
                    }
                
                # 大模型走棋成功，检查游戏是否结束
                board_str = self._format_board()
                print_current(f"🤖 大模型在位置 ({row},{col}) 下棋")
                print_current(board_str)
                
                result_msg = f"大模型在位置 ({row},{col}) 下棋\n\n"
                result_msg += board_str
                
                if self.game_over:
                    if self.winner == 'X':
                        result_msg += "\n\n🎉 游戏结束！大模型获胜！"
                        print_current("🎉 游戏结束！大模型获胜！")
                    elif self.winner == 'O':
                        result_msg += "\n\n❌ 游戏结束！环境获胜！"
                        print_current("❌ 游戏结束！环境获胜！")
                    else:
                        result_msg += "\n\n🤝 游戏结束！平局！"
                        print_current("🤝 游戏结束！平局！")
                    
                    return {
                        "status": "success",
                        "message": result_msg,
                        "board": board_str,
                        "game_over": True,
                        "winner": self.winner,
                        "last_move": (row, col, 'X'),
                        "game_result": "win" if self.winner == 'X' else ("lose" if self.winner == 'O' else "draw")
                    }
                
                # 游戏未结束，环境自动走棋
                env_row, env_col = self._environment_move()
                if env_row is not None and env_col is not None:
                    self._make_move(env_row, env_col, 'O')
                    board_str = self._format_board()
                    print_current(f"⚙️ 环境在位置 ({env_row},{env_col}) 下棋")
                    print_current(board_str)
                    
                    result_msg += f"\n\n环境在位置 ({env_row},{env_col}) 下棋\n\n"
                    result_msg += board_str
                    
                    if self.game_over:
                        if self.winner == 'X':
                            result_msg += "\n\n🎉 游戏结束！大模型获胜！"
                            print_current("🎉 游戏结束！大模型获胜！")
                        elif self.winner == 'O':
                            result_msg += "\n\n❌ 游戏结束！环境获胜！"
                            print_current("❌ 游戏结束！环境获胜！")
                        else:
                            result_msg += "\n\n🤝 游戏结束！平局！"
                            print_current("🤝 游戏结束！平局！")
                    else:
                        result_msg += "\n\n轮到您（大模型）走棋"
                        print_current("➡️ 轮到您（大模型）走棋")
                    
                    return {
                        "status": "success",
                        "message": result_msg,
                        "board": board_str,
                        "game_over": self.game_over,
                        "winner": self.winner,
                        "last_move": (row, col, 'X'),
                        "environment_move": (env_row, env_col, 'O'),
                        "game_result": "win" if self.winner == 'X' else ("lose" if self.winner == 'O' else "draw") if self.game_over else None
                    }
                else:
                    # 没有可用位置（理论上不应该发生）
                    return {
                        "status": "error",
                        "message": "没有可用的移动位置",
                        "board": self._format_board()
                    }
                    
            except ValueError as e:
                return {
                    "status": "error",
                    "message": f"位置参数格式错误：{arg}。正确格式为 '(行,列)' 或 '行,列'，例如 '(0,0)' 或 '0,0'。行号和列号必须是0-{self.BOARD_SIZE-1}之间的整数。"
                }
        
        # 未知命令
        return {
            "status": "error",
            "message": f"未知命令: {command}。支持的命令：help, reset, status, move。使用 help 命令查看详细说明。"
        }


class EchoTool:
    """
    Echo工具：简单地将输入的字符串返回到输出
    """
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        初始化Echo工具
        
        Args:
            workspace_root: 工作空间根目录（未使用，保持接口一致性）
        """
        self.workspace_root = workspace_root or ""
    
    def execute_command(self, command: str, arg: Optional[str] = None) -> Dict[str, Any]:
        """
        执行echo命令：返回输入的字符串
        
        Args:
            command: 命令字符串（将被返回）
            arg: 可选参数字符串（将被返回）
            
        Returns:
            执行结果字典，包含输入的字符串
        """
        result_message = ""
        
        if command:
            result_message += f"命令: {command}"
        
        if arg:
            if result_message:
                result_message += f"\n参数: {arg}"
            else:
                result_message = f"参数: {arg}"
        
        # 如果没有输入任何内容，返回提示
        if not result_message:
            result_message = "Echo工具：没有接收到任何输入内容"
        
        return {
            "status": "success",
            "message": result_message,
            "command": command,
            "arg": arg,
            "echo_output": result_message
        }


class CustomTool:
    """
    自定义工具主类：根据 type 参数选择不同的工具类型
    """
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        初始化自定义工具
        
        Args:
            workspace_root: 工作空间根目录
        """
        self.workspace_root = workspace_root or ""
        # 初始化各个子工具
        self.game_tool = CustomGameTool(workspace_root=workspace_root)
        self.echo_tool = EchoTool(workspace_root=workspace_root)
    
    def execute_command(self, command: str, type: Optional[str] = None, arg: Optional[str] = None) -> Dict[str, Any]:
        """
        执行命令，根据 type 参数路由到不同的工具
        
        Args:
            command: 命令字符串
            type: 工具类型，'game' 或 'echo'（可选，默认为 None，将使用 echo）
            arg: 可选参数字符串
            
        Returns:
            执行结果字典
        """
        tool_type = type.strip().lower() if type else ""
        
        # 如果选择了 'game'，转给 CustomGameTool
        if tool_type == "game":
            return self.game_tool.execute_command(command, arg)
        
        # 否则转给 EchoTool
        # 如果没有指定 type 或 type 不是 'game'，都使用 echo
        return self.echo_tool.execute_command(command, arg)

