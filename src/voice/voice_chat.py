#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音对话程序
Voice Chat CLI

完整的语音对话流程：
1. 录音 (麦克风输入)
2. 语音识别 (ASR)
3. 大模型推理 (LLM)
4. 语音合成 (TTS)
5. 播放语音
"""

import os
import sys
import time
import asyncio
import argparse
import tempfile
import subprocess
import threading
import json
import re
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile
    import torch
except ImportError as e:
    print(f"❌ 缺少必需依赖: {e}")
    print("💡 请运行: pip install numpy sounddevice scipy torch")
    sys.exit(1)

# 尝试导入键盘监听库
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("💡 提示: 安装 pynput 可以支持空格键跳过播放: pip install pynput")


class VoiceChat:
    """语音对话类"""
    
    def __init__(self, config_path="config/config.txt"):
        """初始化"""
        self.config_path = config_path
        self.config = {}
        self.conversation_history = []
        
        # 音频参数（强制 16000Hz - Paraformer 模型要求）
        self.sample_rate = 16000  # 必须是 16000Hz
        self.channels = 1  # 必须是单声道
        
        # VAD 模型
        self.vad_model = None
        self.vad_threshold = 0.5
        self.silence_duration = 1.0  # 停顿1秒判断结束
        
        # 任务管理
        self.current_task = None  # 当前运行的任务信息
        self.task_process = None  # 任务进程
        self.task_thread = None  # 任务监控线程
        self.completed_tasks = []  # 已完成的任务列表
        self.task_lock = threading.Lock()
        
        # 播放控制
        self.skip_playback = False  # 是否跳过播放
        self.playback_lock = threading.Lock()
        
        # 加载配置
        self.load_config()
        
        # 初始化 VAD
        self.init_vad()
        
    def load_config(self):
        """加载配置文件"""
        print("🔧 加载配置文件...")
        
        if not os.path.exists(self.config_path):
            print(f"❌ 配置文件不存在: {self.config_path}")
            sys.exit(1)
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip()
            
            # 获取配置
            self.asr_provider = self.config.get('asr_provider', 'sherpa')
            self.tts_provider = self.config.get('tts_provider', 'edge_tts')
            self.tts_voice = self.config.get('tts_voice', 'zh-CN-XiaoxiaoNeural')
            self.sample_rate = int(self.config.get('audio_sample_rate', '16000'))
            
            # LLM 配置
            self.api_key = self.config.get('api_key', '')
            self.api_base = self.config.get('api_base', '')
            self.model = self.config.get('model', '')
            
            print(f"✅ 配置加载成功")
            print(f"   ASR: {self.asr_provider}")
            print(f"   TTS: {self.tts_provider}")
            print(f"   LLM: {self.model}")
            
        except Exception as e:
            print(f"❌ 加载配置失败: {str(e)}")
            sys.exit(1)
    
    def init_vad(self):
        """初始化 VAD 模型"""
        try:
            print("🔧 初始化 VAD 模型...")
            # 使用 Silero VAD
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            print("✅ VAD 模型加载成功")
        except Exception as e:
            print(f"⚠️  VAD 模型加载失败: {str(e)}")
            print("💡 将使用简单的音量检测作为替代")
            self.vad_model = None
    
    def parse_requirement_summary(self, response_text):
        """解析需求总结"""
        try:
            # 查找 REQUIREMENT_SUMMARY: {...} 格式
            match = re.search(r'REQUIREMENT_SUMMARY:\s*(\{.*?\})', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                requirement = json.loads(json_str)
                return requirement
            return None
        except Exception as e:
            print(f"⚠️  解析需求总结失败: {str(e)}")
            return None
    
    def start_task(self, requirement):
        """启动任务执行"""
        try:
            # 生成任务目录名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            task_dir = f"voice_task_{timestamp}"
            
            # 构建命令
            title = requirement.get('title', '未命名任务')
            description = requirement.get('description', '')
            
            # 完整的需求描述
            full_requirement = f"{title}\n\n{description}"
            if 'key_points' in requirement:
                full_requirement += "\n\n关键要点：\n"
                for i, point in enumerate(requirement['key_points'], 1):
                    full_requirement += f"{i}. {point}\n"
            
            # 启动 agia.py
            cmd = [
                sys.executable,  # 使用当前 Python 解释器
                "agia.py",
                "-r", full_requirement,
                "-d", task_dir
            ]
            
            print(f"\n🚀 启动任务: {title}")
            print(f"📁 任务目录: {task_dir}")
            
            # 在新线程中启动进程
            def run_task():
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )
                    
                    with self.task_lock:
                        self.task_process = process
                        self.current_task = {
                            'title': title,
                            'description': description,
                            'task_dir': task_dir,
                            'start_time': datetime.now(),
                            'status': 'running'
                        }
                    
                    # 等待进程完成
                    process.wait()
                    
                    with self.task_lock:
                        if self.current_task:
                            self.current_task['status'] = 'completed'
                            self.current_task['end_time'] = datetime.now()
                            self.completed_tasks.append(self.current_task.copy())
                            print(f"\n✅ 任务完成: {title}")
                    
                except Exception as e:
                    print(f"\n❌ 任务执行失败: {str(e)}")
                    with self.task_lock:
                        if self.current_task:
                            self.current_task['status'] = 'failed'
                            self.current_task['error'] = str(e)
            
            # 启动任务线程
            self.task_thread = threading.Thread(target=run_task, daemon=True)
            self.task_thread.start()
            
            # 等待一小段时间确保任务已启动
            time.sleep(0.5)
            
            # 向对话历史中添加任务启动确认信息
            task_started_msg = f"任务已启动！AGIAgent 正在后台执行任务「{title}」，目标文件夹是 {task_dir}。您可以继续对话追加需求，或者询问任务进度。"
            self.conversation_history.append({
                "role": "assistant",
                "content": task_started_msg
            })
            
            return True
            
        except Exception as e:
            print(f"❌ 启动任务失败: {str(e)}")
            return False
    
    def send_message_to_manager(self, content):
        """发送消息给 manager"""
        try:
            if not self.current_task:
                print("⚠️  没有正在运行的任务")
                return False
            
            task_dir = self.current_task['task_dir']
            mailbox_dir = os.path.join(task_dir, "mailboxes", "manager", "inbox")
            
            # 确保目录存在
            os.makedirs(mailbox_dir, exist_ok=True)
            
            # 查找下一个消息 ID
            existing_files = [f for f in os.listdir(mailbox_dir) if f.startswith("extmsg_") and f.endswith(".json")]
            if existing_files:
                max_id = max([int(f.split("_")[1].split(".")[0]) for f in existing_files])
                next_id = max_id + 1
            else:
                next_id = 1
            
            # 创建消息
            message = {
                "message_id": f"extmsg_{next_id}",
                "sender_id": "voice_user",
                "receiver_id": "manager",
                "message_type": "COLLABORATION",
                "content": {"text": content},
                "priority": "NORMAL",
                "timestamp": datetime.now().isoformat(),
                "read": False
            }
            
            # 保存消息
            message_file = os.path.join(mailbox_dir, f"extmsg_{next_id}.json")
            with open(message_file, 'w', encoding='utf-8') as f:
                json.dump(message, f, indent=2, ensure_ascii=False)
            
            print(f"📨 已发送消息给 manager")
            return True
            
        except Exception as e:
            print(f"❌ 发送消息失败: {str(e)}")
            return False
    
    def check_completed_tasks(self):
        """检查是否有新完成的任务"""
        with self.task_lock:
            if self.completed_tasks:
                # 返回并清空已完成任务列表
                tasks = self.completed_tasks.copy()
                self.completed_tasks.clear()
                return tasks
        return []
    
    def preprocess_audio(self, audio_data):
        """音频预处理 - 提高识别准确率"""
        try:
            # 1. 如果是立体声，转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)
            
            # 2. 确保是一维数组
            audio_data = audio_data.flatten()
            
            # 3. 检查音频长度
            duration = len(audio_data) / self.sample_rate
            if duration < 0.3:
                print(f"\n⚠️  音频太短 ({duration:.2f}s)")
                return None
            
            # 4. 归一化音量（关键改进！）
            max_val = np.max(np.abs(audio_data))
            if max_val == 0:
                print("\n⚠️  音频全是静音")
                return None
            
            if max_val > 0:
                # 归一化到接近最大值，但留一点余量避免削波
                audio_data = audio_data * (30000.0 / max_val)
                audio_data = np.clip(audio_data, -32768, 32767)
                audio_data = audio_data.astype(np.int16)
            
            # 5. 高通滤波（去除低频噪音）
            try:
                from scipy import signal
                # 80Hz 高通滤波器
                sos = signal.butter(2, 80, 'hp', fs=self.sample_rate, output='sos')
                audio_data = signal.sosfilt(sos, audio_data).astype(np.int16)
            except:
                # 如果 scipy 不可用，跳过滤波
                pass
            
            return audio_data
            
        except Exception as e:
            print(f"\n⚠️  音频预处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return audio_data
    
    def get_volume_bar(self, amplitude):
        """根据音量大小返回对应的可视化字符
        
        使用多级灰度字符表示音量强度：
        - █ 最强 (90-100%)
        - ▓ 很强 (70-90%)
        - ▒ 较强 (50-70%)
        - ░ 中等 (30-50%)
        - ▁ 较弱 (15-30%)
        - ▂ 很弱 (5-15%)
        - · 极弱 (0-5%)
        """
        # 归一化到 0-100
        percentage = min(100, (amplitude / 32768.0) * 100)
        
        if percentage >= 90:
            return "█"  # 最强
        elif percentage >= 70:
            return "▓"  # 很强
        elif percentage >= 50:
            return "▒"  # 较强
        elif percentage >= 30:
            return "░"  # 中等
        elif percentage >= 15:
            return "▁"  # 较弱
        elif percentage >= 5:
            return "▂"  # 很弱
        else:
            return "·"  # 极弱
    
    def record_audio_with_vad(self):
        """使用 VAD 进行录音（连续音频流版本 - 无间隙）"""
        print(f"\n🎤 录音中（停顿 {self.silence_duration}s 自动结束）", end="", flush=True)
        
        try:
            import queue
            
            # 音频队列
            audio_queue = queue.Queue()
            recording_active = True
            
            def audio_callback(indata, frames, time_info, status):
                """音频回调函数 - 在独立线程中持续接收音频"""
                if status:
                    print(f"\n⚠️  录音状态: {status}", file=sys.stderr)
                if recording_active:
                    audio_queue.put(indata.copy())
            
            # 创建连续输入流（关键改进：无间隙录音）
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms 块大小
            )
            
            stream.start()
            
            # 存储所有录音数据
            all_audio = []
            
            # 停顿检测
            silence_chunks = 0
            silence_threshold = int(self.silence_duration / 0.1)  # 100ms 一块
            
            # 是否检测到语音
            speech_detected = False
            
            # 最多录音 60 秒
            max_chunks = 600  # 600 * 0.1s = 60s
            
            try:
                while len(all_audio) < max_chunks:
                    try:
                        # 获取音频块（超时 2 秒）
                        chunk = audio_queue.get(timeout=2.0)
                        all_audio.append(chunk)
                        
                        # 计算当前块的音量
                        max_amplitude = np.max(np.abs(chunk))
                        
                        # 检测是否有语音
                        if self.vad_model is not None:
                            # 使用 Silero VAD
                            audio_float = chunk.flatten().astype(np.float32) / 32768.0
                            audio_tensor = torch.from_numpy(audio_float)
                            
                            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                            
                            if speech_prob > self.vad_threshold:
                                # 检测到语音
                                if not speech_detected:
                                    print(" 🗣️ ", end="", flush=True)
                                    speech_detected = True
                                silence_chunks = 0
                                # 根据音量显示不同的字符
                                print(self.get_volume_bar(max_amplitude), end="", flush=True)
                            else:
                                # 静音
                                if speech_detected:
                                    silence_chunks += 1
                                    print(self.get_volume_bar(max_amplitude), end="", flush=True)
                                    
                                    if silence_chunks >= silence_threshold:
                                        print(" ✅")
                                        break
                        else:
                            # 简单的音量检测
                            if max_amplitude > 500:  # 音量阈值
                                if not speech_detected:
                                    print(" 🗣️ ", end="", flush=True)
                                    speech_detected = True
                                silence_chunks = 0
                                # 根据音量显示不同的字符
                                print(self.get_volume_bar(max_amplitude), end="", flush=True)
                            else:
                                if speech_detected:
                                    silence_chunks += 1
                                    print(self.get_volume_bar(max_amplitude), end="", flush=True)
                                    
                                    if silence_chunks >= silence_threshold:
                                        print(" ✅")
                                        break
                        
                    except queue.Empty:
                        print("\n⚠️  录音超时")
                        break
            
            finally:
                # 停止录音流
                recording_active = False
                stream.stop()
                stream.close()
            
            # 检查是否录到了语音
            if not speech_detected:
                print("\n⚠️  未检测到语音")
                return None
            
            # 合并所有音频（完全连续，无间隙！）
            recording = np.concatenate(all_audio, axis=0)
            
            # 音频预处理和归一化（提高识别准确率）
            recording = self.preprocess_audio(recording)
            
            if recording is None:
                return None
            
            # 不再显示录音时长
            return recording
            
        except Exception as e:
            print(f"❌ 录音失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def record_audio(self, duration=5):
        """录音"""
        print(f"\n🎤 开始录音 ({duration} 秒)...")
        print("💬 请说话...")
        
        try:
            # 录音
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16
            )
            sd.wait()
            
            print("✅ 录音完成")
            
            # 检查音量
            max_amplitude = np.max(np.abs(recording))
            avg_amplitude = np.mean(np.abs(recording))
            
            print(f"📊 音频信息:")
            print(f"   最大振幅: {max_amplitude}")
            print(f"   平均振幅: {avg_amplitude:.2f}")
            
            if max_amplitude < 100:
                print("⚠️  警告: 音量太低，请检查麦克风")
            
            return recording
            
        except Exception as e:
            print(f"❌ 录音失败: {str(e)}")
            return None
    
    def save_audio(self, audio_data, filename):
        """保存音频文件并使用 ffmpeg 优化（提高识别准确率）"""
        try:
            # 1. 先保存原始音频到临时文件
            temp_file = filename.replace('.wav', '_temp.wav')
            wavfile.write(temp_file, self.sample_rate, audio_data)
            
            # 2. 尝试使用 ffmpeg 处理（专业音频处理）
            try:
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', temp_file,
                    '-ar', '16000',           # 精确采样率转换
                    '-ac', '1',               # 单声道
                    '-acodec', 'pcm_s16le',   # PCM 16-bit 编码
                    '-y',                     # 覆盖输出
                    filename
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=10)
                
                if result.returncode == 0:
                    # ffmpeg 成功，删除临时文件
                    os.remove(temp_file)
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # ffmpeg 不可用或超时
                pass
            
            # 3. 如果 ffmpeg 失败，尝试使用 pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(temp_file)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(filename, format='wav')
                os.remove(temp_file)
                return True
            except ImportError:
                # pydub 不可用
                pass
            
            # 4. 如果都失败，使用原始文件
            if os.path.exists(temp_file):
                os.rename(temp_file, filename)
            
            return True
            
        except Exception as e:
            print(f"❌ 保存音频失败: {str(e)}")
            return False
    
    def speech_to_text_sherpa(self, audio_file):
        """使用 SherpaASR 进行语音识别"""
        
        try:
            # 先读取音频检查质量
            sample_rate, audio_data = wavfile.read(audio_file)
            
            # 检查音频时长
            duration = len(audio_data) / sample_rate
            if duration < 0.5:
                # 音频太短，直接跳过不识别
                return None
            
            # 检查音频音量
            max_amplitude = np.max(np.abs(audio_data))
            avg_amplitude = np.mean(np.abs(audio_data))
            
            # 如果音量太低，直接跳过
            if max_amplitude < 1000 or avg_amplitude < 200:
                # 音量太低，可能只是环境噪音
                return None
            
            print("🎯 识别中...", end="", flush=True)
            
            # 抑制 stderr 输出（避免 ONNX 警告）
            import sys
            import os
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            import sherpa_onnx
            
            # 获取模型路径
            model_path = self.config.get('asr_model_path', '')
            if not model_path or not os.path.exists(model_path):
                print(f"❌ 模型路径不存在: {model_path}")
                print("💡 请先下载模型或运行: ./install_voice.sh")
                return None
            
            # 创建识别器
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=os.path.join(model_path, "model.int8.onnx"),
                tokens=os.path.join(model_path, "tokens.txt"),
                num_threads=2,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
            )
            
            # 检查采样率
            if int(sample_rate) != int(self.sample_rate):
                print(f"\n⚠️  采样率不匹配: {sample_rate} != {self.sample_rate}")
            
            # 注意：不再需要立体声转换，preprocess_audio 已经处理过了
            # 直接转换为 float32
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # 创建音频流
            stream = recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_data)
            
            # 识别
            recognizer.decode_stream(stream)
            result = stream.result.text
            
            # 恢复 stderr
            sys.stderr.close()
            sys.stderr = old_stderr
            
            if result:
                print(f" ✅ {result}")
                return result
            else:
                # 识别结果为空，不显示提示
                return None
                
        except ImportError:
            # 恢复 stderr
            if 'old_stderr' in locals():
                sys.stderr.close()
                sys.stderr = old_stderr
            print("❌ sherpa-onnx 未安装")
            print("💡 请运行: pip install sherpa-onnx")
            return None
        except Exception as e:
            # 恢复 stderr
            if 'old_stderr' in locals():
                sys.stderr.close()
                sys.stderr = old_stderr
            print(f" ❌ 识别失败: {str(e)}")
            return None
    
    def speech_to_text_dummy(self, audio_file):
        """模拟语音识别（用于测试）"""
        print("\n🎯 模拟语音识别...")
        print("⚠️  ASR 未配置，使用模拟输入")
        
        # 让用户手动输入
        text = input("请输入您想说的话: ")
        return text if text else None
    
    def speech_to_text(self, audio_file):
        """语音识别（根据配置选择引擎）"""
        if self.asr_provider == 'sherpa':
            return self.speech_to_text_sherpa(audio_file)
        else:
            print(f"⚠️  ASR provider '{self.asr_provider}' 暂不支持，使用模拟输入")
            return self.speech_to_text_dummy(audio_file)
    
    def chat_with_llm(self, user_input):
        """与大模型对话"""
        print(f"🤖 {self.model}:", end="", flush=True)
        
        try:
            # 导入 API 调用模块
            try:
                import anthropic
            except ImportError:
                anthropic = None
            try:
                from openai import OpenAI
            except ImportError:
                OpenAI = None
            
            # 构建消息
            messages = []
            
            # 添加系统提示
            messages.append({
                "role": "system",
                "content": """你是一个友好的AI助手。请用简洁、自然的语言回答用户的问题。回答要口语化，适合语音播报。

重要规则：
1. 当用户说"我的需求就这些，开始设计"或类似表达时，你需要：
   a) 先总结整个对话，形成清晰的需求描述
   b) 然后在回复的最后输出特殊标记：DIALOG_FINISHED
   c) 在 DIALOG_FINISHED 后面的一行，输出 JSON 格式的需求总结：REQUIREMENT_SUMMARY: {...}

2. 需求总结格式：
   REQUIREMENT_SUMMARY: {"title": "任务标题", "description": "详细需求描述", "key_points": ["要点1", "要点2", ...]}

3. 示例：
   用户：我的需求就这些，开始设计
   助手：好的，让我总结一下您的需求：您想要创建一个博客系统，需要支持文章发布、评论功能和用户权限管理。我现在开始为您设计方案。
   DIALOG_FINISHED
   REQUIREMENT_SUMMARY: {"title": "博客系统开发", "description": "创建一个完整的博客系统，包含文章发布、评论和用户管理功能", "key_points": ["文章发布功能", "评论系统", "用户登录和权限管理"]}"""
            })
            
            # 添加历史对话（最多保留最近3轮）
            for msg in self.conversation_history[-6:]:
                messages.append(msg)
            
            # 添加当前用户输入
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # 判断使用哪个 API
            streaming = self.config.get('streaming', 'True').lower() == 'true'
            
            # 检查是否是 Claude API
            is_claude = 'anthropic' in self.api_base.lower() or 'claude' in self.model.lower()
            
            # 调用 API
            response_text = ""
            
            if is_claude:
                # 使用 Anthropic API
                if anthropic is None:
                    print("❌ anthropic 库未安装")
                    print("💡 请运行: pip install anthropic")
                    return None
                
                client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else None
                )
                
                # 转换消息格式（Anthropic 不支持 system role）
                system_content = ""
                api_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    else:
                        api_messages.append(msg)
                
                if streaming:
                    with client.messages.stream(
                        model=self.model,
                        max_tokens=int(self.config.get('max_tokens', '4096')),
                        messages=api_messages,
                        system=system_content if system_content else None,
                        temperature=0.7
                    ) as stream:
                        for text in stream.text_stream:
                            response_text += text
                            print(text, end='', flush=True)
                else:
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=int(self.config.get('max_tokens', '4096')),
                        messages=api_messages,
                        system=system_content if system_content else None,
                        temperature=0.7
                    )
                    response_text = response.content[0].text
                    print(f" {response_text}")
            else:
                # 使用 OpenAI API
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
                
                if streaming:
                    stream = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=int(self.config.get('max_tokens', '4096')),
                        temperature=0.7,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            text = chunk.choices[0].delta.content
                            response_text += text
                            print(text, end='', flush=True)
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=int(self.config.get('max_tokens', '4096')),
                        temperature=0.7
                    )
                    response_text = response.choices[0].message.content
                    print(f" {response_text}")
            
            if response_text:
                # 保存到历史
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # 检查是否包含结束标志
                if "DIALOG_FINISHED" in response_text:
                    print("🎯 [对话结束]")
                    return response_text
                
                return response_text
            else:
                print("⚠️  大模型未返回内容")
                return None
                
        except Exception as e:
            print(f"❌ 大模型调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    async def text_to_speech_edge_tts(self, text, output_file):
        """使用 EdgeTTS 进行语音合成"""
        # 不再显示"合成中"提示
        
        try:
            import edge_tts
            
            communicate = edge_tts.Communicate(text, self.tts_voice)
            await communicate.save(output_file)
            
            return True
            
        except ImportError:
            print("❌ edge-tts 未安装")
            print("💡 请运行: pip install edge-tts")
            return False
        except Exception as e:
            print(f"❌ 语音合成失败: {str(e)}")
            return False
    
    def text_to_speech(self, text, output_file):
        """语音合成（根据配置选择引擎）"""
        if self.tts_provider == 'edge_tts':
            return asyncio.run(self.text_to_speech_edge_tts(text, output_file))
        else:
            print(f"⚠️  TTS provider '{self.tts_provider}' 暂不支持")
            return False
    
    def play_audio(self, audio_file):
        """播放音频（支持空格键跳过）"""
        # 不再显示"播放中"提示
        
        # 重置跳过标志
        with self.playback_lock:
            self.skip_playback = False
        
        # 设置键盘监听
        listener = None
        if KEYBOARD_AVAILABLE:
            def on_press(key):
                try:
                    if key == keyboard.Key.space:
                        with self.playback_lock:
                            self.skip_playback = True
                except:
                    pass
            
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
        
        try:
            # 尝试使用 pygame
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # 循环检查播放状态和跳过标志
                while pygame.mixer.music.get_busy():
                    with self.playback_lock:
                        if self.skip_playback:
                            pygame.mixer.music.stop()
                            if listener:
                                listener.stop()
                            return True
                    time.sleep(0.05)
                
                if listener:
                    listener.stop()
                return True
                
            except ImportError:
                # 如果没有 pygame，尝试使用 sounddevice
                print("ℹ️  pygame 未安装，使用 sounddevice 播放")
                
                # 读取音频文件
                if audio_file.endswith('.mp3'):
                    # MP3 需要转换
                    print("⚠️  MP3 格式需要 pygame，请安装: pip install pygame")
                    if listener:
                        listener.stop()
                    return False
                else:
                    # WAV 文件
                    audio_data, sample_rate = wavfile.read(audio_file)
                    sd.play(audio_data, sample_rate)
                    
                    # 循环检查播放状态和跳过标志
                    while sd.get_stream().active:
                        with self.playback_lock:
                            if self.skip_playback:
                                sd.stop()
                                if listener:
                                    listener.stop()
                                return True
                        time.sleep(0.05)
                    
                    if listener:
                        listener.stop()
                    return True
                    
        except Exception as e:
            print(f"❌ 播放失败: {str(e)}")
            print(f"💡 音频文件已保存到: {audio_file}")
            print(f"💡 可以手动播放")
            if listener:
                listener.stop()
            return False
    
    def chat_once(self, use_vad=True):
        """单次对话"""
        # 不再显示分隔线
        
        # 检查是否有已完成的任务
        completed_tasks = self.check_completed_tasks()
        if completed_tasks:
            for task in completed_tasks:
                title = task.get('title', '任务')
                notification = f"您的任务「{title}」已经执行完毕了。"
                print(f"\n🎉 {notification}")
                
                # 语音通知
                with tempfile.TemporaryDirectory() as temp_dir:
                    notification_audio = os.path.join(temp_dir, "notification.mp3")
                    if self.text_to_speech(notification, notification_audio):
                        self.play_audio(notification_audio)
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            input_audio = os.path.join(temp_dir, "input.wav")
            output_audio = os.path.join(temp_dir, "output.mp3")
            
            # 1. 录音（使用 VAD 自动检测停顿）
            if use_vad:
                audio_data = self.record_audio_with_vad()
            else:
                audio_data = self.record_audio(duration=5)
            
            if audio_data is None:
                return False, False
            
            # 保存录音
            if not self.save_audio(audio_data, input_audio):
                return False, False
            
            # 2. 语音识别
            user_text = self.speech_to_text(input_audio)
            if not user_text:
                return False, False
            
            # 检查是否是退出命令
            exit_keywords = ["退出", "结束", "再见", "拜拜"]
            if any(keyword in user_text for keyword in exit_keywords):
                response_text = "好的，再见！"
                tts_text = response_text
                if self.text_to_speech(tts_text, output_audio):
                    self.play_audio(output_audio)
                return True, True  # 退出对话
            
            # 检查是否是追加需求（如果有正在运行的任务）
            if self.current_task and self.current_task.get('status') == 'running':
                # 判断是否是追加需求的关键词
                add_requirement_keywords = ["还要", "另外", "补充", "追加", "再加", "还需要"]
                if any(keyword in user_text for keyword in add_requirement_keywords):
                    # 发送消息给 manager
                    additional_req = f"追加需求：{user_text}"
                    if self.send_message_to_manager(additional_req):
                        response_text = "好的，我已经将您的追加需求发送给执行系统了。"
                        tts_text = response_text
                        
                        # 语音合成和播放
                        if self.text_to_speech(tts_text, output_audio):
                            self.play_audio(output_audio)
                        
                        return True, False
            
            # 3. 大模型推理
            response_text = self.chat_with_llm(user_text)
            if not response_text:
                return False, False
            
            # 检查是否包含结束标志
            should_exit = "DIALOG_FINISHED" in response_text
            
            # 如果包含结束标志，解析需求总结并启动任务
            if should_exit:
                requirement = self.parse_requirement_summary(response_text)
                if requirement:
                    # 启动任务
                    self.start_task(requirement)
                
                # 从回复中移除特殊标记
                response_text = response_text.replace("DIALOG_FINISHED", "")
                response_text = re.sub(r'REQUIREMENT_SUMMARY:.*', '', response_text, flags=re.DOTALL)
                response_text = response_text.strip()
            
            # 准备用于 TTS 的文本（移除 Markdown 格式符号）
            tts_text = response_text.replace("**", "").strip()
            
            # 4. 语音合成
            if tts_text and not self.text_to_speech(tts_text, output_audio):
                return True, should_exit
            
            # 5. 播放语音
            if tts_text:
                self.play_audio(output_audio)
            
            return True, should_exit
    
    def run(self, use_vad=True, max_rounds=100):
        """运行对话循环"""
        print("\n" + "=" * 60)
        print("🚀 语音对话程序启动")
        print("=" * 60)
        print(f"\n配置信息:")
        print(f"  录音模式: {'VAD 自动检测停顿' if use_vad else '固定时长'}")
        print(f"  停顿检测: {self.silence_duration} 秒")
        print(f"  最大轮数: {max_rounds} 轮")
        print(f"  ASR: {self.asr_provider}")
        print(f"  TTS: {self.tts_provider}")
        print(f"  LLM: {self.model}")
        print(f"\n💡 提示:")
        print(f"  - 说话时会自动录音")
        print(f"  - 停顿 {self.silence_duration} 秒后自动结束录音")
        print(f"  - 说'我的需求就这些，开始设计'可结束对话")
        print(f"  - 按 Ctrl+C 随时退出")
        
        round_count = 0
        
        try:
            while round_count < max_rounds:
                round_count += 1
                # 不再显示"第X轮对话"提示
                
                # 执行一次对话
                success, should_exit = self.chat_once(use_vad=use_vad)
                
                if not success:
                    time.sleep(0.5)
                    continue
                
                # 检查是否需要退出
                if should_exit:
                    print("\n" + "=" * 60)
                    print("✅ 需求收集完成，任务已启动")
                    print("💡 您可以继续对话来追加新的需求")
                    print("💡 说'退出'或按 Ctrl+C 结束对话")
                    print("=" * 60)
                    # 不退出，继续对话以便追加需求
                    continue
                
                time.sleep(1)
            
            if round_count >= max_rounds:
                print(f"\n✅ 已完成 {max_rounds} 轮对话")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            print("👋 再见！")
        
        # 保存对话历史到文件，不再打印到终端
        if self.conversation_history:
            try:
                # 保存到当前目录的 conversation_history.json
                history_file = "conversation_history.json"
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                print(f"\n💾 对话历史已保存到: {history_file}")
            except Exception as e:
                print(f"\n⚠️  保存对话历史失败: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AGI Agent 语音对话程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python voice_chat.py                    # 默认配置（VAD自动检测停顿）
  python voice_chat.py --no-vad -d 5      # 使用固定时长录音（5秒）
  python voice_chat.py -r 50              # 最多50轮对话
  python voice_chat.py --config my.txt    # 使用自定义配置文件
  
说明:
  - 默认使用 VAD 自动检测1秒停顿来判断录音结束
  - 支持多轮对话，说"我的需求就这些，开始设计"可结束对话
  - 大模型会回复 DIALOG_FINISHED 标志来结束程序
        """
    )
    
    parser.add_argument('--no-vad', action='store_true',
                        help='禁用 VAD，使用固定时长录音')
    parser.add_argument('-d', '--duration', type=int, default=5,
                        help='录音时长（秒），仅在禁用 VAD 时有效，默认5秒')
    parser.add_argument('-r', '--rounds', type=int, default=100,
                        help='最大对话轮数，默认100轮')
    parser.add_argument('-c', '--config', type=str, default='config/config.txt',
                        help='配置文件路径')
    parser.add_argument('-s', '--silence', type=float, default=1.0,
                        help='停顿检测时长（秒），默认1.0秒')
    
    args = parser.parse_args()
    
    # 创建对话实例
    chat = VoiceChat(config_path=args.config)
    
    # 设置停顿检测时长
    chat.silence_duration = args.silence
    
    # 运行对话
    chat.run(use_vad=not args.no_vad, max_rounds=args.rounds)


if __name__ == "__main__":
    main()
