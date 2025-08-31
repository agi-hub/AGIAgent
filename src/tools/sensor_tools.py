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

import os
import json
import base64
import subprocess
import time
from typing import Dict, Any, Optional
from datetime import datetime
from .print_system import print_current


class SensorDataCollector:
    """Sensor data collection class for acquiring physical world information."""
    
    def __init__(self, workspace_root: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize sensor data collector.
        
        Args:
            workspace_root: Root directory for saving captured files
            model: Current AI model name for vision API detection
        """
        self.workspace_root = workspace_root or os.getcwd()
        self.output_dir = os.path.join(self.workspace_root, 'sensor_data')
        self.model = model  # Store model name for vision detection
    
    def _ensure_sensor_directory(self):
        """Ensure the sensor data directory exists when needed"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                print_current(f"📁 Created sensor_data directory: {self.output_dir}")
        except Exception as e:
            print_current(f"⚠️ Failed to create sensor_data directory: {e}")
            
    def _resolve_file_path(self, source: str) -> Optional[str]:
        """
        Intelligently resolve file paths for sensor data sources.
        
        Supports three path types:
        1. Absolute paths
        2. Paths relative to workspace/sensor_data  
        3. Paths relative to current working directory
        
        Args:
            source: Source file path
            
        Returns:
            Resolved absolute path if file exists, None otherwise
        """
        # Type 1: Absolute path
        if os.path.isabs(source):
            if os.path.isfile(source):
                print_current(f"📁 Found file using absolute path: {source}")
                return source
        
        # Type 2: Try relative to workspace/sensor_data
        sensor_data_path = os.path.join(self.output_dir, source)
        if os.path.isfile(sensor_data_path):
            print_current(f"📁 Found file in sensor_data directory: {sensor_data_path}")
            return sensor_data_path
        
        # Type 3: Try relative to workspace root
        workspace_path = os.path.join(self.workspace_root, source)
        if os.path.isfile(workspace_path):
            print_current(f"📁 Found file in workspace directory: {workspace_path}")
            return workspace_path
        
        # Type 4: Try relative to current working directory (fallback)
        if os.path.isfile(source):
            current_path = os.path.abspath(source)
            print_current(f"📁 Found file in current directory: {current_path}")
            return current_path
        
        # Additional search: Look in common subdirectories
        search_dirs = [
            self.workspace_root,
            self.output_dir,
            os.path.join(self.workspace_root, 'workspace'),
            os.path.join(self.workspace_root, 'workspace', 'sensor_data'),
            os.getcwd()
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                candidate_path = os.path.join(search_dir, os.path.basename(source))
                if os.path.isfile(candidate_path):
                    print_current(f"📁 Found file by basename search in {search_dir}: {candidate_path}")
                    return candidate_path
        
        # File not found in any location
        print_current(f"❌ File not found in any search location: {source}")
        print_current(f"🔍 Searched locations:")
        print_current(f"   - Absolute: {source if os.path.isabs(source) else 'N/A'}")
        print_current(f"   - Sensor data: {sensor_data_path}")
        print_current(f"   - Workspace: {workspace_path}")
        print_current(f"   - Current dir: {os.path.abspath(source)}")
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print_current(f"   - {search_dir}: {os.path.join(search_dir, os.path.basename(source))}")
        
        return None
    
    def get_sensor_data(self, type: int, source: str, para: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Acquire sensor data from various sources.
        
        Args:
            type: Data type (1=image, 2=video, 3=audio, 4=sensor)
            source: Source identifier (file path or device path)
            para: Parameters dictionary (can include 'vision_mode' for smart optimization)
            
        Returns:
            Dictionary containing acquisition results
        """
        if para is None:
            para = {}
        
        try:
            print_current(f"🔍 Acquiring sensor data: type={type}, source={source}")
            
            # Check for vision mode optimization
            vision_mode = para.get('vision_mode', 'auto')  # auto, full, reference
            if vision_mode == 'auto':
                vision_mode = self._detect_vision_api_support()
            
            # Validate type parameter
            if type not in [1, 2, 3, 4]:
                return self._create_error_result("Invalid type parameter. Must be 1, 2, 3, or 4.")
            
            # Route to appropriate handler based on type
            if type == 1:
                return self._acquire_image_data(source, para, vision_mode)
            elif type == 2:
                return self._acquire_video_data(source, para)
            elif type == 3:
                return self._acquire_audio_data(source, para)
            elif type == 4:
                return self._acquire_sensor_data(source, para)
            else:
                return self._create_error_result("Invalid type parameter. Must be 1, 2, 3, or 4.")
                
        except Exception as e:
            print_current(f"❌ Error acquiring sensor data: {e}")
            return self._create_error_result(f"Failed to acquire sensor data: {str(e)}")
    
    def _detect_vision_api_support(self) -> str:
        """
        Detect whether the system supports the vision API.

        Returns:
            'reference' if vision API is supported, 'full' otherwise
        """
        # Check if the current model supports the Vision API
        try:
            model_name = ""
            
            # Prefer model information passed in via the constructor
            if self.model:
                model_name = str(self.model).lower()
            else:
                # Next, check environment variables
                import os
                model_name = os.environ.get('LLM_MODEL', '').lower()
                
                # If the environment variable is empty, try to get the current model info from tool_executor
                if not model_name:
                    try:
                        # Check if in tool_executor context
                        import inspect
                        frame = inspect.currentframe()
                        while frame:
                            if 'self' in frame.f_locals:
                                obj = frame.f_locals['self']
                                if hasattr(obj, 'model'):
                                    model_name = str(obj.model).lower()
                                    break
                            frame = frame.f_back
                    except:
                        pass
            
            # List of models that support vision - extend to include more models as needed
            vision_models = [
                'claude-3', 'claude-4', 'claude-sonnet', 'claude-opus', 'claude-haiku',
                'gpt-4', 'gpt-4.1', 'gpt-4o', 'gpt-4-vision', 'gpt-4-turbo',
                'gemini', 'gemini-pro', 'gemini-vision'
            ]
            
            if any(vision_model in model_name for vision_model in vision_models):
                print_current(f"🖼️ Vision API support detected (model: {model_name}), enabling smart optimization mode")
                return 'reference'
            else:
                print_current(f"📄 Vision API not detected (model: {model_name}), using full data mode")
                return 'full'
        except Exception as e:
            # Default to full mode to ensure compatibility
            print_current(f"📄 Vision API detection failed ({str(e)}), using full data mode")
            return 'full'
    def _acquire_image_data(self, source: str, para: Dict[str, Any], vision_mode: str = 'full') -> Dict[str, Any]:
        """Acquire image data from camera or file with vision-aware optimization."""
        resolution = para.get('resolution', '640x320')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if source is camera device first (to avoid unnecessary file searching)
        if source.startswith('/dev/') or source.startswith('video') or source.isdigit():
            return self._capture_image_from_camera(source, resolution, timestamp, vision_mode)
        
        # Try to resolve file path intelligently (supports multiple path types)
        resolved_path = self._resolve_file_path(source)
        if resolved_path:
            return self._load_image_from_file(resolved_path, resolution, vision_mode)
        
        return self._create_error_result(f"Invalid image source: {source}")
    
    def _acquire_video_data(self, source: str, para: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire video data from camera or file."""
        resolution = para.get('resolution', '640x320')
        duration = para.get('duration', 5)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if source is camera device first (to avoid unnecessary file searching)
        if source.startswith('/dev/') or source.startswith('video') or source.isdigit():
            return self._capture_video_from_camera(source, resolution, duration, timestamp)
        
        # Try to resolve file path intelligently (supports multiple path types)
        resolved_path = self._resolve_file_path(source)
        if resolved_path:
            return self._load_video_from_file(resolved_path)
        
        return self._create_error_result(f"Invalid video source: {source}")
    
    def _acquire_audio_data(self, source: str, para: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire audio data from microphone or file."""
        sampling_rate = para.get('sampling_rate', 16000)
        duration = para.get('duration', 5)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Try to resolve file path intelligently (supports multiple path types)
        resolved_path = self._resolve_file_path(source)
        if resolved_path:
            return self._load_audio_from_file(resolved_path)
        
        # Try to capture from audio device
        if source.startswith('/dev/') or source.startswith('audio') or 'mic' in source.lower():
            return self._capture_audio_from_microphone(source, sampling_rate, duration, timestamp)
        
        return self._create_error_result(f"Invalid audio source: {source}")
    
    def _acquire_sensor_data(self, source: str, para: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire sensor data from various sensors."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Try to resolve file path intelligently (supports multiple path types)
        resolved_path = self._resolve_file_path(source)
        if resolved_path:
            return self._load_sensor_from_file(resolved_path)
        
        # Try to read from sensor device
        if source.startswith('/dev/') or source.startswith('/sys/'):
            return self._read_sensor_device(source, para, timestamp)
        
        return self._create_error_result(f"Invalid sensor source: {source}")
    
    def _load_image_from_file(self, filepath: str, target_resolution: Optional[str] = None, vision_mode: str = 'full') -> Dict[str, Any]:
        """Load image from file and convert to base64, with optional resolution adjustment."""
        try:
            # Check if we need to resize the image
            should_resize = target_resolution and target_resolution != "original"
            
            if should_resize:
                # Try to resize image using PIL
                try:
                    from PIL import Image
                    import io
                    
                    # Parse target resolution
                    if not target_resolution:
                        raise ValueError("Target resolution is required for resizing")
                    width, height = map(int, target_resolution.split('x'))
                    
                    # Open and resize image
                    with Image.open(filepath) as img:
                        # Convert to RGB if necessary (for JPEG output)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        # Resize image maintaining aspect ratio (compatible with different PIL versions)
                        try:
                            # Calculate the best fit size that maintains aspect ratio
                            original_width, original_height = img.size
                            original_aspect = original_width / original_height
                            target_aspect = width / height
                            
                            if original_aspect > target_aspect:
                                # Original image is wider, fit to width
                                new_width = width
                                new_height = int(width / original_aspect)
                            else:
                                # Original image is taller, fit to height
                                new_height = height
                                new_width = int(height * original_aspect)
                            
                            print_current(f"📐 Aspect ratio preserved: {original_width}x{original_height} → {new_width}x{new_height}")
                            
                            # Try to get LANCZOS constant (value = 1)
                            if hasattr(Image, 'Resampling'):
                                # New PIL API
                                resample = Image.Resampling.LANCZOS  # type: ignore
                            else:
                                # Old PIL API, use numeric constant (LANCZOS = 1)
                                resample = 1
                            img_resized = img.resize((new_width, new_height), resample)
                        except Exception:
                            # Fall back to basic resize maintaining aspect ratio
                            original_width, original_height = img.size
                            original_aspect = original_width / original_height
                            target_aspect = width / height
                            
                            if original_aspect > target_aspect:
                                new_width = width
                                new_height = int(width / original_aspect)
                            else:
                                new_height = height
                                new_width = int(height * original_aspect)
                            
                            img_resized = img.resize((new_width, new_height))
                        
                        # Save to bytes buffer
                        buffer = io.BytesIO()
                        img_resized.save(buffer, format='JPEG', quality=85, optimize=True)
                        image_data = buffer.getvalue()
                        
                        print_current(f"📸 Image resized from {img.size} to {img_resized.size}")
                        actual_resolution = f"{img_resized.size[0]}x{img_resized.size[1]}"
                        
                except ImportError:
                    print_current("⚠️ PIL not available, loading image without resizing")
                    # Fall back to original image
                    with open(filepath, 'rb') as f:
                        image_data = f.read()
                    actual_resolution = "original"
                except Exception as e:
                    print_current(f"⚠️ Image resize failed: {e}, loading original")
                    # Fall back to original image
                    with open(filepath, 'rb') as f:
                        image_data = f.read()
                    actual_resolution = "original"
            else:
                # Load original image
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                actual_resolution = "original"
            
            # Encode to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Get file format
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                format_type = 'image/jpeg'
            elif file_ext == '.png':
                format_type = 'image/png'
            elif file_ext in ['.bmp', '.bitmap']:
                format_type = 'image/bmp'
            else:
                format_type = 'image/unknown'
            
            result_info = f"📸 Successfully loaded image from file: {filepath}"
            if should_resize:
                result_info += f" (resized to {target_resolution})"
            print_current(result_info)
            
            # Add file path metadata tag before base64 data (for history optimizer identification)
            source_path_marker = f"[FILE_SOURCE:{filepath}]"
            marked_base64_data = f"{source_path_marker}{base64_data}"
            
            if vision_mode == 'reference':
                # For vision mode, return a minimal reference format
                return {
                    'status': 'success',
                    'data': marked_base64_data, # Still include full base64 for vision API
                    'dataformat': f'base64 encoded {format_type}',
                    'source': filepath,
                    'type': 'image_file',
                    'file_size': len(image_data),
                    'original_file_size': os.path.getsize(filepath),
                    'resolution': actual_resolution if should_resize else 'original',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # For full mode, return the full base64 data
                return {
                    'status': 'success',
                    'data': marked_base64_data,
                    'dataformat': f'base64 encoded {format_type}',
                    'source': filepath,
                    'type': 'image_file',
                    'file_size': len(image_data),
                    'original_file_size': os.path.getsize(filepath),
                    'resolution': actual_resolution if should_resize else 'original',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return self._create_error_result(f"Failed to load image from file: {str(e)}")
    
    def _capture_image_from_camera(self, source: str, resolution: str, timestamp: str, vision_mode: str = 'full') -> Dict[str, Any]:
        """Capture image from camera using system tools with minimal dependencies."""
        try:
            # Ensure sensor data directory exists when needed
            self._ensure_sensor_directory()
            
            # Parse resolution
            width, height = resolution.split('x')
            
            # Generate output filename
            output_file = os.path.join(self.output_dir, f"image_{timestamp}.jpg")
            
            # Device number extraction - ensure correct handling of source='0' case
            if source.isdigit():
                device_num = source  # 保持字符串格式用于命令行
                device_int = int(source)  # Numeric format for logging
            elif source.startswith('/dev/video'):
                device_num = source.replace('/dev/video', '')
                device_int = int(device_num)
            elif source.startswith('video'):
                device_num = source.replace('video', '')
                device_int = int(device_num)
            else:
                device_num = '0'  # Default to first camera
                device_int = 0
            
            print_current(f"📸 Attempting to capture image from camera {device_int}")
            
            # Try different capture methods
            capture_success = False
            
            # Method 1: Try fswebcam (Linux) - most commonly used and minimal dependencies
            if os.name == 'posix':
                try:
                    cmd = [
                        'fswebcam', 
                        '-d', f'/dev/video{device_num}',
                        '-r', f'{width}x{height}',
                        '--no-banner',
                        '--skip', '5',  # Skip first 5 frames to ensure camera stability
                        output_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        capture_success = True
                        print_current(f"📸 Image captured successfully using fswebcam")
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass  # Try next method
            
            # Method 2: Try ffmpeg (cross-platform support)
            if not capture_success:
                try:
                    if os.name == 'posix':
                        # Linux/macOS
                        cmd = [
                            'ffmpeg',
                            '-f', 'v4l2',
                            '-i', f'/dev/video{device_num}',
                            '-frames:v', '1',
                            '-s', f'{width}x{height}',
                            '-y',
                            output_file
                        ]
                    else:
                        # Windows
                        cmd = [
                            'ffmpeg',
                            '-f', 'dshow',
                            '-i', f'video="USB Video Device"',
                            '-frames:v', '1',
                            '-s', f'{width}x{height}',
                            '-y',
                            output_file
                        ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        capture_success = True
                        print_current(f"📸 Image captured successfully using ffmpeg")
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass  # Try next method
            
            # Method 3: Try imagesnap (macOS)
            if not capture_success and os.uname().sysname == 'Darwin':
                try:
                    # First get available camera device list
                    list_result = subprocess.run(['imagesnap', '-l'], capture_output=True, text=True, timeout=5)
                    available_cameras = []
                    
                    if list_result.returncode == 0:
                        lines = list_result.stdout.strip().split('\n')
                        for line in lines:
                            if '=>' in line:
                                # Find default camera
                                camera_name = line.split('=>')[1].strip()
                                available_cameras.append((camera_name, True))  # True indicates default
                            elif line.strip() and not line.startswith('Video Devices:'):
                                # Other cameras
                                camera_name = line.strip()
                                if camera_name:
                                    available_cameras.append((camera_name, False))
                    
                    # Select camera based on device number
                    if available_cameras:
                        if device_int == 0 and any(is_default for _, is_default in available_cameras):
                            # Use default camera (device 0)
                            selected_camera = next(name for name, is_default in available_cameras if is_default)
                            cmd = ['imagesnap', '-d', selected_camera, output_file]
                        elif device_int < len(available_cameras):
                            # Use camera with specified index
                            selected_camera = available_cameras[device_int][0]
                            cmd = ['imagesnap', '-d', selected_camera, output_file]
                        else:
                            # Fallback to numeric device number
                            cmd = ['imagesnap', '-d', device_num, output_file]
                    else:
                        # No camera list found
                        cmd = ['imagesnap', output_file]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        capture_success = True
                        print_current(f"📸 Image captured successfully using imagesnap")
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass  # Continue to error handling
            
            if capture_success:
                # Load the captured image
                with open(output_file, 'rb') as f:
                    image_data = f.read()
                
                # Convert to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # Add file path metadata tag before base64 data (for history optimizer identification)
                file_path_marker = f"[FILE_SAVED:{output_file}]"
                marked_base64_data = f"{file_path_marker}{base64_data}"
                
                if vision_mode == 'reference':
                    # For vision mode, return a minimal reference format
                    return {
                        'status': 'success',
                        'data': marked_base64_data, # Still include full base64 for vision API
                        'dataformat': 'base64 encoded image/jpeg',
                        'source': source,
                        'type': 'camera_capture',
                        'resolution': resolution,
                        'file_path': output_file,
                        'file_size': len(image_data),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # For full mode, return the full base64 data
                    return {
                        'status': 'success',
                        'data': marked_base64_data,
                        'dataformat': 'base64 encoded image/jpeg',
                        'source': source,
                        'type': 'camera_capture',
                        'resolution': resolution,
                        'file_path': output_file,
                        'file_size': len(image_data),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return self._create_error_result("Failed to capture image: No suitable capture method available")
                
        except Exception as e:
            return self._create_error_result(f"Failed to capture image from camera: {str(e)}")
    
    def _capture_video_from_camera(self, source: str, resolution: str, duration: int, timestamp: str) -> Dict[str, Any]:
        """Capture video from camera using system tools with minimal dependencies."""
        try:
            # Ensure sensor data directory exists when needed
            self._ensure_sensor_directory()
            
            # Parse resolution
            width, height = resolution.split('x')
            
            # Generate output filename
            output_file = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
            
            # Device number extraction - ensure correct handling of source='0' case
            if source.isdigit():
                device_num = source  # 保持字符串格式用于命令行
                device_int = int(source)  # Numeric format for logging
            elif source.startswith('/dev/video'):
                device_num = source.replace('/dev/video', '')
                device_int = int(device_num)
            elif source.startswith('video'):
                device_num = source.replace('video', '')
                device_int = int(device_num)
            else:
                device_num = '0'
                device_int = 0
            
            print_current(f"🎥 Attempting to capture {duration}s video from camera {device_int}")
            
            # Try different capture methods
            capture_success = False
            
            # Method 1: Try ffmpeg (most commonly used video capture tool)
            try:
                if os.name == 'posix':
                    # Linux/macOS
                    cmd = [
                        'ffmpeg',
                        '-f', 'v4l2',
                        '-i', f'/dev/video{device_num}',
                        '-t', str(duration),
                        '-s', f'{width}x{height}',
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',  # Fast encoding
                        '-y',
                        output_file
                    ]
                else:
                    # Windows
                    cmd = [
                        'ffmpeg',
                        '-f', 'dshow',
                        '-i', f'video="USB Video Device"',
                        '-t', str(duration),
                        '-s', f'{width}x{height}',
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',
                        '-y',
                        output_file
                    ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 15)
                
                if result.returncode == 0 and os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    capture_success = True
                    print_current(f"🎥 Video captured successfully using ffmpeg")
                    
                    return {
                        'status': 'success',
                        'data': output_file,
                        'dataformat': 'MP4 video file',
                        'source': source,
                        'type': 'video_capture',
                        'resolution': resolution,
                        'duration': duration,
                        'file_path': output_file,
                        'file_size': file_size,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Continue to error handling
            
            # If all methods fail
            if not capture_success:
                return self._create_error_result(f"Failed to capture video: No suitable capture method available")
                
        except Exception as e:
            return self._create_error_result(f"Failed to capture video from camera: {str(e)}")
    
    def _capture_audio_from_microphone(self, source: str, sampling_rate: int, duration: int, timestamp: str) -> Dict[str, Any]:
        """Capture audio from microphone."""
        try:
            # Ensure sensor data directory exists when needed
            self._ensure_sensor_directory()
            
            # Generate output filename
            output_file = os.path.join(self.output_dir, f"audio_{timestamp}.wav")
            
            # Try different audio capture methods
            capture_success = False
            
            # Method 1: Try arecord (ALSA)
            if os.name == 'posix':
                try:
                    cmd = [
                        'arecord',
                        '-D', 'default',
                        '-f', 'S16_LE',
                        '-r', str(sampling_rate),
                        '-d', str(duration),
                        output_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 5)
                    if result.returncode == 0 and os.path.exists(output_file):
                        capture_success = True
                        print_current(f"🎤 Audio captured using arecord")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            # Method 2: Try ffmpeg
            if not capture_success:
                try:
                    cmd = [
                        'ffmpeg',
                        '-f', 'pulse',
                        '-i', 'default',
                        '-t', str(duration),
                        '-ar', str(sampling_rate),
                        '-ac', '1',
                        '-y',
                        output_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 5)
                    if result.returncode == 0 and os.path.exists(output_file):
                        capture_success = True
                        print_current(f"🎤 Audio captured using ffmpeg")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            if capture_success:
                file_size = os.path.getsize(output_file)
                return {
                    'status': 'success',
                    'data': output_file,
                    'dataformat': 'WAV audio file',
                    'source': source,
                    'type': 'audio_capture',
                    'sampling_rate': sampling_rate,
                    'duration': duration,
                    'file_path': output_file,
                    'file_size': file_size,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._create_error_result("Failed to capture audio: No suitable capture method available")
                
        except Exception as e:
            return self._create_error_result(f"Failed to capture audio from microphone: {str(e)}")
    
    def _load_video_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load video from file."""
        try:
            file_size = os.path.getsize(filepath)
            file_ext = os.path.splitext(filepath)[1].lower()
            
            print_current(f"🎥 Successfully loaded video from file: {filepath}")
            return {
                'status': 'success',
                'data': filepath,
                'dataformat': f'{file_ext.upper()} video file',
                'source': filepath,
                'type': 'video_file',
                'file_path': filepath,
                'file_size': file_size,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return self._create_error_result(f"Failed to load video from file: {str(e)}")
    
    def _load_audio_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load audio from file."""
        try:
            file_size = os.path.getsize(filepath)
            file_ext = os.path.splitext(filepath)[1].lower()
            
            print_current(f"🎤 Successfully loaded audio from file: {filepath}")
            return {
                'status': 'success',
                'data': filepath,
                'dataformat': f'{file_ext.upper()} audio file',
                'source': filepath,
                'type': 'audio_file',
                'file_path': filepath,
                'file_size': file_size,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return self._create_error_result(f"Failed to load audio from file: {str(e)}")
    
    def _load_sensor_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load sensor data from file."""
        try:
            with open(filepath, 'r') as f:
                sensor_data = f.read()
            
            # Try to parse as JSON
            try:
                parsed_data = json.loads(sensor_data)
                dataformat = "JSON sensor data"
            except json.JSONDecodeError:
                parsed_data = sensor_data
                dataformat = "Raw sensor data"
            
            print_current(f"📊 Successfully loaded sensor data from file: {filepath}")
            return {
                'status': 'success',
                'data': parsed_data,
                'dataformat': dataformat,
                'source': filepath,
                'type': 'sensor_file',
                'file_path': filepath,
                'file_size': len(sensor_data),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return self._create_error_result(f"Failed to load sensor data from file: {str(e)}")
    
    def _read_sensor_device(self, source: str, para: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Read sensor data from device."""
        try:
            # Check if the device/file exists
            if not os.path.exists(source):
                return self._create_error_result(f"Sensor source does not exist: {source}")
            
            # Try to read from the device
            with open(source, 'r') as f:
                sensor_data = f.read().strip()
            
            # Try to parse as JSON
            try:
                parsed_data = json.loads(sensor_data)
                dataformat = "JSON sensor data from device"
            except json.JSONDecodeError:
                # Try to parse as numeric value
                try:
                    parsed_data = float(sensor_data)
                    dataformat = "Numeric sensor value"
                except ValueError:
                    parsed_data = sensor_data
                    dataformat = "Raw sensor data from device"
            
            print_current(f"📊 Successfully read sensor data from device: {source}")
            return {
                'status': 'success',
                'data': parsed_data,
                'dataformat': dataformat,
                'source': source,
                'type': 'sensor_device',
                'raw_data': sensor_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return self._create_error_result(f"Failed to read sensor device: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'status': 'failed',
            'data': None,
            'dataformat': None,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        } 