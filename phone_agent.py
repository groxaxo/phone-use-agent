import os
import json
import time
import logging
import tempfile
import subprocess
import base64
from datetime import datetime
from pathlib import Path
import ast

class PhoneAgent:
    """
    A phone agent that uses Qwen2.5-VL to analyze screenshots and control a physical 
    Android phone via ADB.
    """
    
    def __init__(self, config=None):
        """
        Initialize the phone agent with configuration.
        
        Args:
            config (dict): Configuration for the agent
        """
        default_config = {
            'device_id': None,  # Will use first connected device if None
            'screen_width': 1080,  # Pixel 5 dimensions
            'screen_height': 2340,
            'omniparser_path': './omniparser',
            'screenshot_dir': './screenshots',
            'max_retries': 3,
            'qwen_model_path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'use_gpu': True,
            'temperature': 0.1,
            # External provider configuration
            'use_external_provider': False,  # Set to True to use external API
            'external_provider': {
                'api_base': 'http://localhost:8000/v1',  # vLLM, LM Studio, etc.
                'api_key': None,  # Optional API key
                'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',  # Model name for API
                'timeout': 120  # Request timeout in seconds
            }
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
            # Deep merge for external_provider
            if 'external_provider' in config:
                self.config['external_provider'] = {
                    **default_config['external_provider'],
                    **config['external_provider']
                }
        
        self.context = {
            'previous_actions': [],
            'current_app': "Home",
            'current_state': "",
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"phone_agent_{self.context['session_id']}.log"),
                logging.StreamHandler()
            ]
        )

        # Initialize the VL agent (local or external)
        self._initialize_vl_agent()
        
        self._setup_directories()
        self._check_adb_connection()
    
    def _initialize_vl_agent(self):
        """
        Initialize the vision-language agent based on configuration.
        
        Uses external API provider if 'use_external_provider' is True,
        otherwise uses local vLLM-based Qwen model.
        """
        if self.config.get('use_external_provider', False):
            # Use external provider (vLLM server, LM Studio, etc.)
            from external_vl_agent import ExternalVLAgent
            
            ext_config = self.config.get('external_provider', {})
            logging.info(f"Initializing external VL agent with API: {ext_config.get('api_base')}")
            
            self.vl_agent = ExternalVLAgent(
                api_base=ext_config.get('api_base', 'http://localhost:8000/v1'),
                api_key=ext_config.get('api_key'),
                model_name=ext_config.get('model_name', self.config['qwen_model_path']),
                temperature=self.config['temperature'],
                max_tokens=ext_config.get('max_tokens', 1024),
                timeout=ext_config.get('timeout', 120)
            )
        else:
            # Use local vLLM-based Qwen model
            from qwen_vl_agent import QwenVLAgent
            
            logging.info("Initializing local Qwen VL agent... may take a while")
            self.vl_agent = QwenVLAgent(
                model_path=self.config['qwen_model_path'],
                use_gpu=self.config['use_gpu'],
                temperature=self.config['temperature'],
                cuda_config=self.config.get('cuda_config', None)
            )
    
    def _setup_directories(self):
        """Create necessary directories for storing screenshots."""
        Path(self.config['screenshot_dir']).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {self.config['screenshot_dir']}")
    
    def _check_adb_connection(self):
        """Verify ADB connection to the device."""
        try:
            # Check for connected devices
            result = subprocess.run(
                "adb devices",
                shell=True, check=True, capture_output=True, text=True
            )
            
            # Get the device ID
            if self.config['device_id'] is None:
                # Extract the first device from the list
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # First line is "List of devices attached"
                    device_info = lines[1].split('\t')
                    if len(device_info) > 0 and device_info[1].strip() == 'device':
                        self.config['device_id'] = device_info[0].strip()
                        logging.info(f"Using device: {self.config['device_id']}")
                    else:
                        raise Exception("No authorized device found")
                else:
                    raise Exception("No devices connected")
            
            # Test a simple ADB command to verify connection
            device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
            subprocess.run(
                f"adb {device_cmd_prefix} shell echo 'Connected'",
                shell=True, check=True, capture_output=True, text=True
            )
            
            # Detect Android version
            self._detect_android_version()
            
            logging.info("ADB connection verified")
        except subprocess.CalledProcessError as e:
            logging.error(f"ADB connection error: {e}")
            raise Exception("Failed to connect to device via ADB. Make sure USB debugging is enabled.")
    
    def _detect_android_version(self):
        """Detect the Android version of the connected device."""
        try:
            device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
            result = subprocess.run(
                f"adb {device_cmd_prefix} shell getprop ro.build.version.release",
                shell=True, check=True, capture_output=True, text=True
            )
            android_version = result.stdout.strip()
            
            # Get SDK version for more precise version detection
            sdk_result = subprocess.run(
                f"adb {device_cmd_prefix} shell getprop ro.build.version.sdk",
                shell=True, check=True, capture_output=True, text=True
            )
            sdk_version = int(sdk_result.stdout.strip())
            
            self.context['android_version'] = android_version
            self.context['android_sdk'] = sdk_version
            
            logging.info(f"Detected Android version: {android_version} (SDK {sdk_version})")
            
            # Android 15 is SDK 35
            if sdk_version >= 35:
                logging.info("Android 15+ detected - using optimized settings")
                self._apply_android_15_optimizations()
            elif sdk_version >= 33:
                logging.info("Android 13+ detected")
            
        except Exception as e:
            logging.warning(f"Could not detect Android version: {e}")
            self.context['android_version'] = "unknown"
            self.context['android_sdk'] = 0
    
    def _apply_android_15_optimizations(self):
        """Apply Android 15 specific optimizations."""
        android_config = self.config.get('android_config', {})
        
        # Enable predictive back gesture support
        if android_config.get('enable_gesture_nav', True):
            logging.info("Gesture navigation optimizations enabled for Android 15")
        
        logging.info("Applied Android 15 optimizations")
    
    def _run_adb_command(self, command):
        """
        Run an ADB command on the connected device.
        
        Args:
            command (str): ADB command to run
            
        Returns:
            str: Command output
        """
        device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
        full_command = f"adb {device_cmd_prefix} {command}"
        
        try:
            result = subprocess.run(
                full_command,
                shell=True, check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"ADB command error: {e}")
            logging.error(f"Command stderr: {e.stderr}")
            raise
    
    def capture_screen(self):
        """
        Capture screenshot from the device.
        Optimized for Android 15 with improved performance.
        
        Returns:
            str: Path to the saved screenshot
        """
        timestamp = int(time.time())
        android_config = self.config.get('android_config', {})
        screenshot_format = android_config.get('screenshot_format', 'png')
        screenshot_path = os.path.join(
            self.config['screenshot_dir'],
            f"screen_{self.context['session_id']}_{timestamp}.{screenshot_format}"
        )
        
        try:
            # Android 15 optimization: Use exec-out for faster screenshot capture
            # This avoids writing to device storage and is faster
            android_sdk = self.context.get('android_sdk', 0)
            
            if android_sdk >= 33:  # Android 13+ supports exec-out efficiently
                # Direct stream from device to file (faster for Android 13+)
                device_cmd_prefix = f"-s {self.config['device_id']}" if self.config['device_id'] else ""
                with open(screenshot_path, 'wb') as f:
                    result = subprocess.run(
                        f"adb {device_cmd_prefix} exec-out screencap -p",
                        shell=True, check=True, capture_output=True
                    )
                    f.write(result.stdout)
                logging.info(f"Screenshot captured using fast exec-out method")
            else:
                # Fallback to traditional method for older Android versions
                self._run_adb_command(f"shell screencap -p /sdcard/screenshot.png")
                self._run_adb_command(f"pull /sdcard/screenshot.png {screenshot_path}")
                self._run_adb_command(f"shell rm /sdcard/screenshot.png")
                logging.info(f"Screenshot captured using traditional method")
            
            logging.info(f"Screenshot saved to: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")
            # Fallback to traditional method if exec-out fails
            try:
                self._run_adb_command(f"shell screencap -p /sdcard/screenshot.png")
                self._run_adb_command(f"pull /sdcard/screenshot.png {screenshot_path}")
                self._run_adb_command(f"shell rm /sdcard/screenshot.png")
                logging.info(f"Screenshot captured using fallback method")
                return screenshot_path
            except Exception as fallback_error:
                logging.error(f"Fallback screenshot method also failed: {fallback_error}")
                raise
    
    def parse_screen(self, screenshot_path):
        """
        Run omniparser on the screenshot to extract UI elements.
        
        Args:
            screenshot_path (str): Path to the screenshot
            
        Returns:
            list: Processed screen elements
        """
        try:
            # Use our custom omniparser_runner.py script
            runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omniparser_runner.py")
            
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                output_path = tmp.name
            
            # Build the command with parameters from config
            cmd = f"python {runner_script} --input {screenshot_path} --output {output_path}"
            
            # Add optional parameters from config
            omniparser_config = self.config.get('omniparser_config', {})
            if omniparser_config.get('use_paddleocr', True):
                cmd += " --use_paddleocr"
            if 'box_threshold' in omniparser_config:
                cmd += f" --box_threshold {omniparser_config['box_threshold']}"
            if 'iou_threshold' in omniparser_config:
                cmd += f" --iou_threshold {omniparser_config['iou_threshold']}"
            if 'imgsz' in omniparser_config:
                cmd += f" --imgsz {omniparser_config['imgsz']}"
            
            # Log the command we're executing
            logging.info(f"Running OmniParser with command: {cmd}")
            
            # Execute the command
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            if result.stderr:
                logging.warning(f"Omniparser warnings: {result.stderr}")
            
            # Read and process the output
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    parsed_output = json.load(f)
                os.unlink(output_path)
                
                # Save the annotated image for debugging (optional)
                if 'annotated_image' in parsed_output:
                    debug_path = f"{os.path.splitext(screenshot_path)[0]}_annotated.png"
                    with open(debug_path, 'wb') as f:
                        f.write(base64.b64decode(parsed_output['annotated_image']))
                    logging.info(f"Saved annotated image to: {debug_path}")
                
                # Return the parsed elements
                elements = parsed_output.get('elements', [])
                logging.info(f"Detected {len(elements)} screen elements")
                
                # Convert elements to the expected format
                return self._process_omniparser_output("\n".join([f'icon {i}: {element}' for i, element in enumerate(elements)]))
            else:
                raise FileNotFoundError(f"Output file not found: {output_path}")
        
        except subprocess.CalledProcessError as e:
            logging.error(f"Error parsing screen with omniparser: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            raise
        except Exception as e:
            logging.error(f"Error in parse_screen: {e}")
            raise
    
    def _process_omniparser_output(self, output):
        """
        Process omniparser output and extract useful information.
        
        Args:
            output (str): Raw omniparser output
            
        Returns:
            list: Structured screen elements
        """
        screen_elements = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if not line.startswith('icon '):
                continue
            
            try:
                # Extract the icon number and content
                icon_match = line.split(':', 1)
                if len(icon_match) != 2:
                    continue
                
                icon_id = icon_match[0].replace('icon ', '')
                
                # Convert the content to proper dict format
                content_text = icon_match[1].strip()
                # Convert single quotes to double quotes for JSON parsing
                content_text = content_text.replace("'", '"')
                content_text = content_text.replace('True', 'true').replace('False', 'false')
                
                try:
                    content = json.loads(content_text)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for handling Python dict representation
                    content = ast.literal_eval(icon_match[1].strip())
                
                # Add the icon id
                content['id'] = int(icon_id)
                
                # Calculate center point for interaction
                bbox = content['bbox']
                center = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ]
                
                # Create a cleaner representation
                screen_element = {
                    'id': content['id'],
                    'type': content['type'],
                    'content': content['content'].strip(),
                    'interactivity': content.get('interactivity', False),
                    'position': {
                        'x': round(center[0], 3),
                        'y': round(center[1], 3)
                    },
                    'bbox': [round(coord, 3) for coord in bbox]
                }
                
                screen_elements.append(screen_element)
            except Exception as e:
                logging.error(f"Error processing line: {line}")
                logging.error(f"Exception: {e}")
                continue
        
        logging.info(f"Processed {len(screen_elements)} screen elements")
        return screen_elements
    
    def execute_action(self, action):
        """
        Execute the action on the device.
        
        Args:
            action (dict): Action to execute
            
        Returns:
            dict: Result of the action execution
        """
        try:
            # Manually convert 'click' to 'tap' for compatibility - sometimes the model will not strictly follow the prompt so we can force the correct action here 
            if action['action'] == 'click':
                logging.info(f"Converting 'click' action to 'tap'")
                action['action'] = 'tap'
            
            logging.info(f"Executing action: {action['action']} on element \"{action.get('elementName', 'unknown')}\"")

            # Fallback: Type → Tap (focus) if we never tapped it - prevents the model from trying to enter text before it has selected a text field
            if action['action'] == 'type':
                last_actions = self.context['previous_actions']
                
                # Make sure we at least have 'coordinates' in the original action JSON
                if 'coordinates' not in action:
                    # Without coordinates, we can’t convert "type" → "tap"
                    # so either skip or raise an error
                    msg = "No 'coordinates' given for a 'type' action. Skipping."
                    logging.info(msg)
                    return {
                        'success': False,
                        'error': msg,
                        'action': action
                    }
                
                # Also check if we tapped the same element last cycle
                if not last_actions or last_actions[-1].get('action') != 'tap' \
                or last_actions[-1].get('elementId') != action.get('elementId'):
                    
                    logging.info("Overriding 'type' action with a 'tap' first to focus text field.")
                    action = {
                        "action": "tap",
                        "elementId": action['elementId'],
                        "elementName": action.get('elementName', ''),
                        "coordinates": action['coordinates'],  # now we know 'coordinates' exist
                        "reasoning": "Need to tap first to focus the text field."
                    }

            # -------------------------------------------------------
            # Now handle whichever action we end up with
            # -------------------------------------------------------
            if action['action'] == 'tap':
                # Convert normalized coordinates to pixel coordinates
                x, y = self._translate_coordinates(
                    float(action['coordinates'][0]),
                    float(action['coordinates'][1])
                )
                # Execute the tap
                self._run_adb_command(f"shell input tap {x} {y}")
                logging.info(f"Tapped at coordinates ({x}, {y})")
            
            elif action['action'] == 'swipe':
                start_x = self.config['screen_width'] // 2
                start_y = self.config['screen_height'] // 2
                
                end_x, end_y = start_x, start_y
                direction = action['direction']
                
                if direction == 'up':
                    end_y = int(start_y * 0.3)
                elif direction == 'down':
                    end_y = int(start_y * 1.7)
                elif direction == 'left':
                    end_x = int(start_x * 0.3)
                elif direction == 'right':
                    end_x = int(start_x * 1.7)
                
                self._run_adb_command(f"shell input swipe {start_x} {start_y} {end_x} {end_y} 300")
                logging.info(f"Swiped {direction}")
            
            elif action['action'] == 'type':
                # Escape shell chars:
                escaped_text = action['text'].replace("'", "\\'").replace('"', '\\"')
                # Replace spaces with %s so the phone interprets them correctly - without this the text gets entered weird
                escaped_text = escaped_text.replace(" ", "%s")

                # Now run the command
                self._run_adb_command(f"shell input text \"{escaped_text}\"")
                logging.info(f"Typed text: \"{action['text']}\"")
            
            elif action['action'] == 'wait':
                wait_time = action.get('waitTime', 1000)  # Default to 1 second
                logging.info(f"Waiting for {wait_time}ms")
                time.sleep(wait_time / 1000)
            
            else:
                raise ValueError(f"Unknown action type: {action['action']}")
            
            return {
                'success': True,
                'action': action
            }
        
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action
            }

    
    def _translate_coordinates(self, normalized_x, normalized_y):
        """
        Translate normalized coordinates to actual pixel coordinates.
        
        Args:
            normalized_x (float): Normalized x coordinate (0-1)
            normalized_y (float): Normalized y coordinate (0-1)
            
        Returns:
            tuple: (x, y) pixel coordinates
        """
        x = int(normalized_x * self.config['screen_width'])
        y = int(normalized_y * self.config['screen_height'])
        return x, y
    
    def execute_cycle(self, user_request):
        """
        Execute a single interaction cycle.
        
        Args:
            user_request (str): User's request
            
        Returns:
            dict: Result of the action execution
        """
        try:
            # Step 1: Capture the screen
            screenshot_path = self.capture_screen()
            
            # Step 2: Parse the screen with omniparser
            screen_elements = self.parse_screen(screenshot_path)
            
            # Step 3: Use Qwen VL to analyze the screenshot and determine the action
            action = self.vl_agent.analyze_screenshot(
                screenshot_path,
                user_request,
                screen_elements,
                self.context
            )

            # 1) If there is an elementId but no coordinates, fill them from screen_elements
            if action.get("elementId") is not None and "coordinates" not in action:
                for elem in screen_elements:
                    if elem["id"] == action["elementId"]:
                        # Use the normalized position from OmniParser
                        x = elem["position"]["x"]
                        y = elem["position"]["y"]
                        action["coordinates"] = [x, y]
                        break
            
            if not action:
                raise Exception("Failed to determine action from screenshot")
            
            # Step 4: Execute the action on the device
            result = self.execute_action(action)
            
            # result['action'] is the final, possibly overridden action
            executed_action = result['action']

            # Update context with what was *actually* executed
            self.context['previous_actions'].append({
                'action': executed_action['action'],            # Store the final corrected action
                'elementId': executed_action.get('elementId'),  # etc.
                'elementName': executed_action.get('elementName', ''),
                'timestamp': time.time()
            })
            
            # If we're opening an app, update the context
            if (action['action'] == 'tap' and 
                action.get('elementName') and 
                action['elementName'] not in ['Circle', 'Dictate', 'Paste']):
                self.context['current_app'] = action['elementName']
            
            # Cleanup old screenshots to save disk space
            self._cleanup_old_screenshots()
            
            return result
        except Exception as e:
            logging.error(f"Error in execution cycle: {e}")
            raise
    
    def _cleanup_old_screenshots(self, keep_last=10):
        """
        Clean up old screenshots to save disk space, keeping only the most recent ones.
        
        Args:
            keep_last (int): Number of recent screenshots to keep
        """
        try:
            screenshot_dir = Path(self.config['screenshot_dir'])
            if not screenshot_dir.exists():
                return
            
            # Get screenshot format from config
            android_config = self.config.get('android_config', {})
            screenshot_format = android_config.get('screenshot_format', 'png')
            
            # Get all screenshots for this session with the configured format
            pattern = f"screen_{self.context['session_id']}_*.{screenshot_format}"
            screenshots = sorted(screenshot_dir.glob(pattern))
            
            # Delete older screenshots if we have more than keep_last
            if len(screenshots) > keep_last:
                for screenshot in screenshots[:-keep_last]:
                    screenshot.unlink()
                    # Also delete annotated versions
                    annotated = screenshot.with_name(screenshot.stem + f"_annotated.{screenshot_format}")
                    if annotated.exists():
                        annotated.unlink()
                logging.debug(f"Cleaned up {len(screenshots) - keep_last} old screenshots")
        except Exception as e:
            logging.warning(f"Error cleaning up screenshots: {e}")
    
    def execute_task(self, user_request, max_cycles=10):
        """
        Execute a task by running multiple interaction cycles.
        
        Args:
            user_request (str): User's request
            max_cycles (int): Maximum number of cycles to execute
            
        Returns:
            dict: Result of the task execution
        """
        logging.info(f"Starting task: \"{user_request}\"")
        
        # Log initial GPU memory if available
        self._log_gpu_memory("Task start")
        
        cycles = 0
        while cycles < max_cycles:
            try:
                result = self.execute_cycle(user_request)
                logging.info(f"Cycle {cycles + 1} completed: {result}")
                
                # Log GPU memory usage periodically
                if (cycles + 1) % 5 == 0:
                    self._log_gpu_memory(f"After cycle {cycles + 1}")
                
                # Check if we should continue or if the task is complete
                # This could be determined by asking the LLM if the task is complete
                # or by checking for specific elements on the screen
                
                cycles += 1
                
                # Wait a moment for the UI to update
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in cycle {cycles + 1}: {e}")
                
                # Retry logic
                if cycles < self.config['max_retries']:
                    logging.info(f"Retrying... ({cycles + 1}/{self.config['max_retries']})")
                    continue
                
                raise Exception(f"Failed to complete task after {cycles} cycles: {str(e)}")
        
        logging.info(f"Task completed after {cycles} cycles")
        self._log_gpu_memory("Task end")
        
        return {
            'success': True,
            'cycles': cycles,
            'context': self.context
        }
    
    def _log_gpu_memory(self, stage=""):
        """
        Log current GPU memory usage for monitoring and debugging.
        
        Args:
            stage (str): Description of current stage for logging context
        """
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logging.info(
                        f"GPU {i} Memory [{stage}]: "
                        f"Allocated: {allocated:.2f}GB, "
                        f"Reserved: {reserved:.2f}GB, "
                        f"Total: {total:.2f}GB"
                    )
        except Exception as e:
            logging.debug(f"Could not log GPU memory: {e}")


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load configuration from file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config.json: {e}")
        print("Using default configuration...")
        config = {
            'device_id': None,  # Will use first connected device
            'screen_width': 1080,
            'screen_height': 2340,
            'qwen_model_path': 'Qwen/Qwen2.5-VL-3B-Instruct'
        }
    
    # Create agent with loaded config
    agent = PhoneAgent(config)
    
    try:
        result = agent.execute_task('Open Chrome and search for the weather in New York')
        print('Task execution result:', result)
    except Exception as e:
        print(f"Task execution failed: {e}")
