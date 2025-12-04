import base64
import json
import logging
import re
from typing import Optional

import requests


class ExternalVLAgent:
    """
    Vision-Language LLM integration for the phone agent using external API providers.
    
    This class enables connecting to external VL model servers that expose OpenAI-compatible
    APIs, such as:
    - vLLM (with `--api-key` and `--served-model-name` options)
    - LM Studio
    - Ollama (with OpenAI-compatible endpoint)
    - Any other OpenAI-compatible vision-language model server
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120
    ):
        """
        Initialize the external VL agent.
        
        Args:
            api_base: Base URL for the OpenAI-compatible API endpoint
            api_key: API key for authentication (optional, depends on provider)
            model_name: Name of the model to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # System prompt (same as local agent)
        self.system_prompt = """
	You are a phone UI automation agent. Your task is to analyze phone screenshots and determine
	the next action to take based on the user's request. You will be shown a single screenshot
	of a phone screen along with information about interactive elements.
    
    IMPORTANT UI RULES:
    1. If you need to enter text into a text field, you MUST first 'tap' that text field (even if it appears selected) in one cycle. 
    2. On the *next* cycle, you can 'type' into that field. Never 'type' without a prior 'tap' on the same element.

	IMPORTANT: 
	1. You must respond ONLY with a JSON object containing a single action to perform.
	2. Valid actions are 'tap', 'swipe', 'type', and 'wait'.
	3. For tap actions, you must include the element ID and coordinates.
	4. Include a brief reasoning explaining why you chose this action.
        """
        
        logging.info(f"Initialized external VL agent with API base: {self.api_base}")
        logging.info(f"Using model: {self.model_name}")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image to base64 for the API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def _build_headers(self) -> dict:
        """Build HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def analyze_screenshot(self, screenshot_path: str, user_request: str, 
                          screen_elements: list, context: dict = None) -> Optional[dict]:
        """
        Analyze a screenshot and determine the next action using external API.
        
        Args:
            screenshot_path: Path to the screenshot image
            user_request: The user's request
            screen_elements: List of interactive elements detected by omniparser
            context: Current context information
            
        Returns:
            Action dictionary or None if failed
        """
        # Encode the screenshot
        img_b64 = self._encode_image(screenshot_path)
        
        # Format the screen elements for the prompt
        formatted_elements = "\n".join([
            f"ID: {el['id']} | Type: {el['type']} | Content: \"{el['content']}\" | "
            f"Position: ({el['position']['x']}, {el['position']['y']}) | "
            f"Interactive: {el.get('interactivity', False)}"
            for el in screen_elements
        ])
        
        # Create context description
        if context:
            context_info = [
                f"Previous actions: {', '.join([str(a) for a in context.get('previous_actions', [])])}",
                f"Current app: {context.get('current_app', 'Unknown')}",
                f"Current state: {context.get('current_state', '')}"
            ]
            context_description = "\n".join(context_info)
        else:
            context_description = "No prior context"
        
        # Build the user message
        user_message = f"""
        # Phone Screen Analysis
        
        ## User Request
        "{user_request}"
        
        ## Context
        {context_description}
        
        ## Screen Elements
        {formatted_elements}
        
        ## Instructions
        Analyze the screen and determine a single action to take. Provide your response as a JSON object with the following structure:
        {{
          "action": "tap" | "swipe" | "type" | "wait",
          "elementId": number,  // The ID of the element to interact with (only for tap)
          "elementName": string,  // The name of the element (for reference)
          "coordinates": [x, y],  // For tap or swipe actions (normalized 0-1 coordinates)
          "direction": "up" | "down" | "left" | "right",  // Only for swipe
          "text": string,  // Only for type action
          "waitTime": number,  // In milliseconds, only for wait action
          "reasoning": string  // A brief explanation of why this action was chosen
        }}
        """
        
        # Build the OpenAI-compatible request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.9
        }
        
        try:
            # Make the API request
            endpoint = f"{self.api_base}/chat/completions"
            logging.info(f"Sending request to external API: {endpoint}")
            
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text
            if not result.get("choices") or not result["choices"][0].get("message"):
                logging.error("Invalid response format from API")
                logging.error(f"Response: {result}")
                return None
            
            generated_text = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            return self._parse_response(generated_text)
            
        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error to {self.api_base}: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            logging.error(f"Response: {e.response.text if e.response else 'No response'}")
            return None
        except Exception as e:
            logging.error(f"Error calling external API: {e}")
            return None
    
    def _parse_response(self, generated_text: str) -> Optional[dict]:
        """
        Parse the model's response to extract the action JSON.
        
        Args:
            generated_text: Raw text from the model
            
        Returns:
            Parsed action dictionary or None
        """
        try:
            # Try to extract JSON if it's within a code block or other formatting
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', generated_text)
            if json_match:
                action_json = json.loads(json_match.group(1).strip())
            else:
                # Try to parse the whole text as JSON
                action_json = json.loads(generated_text.strip())
            
            logging.info(f"Generated action: {action_json}")
            return action_json
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error parsing model output: {e}")
            logging.error(f"Raw output: {generated_text}")
            return None
