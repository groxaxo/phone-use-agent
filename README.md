# Phone Use Agent

An experimental Python agent that controls Android phones using Qwen2.5-VL, OmniParser, and ADB.

![Phone Use Agent Architecture](docs/workflow.png)

## Overview

The Phone Use Agent automates interactions with Android devices by:
- Taking screenshots via ADB
- Analyzing UI elements with OmniParser
- Making decisions with Qwen2.5-VL vision language model through vLLM
- Executing actions (tap, swipe, type) through ADB

## Requirements

- Python 3.10
- Linux operating system
- Android Debug Bridge (ADB)
- CUDA-capable GPU (Optimized for CUDA 12.4+, tested on 30xx/40xx GPUs)
- Connected Android device with USB debugging enabled (Android 10-15 supported, optimized for Android 15)

## Installing ADB on Linux

ADB is required for the Phone Agent to communicate with your Android device. Install it on Linux with:

```bash
sudo apt update
sudo apt install adb
```

Verify the installation with:
```bash
adb version
```

## Setup with OmniParser

1. Clone this repository:
   ```bash
   git clone https://github.com/OminousIndustries/phone-use-agent.git
   cd phone-use-agent
   ```

2. Clone OmniParser into the phone-use-agent directory:
   ```bash
   git clone https://github.com/microsoft/OmniParser.git
   ```

3. Create and activate conda environment:
   ```bash
   conda create -n "phone_agent" python==3.10
   conda activate phone_agent
   ```

4. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download OmniParser weights:
   ```bash
   cd OmniParser

   # Create a folder for icon_detect but NOT icon_caption_florence:
   mkdir -p weights/icon_detect

   # Download weights from HF
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do
       huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
   done

   # Rename the icon_caption -> icon_caption_florence
   mv weights/icon_caption weights/icon_caption_florence
   ```

6. Return to main directory:
   ```bash
   cd ..
   ```

## Device Configuration

**Important:** You must set the correct screen resolution for your specific device in `config.json`. The default values are for a Pixel 5:

```json
{
  "screen_width": 1080,
  "screen_height": 2340
}
```

To find your device's resolution, run:
```bash
adb shell wm size
```

Update the values in `config.json` to match your device's resolution exactly. Incorrect resolution settings will cause the agent to tap in the wrong locations.

## Usage Options

### Command Line Interface

1. Connect your Android device via USB and enable USB debugging in Developer Options
2. Ensure conda environment is activated:
   ```bash
   conda activate phone_agent
   ```
3. Reccomended to run the first time through the CLI so we can see vLLM Qwen2.5VL download process
4. Run a task:
   ```bash
   python main.py --task "Open Chrome and search for weather in New York" --max-cycles 10
   ```

5. Additional options:
   ```bash
   python main.py --help
   ```

### Using External Providers (vLLM Server, LM Studio, etc.)

Instead of running the vision-language model locally, you can connect to external API providers that offer OpenAI-compatible endpoints. This is useful when:
- You want to run the model on a separate GPU server
- You're using cloud-hosted models
- You have limited local GPU memory

#### Supported Providers

- **vLLM Server**: Start vLLM with `vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-key <your-key>`
- **LM Studio**: Enable the local server in LM Studio settings
- **Ollama**: Use the OpenAI-compatible endpoint at `http://localhost:11434/v1`
- **Any OpenAI-compatible API**: Works with any provider that supports vision models

#### Command Line Usage

```bash
# Using vLLM server running locally
python main.py --task "Open Chrome" --external-provider --api-base http://localhost:8000/v1

# Using LM Studio
python main.py --task "Open Chrome" --external-provider --api-base http://localhost:1234/v1

# With API key authentication
python main.py --task "Open Chrome" --external-provider --api-base http://your-server:8000/v1 --api-key your-api-key

# Specifying a different model name
python main.py --task "Open Chrome" --external-provider --api-base http://localhost:8000/v1 --model-name my-model
```

#### Configuration File Usage

You can also configure external providers in `config.json`:

```json
{
  "use_external_provider": true,
  "external_provider": {
    "api_base": "http://localhost:8000/v1",
    "api_key": null,
    "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "timeout": 120
  }
}
```

### Graphical User Interface

A simple Gradio UI is provided to visualize the agent's progress:

```bash
python ui.py
```

The UI provides:
- Input field for your task
- View of the phone's screen at screenshot intervals
- Log output
- Auto-refresh functionality
- **External provider settings** - Configure and use external API providers directly from the UI

To use an external provider from the UI:
1. Expand the "External Provider Settings" accordion
2. Enable "Use External Provider"
3. Set the API Base URL to your server's endpoint (e.g., `http://localhost:8000/v1`)
4. Add an API key if required by your provider
5. Specify the model name if different from the default

## Configuration

Edit `config.json` to configure:
- Device dimensions (must match your actual device)
- Model selection (3B vs 7B)
- External provider settings
- **CUDA optimizations** (GPU memory utilization, precision, caching)
- **Android version compatibility** (Android 10-15 support)
- OmniParser settings
- General execution parameters

```json
{
  "device_id": null,
  "screen_width": 1080,
  "screen_height": 2340,
  "omniparser_path": "./OmniParser",
  "screenshot_dir": "./screenshots",
  "max_retries": 3,
  "qwen_model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
  "use_gpu": true,
  "temperature": 0.1,

  "use_external_provider": false,
  "external_provider": {
    "api_base": "http://localhost:8000/v1",
    "api_key": null,
    "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "timeout": 120
  },

  "cuda_config": {
    "gpu_memory_utilization": 0.90,
    "dtype": "bfloat16",
    "enable_flash_attention": true,
    "max_model_len": 32768,
    "enforce_eager": false,
    "enable_chunked_prefill": true,
    "tensor_parallel_size": 1,
    "enable_prefix_caching": true,
    "disable_custom_all_reduce": false
  },

  "android_config": {
    "min_android_version": 10,
    "target_android_version": 15,
    "enable_gesture_nav": true,
    "screenshot_format": "png",
    "use_scrcpy": false,
    "adb_timeout": 30
  },

  "omniparser_config": {
    "use_paddleocr": true,
    "box_threshold": 0.05,
    "iou_threshold": 0.1,
    "imgsz": 640,
    "device": "cuda",
    "half_precision": true,
    "enable_xformers": false
  }
}
```

### CUDA Configuration Options

The `cuda_config` section provides fine-grained control over GPU performance:

- **`gpu_memory_utilization`** (0.0-1.0): Fraction of GPU memory to use (default: 0.90)
  - Higher values allow larger context but may cause OOM errors
  - Lower values (0.70-0.85) recommended for dual-GPU setups or limited VRAM
- **`dtype`**: Model precision (default: "bfloat16")
  - `bfloat16`: Best balance of speed and accuracy for Ampere+ GPUs (recommended)
  - `float16`: Slightly faster but may have numerical stability issues
  - `float32`: Highest precision but slowest and uses most memory
- **`enable_flash_attention`**: Use Flash Attention 2 if available (default: true)
  - Significantly faster and more memory efficient for long sequences
  - Automatically enabled on compatible GPUs (Ampere/Ada/Hopper)
- **`enable_chunked_prefill`**: Process long prompts in chunks (default: true)
  - Reduces memory spikes during prefill phase
- **`enable_prefix_caching`**: Cache common prompt prefixes (default: true)
  - Speeds up repeated queries with similar prompts
- **`tensor_parallel_size`**: Number of GPUs for tensor parallelism (default: 1)
  - Set to 2 or more for multi-GPU inference
  - Requires multiple CUDA-capable GPUs

### Android Configuration Options

The `android_config` section provides Android version-specific optimizations:

- **`target_android_version`**: Target Android API level (default: 15)
  - Enables version-specific optimizations and features
- **`enable_gesture_nav`**: Support for gesture navigation (default: true)
  - Optimized for Android 10+ gesture navigation
- **`screenshot_format`**: Format for screenshots (default: "png")
  - "png": Lossless, larger files
  - "jpg": Lossy compression, smaller files (future support)
- **`adb_timeout`**: Timeout for ADB commands in seconds (default: 30)

**Android 15 Specific Optimizations:**
- Faster screenshot capture using `exec-out` streaming (no intermediate file)
- Automatic version detection and adaptive behavior
- Optimized for predictive back gestures and new permission models

## How It Works

The Phone Agent follows this workflow:

1. **User Request**: Define a task like "Open Chrome and search for weather"
2. **Capture**: Take a screenshot of the phone screen via ADB
3. **Analyze**: Use OmniParser to identify UI elements (buttons, text fields, icons)
4. **Decide**: Qwen2.5-VL analyzes screenshot and elements to determine next action
5. **Execute**: ADB performs the action (tap, swipe, type text)
6. **Repeat**: Continue the cycle until task completion or max cycles reached

The Main Controller manages execution cycles, tracks context between actions, handles errors, and implements retry logic when actions fail.

## Components

- **ADB Bridge**: Handles communication with the Android device
- **OmniParser**: Identifies interactive elements on the screen
- **Qwen VL Agent**: Makes decisions based on visual input and task context
- **Main Controller**: Orchestrates the execution cycles and manages state

## Troubleshooting

- **Wrong tap locations**: Verify your device resolution in `config.json` matches the actual device
- **ADB connection issues**: Make sure USB debugging is enabled and you've authorized the computer on your device
- **OmniParser errors**: Check that all model weights are correctly downloaded and placed in the proper directories
- **Gradio errors**: If using the UI, make sure you have gradio installed (`pip install gradio`)
- **OOM Errors from vLLM**: The Qwen2.5VL 3B and 7B models can take up a lot of memory. Solutions:
  - Reduce `gpu_memory_utilization` in `cuda_config` (try 0.70-0.85)
  - Use the 3B model instead of 7B: `"qwen_model_path": "Qwen/Qwen2.5-VL-3B-Instruct"`
  - For dual GPU setups: Uncomment `# os.environ["CUDA_VISIBLE_DEVICES"] = "1"` on line 21 of omniparser_runner.py to run OmniParser on GPU 1
  - Enable external provider mode to run the model on a separate server
- **Slow inference**: 
  - Ensure `enable_flash_attention` is `true` in `cuda_config`
  - Enable `enable_prefix_caching` for repeated queries
  - Verify GPU drivers are up to date
  - Check that CUDA 12.4+ is properly installed
- **Android 15 compatibility issues**:
  - The agent automatically detects Android version and applies optimizations
  - If screenshot capture fails, the agent falls back to traditional method automatically
  - Check ADB version is up to date: `adb version` (recommended: 34.0.0+) 

## Performance Optimization Tips

### For CUDA GPUs

1. **GPU Memory Optimization**:
   - Single GPU (24GB): Use default `gpu_memory_utilization: 0.90`
   - Dual GPU setup: Run vLLM on GPU 0 and OmniParser on GPU 1
   - Lower VRAM (16GB or less): Reduce to `0.70-0.80` or use external provider

2. **Inference Speed**:
   - Flash Attention 2 provides 2-3x speedup on Ampere+ GPUs (RTX 30xx/40xx)
   - Enable prefix caching for repeated prompts (10-20% faster)
   - Use chunked prefill to reduce memory spikes
   - Consider bfloat16 over float32 for 2x speed improvement

3. **Multi-GPU Setup**:
   - Set `tensor_parallel_size` to split model across multiple GPUs
   - Example: For 2x RTX 3090, set `"tensor_parallel_size": 2`

4. **TF32 Acceleration**:
   - Automatically enabled on Ampere+ GPUs for matmul operations
   - Provides up to 8x speedup with minimal accuracy impact

### For Android Devices

1. **Android 15 Optimizations**:
   - Faster screenshot capture using streaming (no temp files)
   - Automatic version detection and adaptive behavior
   - Reduced latency for all ADB operations

2. **General Tips**:
   - Keep device screen on during operation for consistent timing
   - Disable battery optimization for ADB to prevent connection drops
   - Use USB 3.0 ports for faster screenshot transfer
   - Clear device cache regularly if storage is limited

3. **Network Performance** (for external providers):
   - Run vLLM server on same machine or low-latency network
   - Use wired ethernet instead of WiFi when possible
   - Increase timeout for slow connections: `"timeout": 180`

## Experimental Status

This project is experimental and intended for research purposes. It may not work perfectly for all devices or UI layouts.

