#!/usr/bin/env python3
"""
Setup Validation Script for Phone Use Agent

This script checks if the system is properly configured for optimal
performance with CUDA GPUs and Android devices.
"""

import sys
import subprocess
import json
from pathlib import Path


def check_cuda():
    """Check CUDA availability and version."""
    print("\n=== CUDA Configuration ===")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            
            # Check each GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
                
                # Check if Flash Attention is likely supported (Ampere+, compute 8.0+)
                if props.major >= 8:
                    print(f"    ✓ Flash Attention 2 supported (Ampere/Ada/Hopper)")
                else:
                    print(f"    ⚠ Flash Attention 2 may not be optimal (pre-Ampere)")
                
                # Check TF32 support (Ampere+)
                if props.major >= 8:
                    print(f"    ✓ TF32 supported for matmul acceleration")
                    print(f"    TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        else:
            print("✗ CUDA is not available")
            print("  The agent will run on CPU (very slow)")
            return False
            
    except ImportError:
        print("✗ PyTorch is not installed")
        print("  Install with: pip install torch torchvision torchaudio")
        return False
    
    return True


def check_vllm():
    """Check vLLM installation."""
    print("\n=== vLLM Configuration ===")
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
        
        # Check for xformers (used by vLLM for optimizations)
        try:
            import xformers
            print(f"✓ xformers version: {xformers.__version__}")
        except ImportError:
            print("⚠ xformers not found (optional but recommended)")
            
    except ImportError:
        print("✗ vLLM is not installed")
        print("  Install with: pip install vllm")
        return False
    
    return True


def check_adb():
    """Check ADB installation and connected devices."""
    print("\n=== ADB Configuration ===")
    try:
        # Check ADB version
        result = subprocess.run(
            "adb version",
            shell=True, capture_output=True, text=True, check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ ADB installed: {version_line}")
        
        # Check connected devices
        result = subprocess.run(
            "adb devices",
            shell=True, capture_output=True, text=True, check=True
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            devices = []
            for line in lines[1:]:
                if '\t' in line:
                    device_info = line.split('\t')
                    if len(device_info) >= 2 and device_info[1].strip() == 'device':
                        devices.append(device_info[0].strip())
            
            if devices:
                print(f"✓ Found {len(devices)} connected device(s):")
                for device in devices:
                    print(f"  - {device}")
                    
                    # Get Android version
                    try:
                        ver_result = subprocess.run(
                            f"adb -s {device} shell getprop ro.build.version.release",
                            shell=True, capture_output=True, text=True, check=True
                        )
                        android_version = ver_result.stdout.strip()
                        
                        sdk_result = subprocess.run(
                            f"adb -s {device} shell getprop ro.build.version.sdk",
                            shell=True, capture_output=True, text=True, check=True
                        )
                        sdk_version = sdk_result.stdout.strip()
                        
                        print(f"    Android version: {android_version} (SDK {sdk_version})")
                        
                        if int(sdk_version) >= 35:
                            print(f"    ✓ Android 15+ detected - optimized fast screenshot capture")
                        elif int(sdk_version) >= 33:
                            print(f"    ✓ Android 13+ detected - fast screenshot capture supported")
                        else:
                            print(f"    ⚠ Android {android_version} - using traditional screenshot method")
                            
                    except Exception as e:
                        print(f"    Could not detect Android version: {e}")
                        
                return True
            else:
                print("✗ No authorized devices found")
                print("  Make sure USB debugging is enabled and device is authorized")
                return False
        else:
            print("✗ No devices connected")
            print("  Connect an Android device via USB")
            return False
            
    except subprocess.CalledProcessError:
        print("✗ ADB is not installed or not in PATH")
        print("  Install with: sudo apt install adb")
        return False
    except FileNotFoundError:
        print("✗ ADB command not found")
        print("  Install with: sudo apt install adb")
        return False


def check_omniparser():
    """Check OmniParser installation."""
    print("\n=== OmniParser Configuration ===")
    omniparser_path = Path("OmniParser")
    
    if not omniparser_path.exists():
        print("✗ OmniParser directory not found")
        print("  Clone with: git clone https://github.com/microsoft/OmniParser.git")
        return False
    
    print(f"✓ OmniParser directory found")
    
    # Check for model weights
    weights_path = omniparser_path / "weights"
    if not weights_path.exists():
        print("✗ OmniParser weights directory not found")
        print("  Download weights following the README instructions")
        return False
    
    # Check for required weight files
    required_files = [
        "icon_detect/model.pt",
        "icon_caption_florence/config.json",
        "icon_caption_florence/model.safetensors"
    ]
    
    all_present = True
    for file in required_files:
        file_path = weights_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} missing")
            all_present = False
    
    if not all_present:
        print("\n  Download missing weights following the README instructions")
        return False
    
    return True


def check_config():
    """Check configuration file."""
    print("\n=== Configuration File ===")
    config_path = Path("config.json")
    
    if not config_path.exists():
        print("✗ config.json not found")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print("✓ config.json is valid JSON")
        
        # Check CUDA config
        if "cuda_config" in config:
            print("✓ CUDA configuration present")
            cuda_config = config["cuda_config"]
            
            mem_util = cuda_config.get("gpu_memory_utilization", 0.90)
            print(f"  GPU memory utilization: {mem_util * 100:.0f}%")
            
            dtype = cuda_config.get("dtype", "bfloat16")
            print(f"  Data type: {dtype}")
            
            if cuda_config.get("enable_flash_attention", False):
                print(f"  ✓ Flash Attention enabled")
            
            if cuda_config.get("enable_prefix_caching", False):
                print(f"  ✓ Prefix caching enabled")
        else:
            print("⚠ CUDA configuration not found (using defaults)")
        
        # Check Android config
        if "android_config" in config:
            print("✓ Android configuration present")
            android_config = config["android_config"]
            target_version = android_config.get("target_android_version", 15)
            print(f"  Target Android version: {target_version}")
        else:
            print("⚠ Android configuration not found (using defaults)")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ config.json is invalid: {e}")
        return False


def check_python_packages():
    """Check required Python packages."""
    print("\n=== Python Packages ===")
    
    required_packages = [
        ("transformers", "Hugging Face Transformers"),
        ("qwen_vl_utils", "Qwen VL Utils"),
        ("PIL", "Pillow (PIL)"),
    ]
    
    all_present = True
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} not installed")
            all_present = False
    
    if not all_present:
        print("\nInstall missing packages with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main validation function."""
    print("=" * 60)
    print("Phone Use Agent - Setup Validation")
    print("=" * 60)
    
    results = {
        "Python Packages": check_python_packages(),
        "CUDA": check_cuda(),
        "vLLM": check_vllm(),
        "ADB": check_adb(),
        "OmniParser": check_omniparser(),
        "Configuration": check_config(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_good = True
    for component, status in results.items():
        status_str = "✓ OK" if status else "✗ NEEDS ATTENTION"
        print(f"{component:20s}: {status_str}")
        if not status:
            all_good = False
    
    print("=" * 60)
    
    if all_good:
        print("\n✓ All checks passed! Your setup is ready for optimal performance.")
        print("\nRecommended next steps:")
        print("1. Run a test task: python main.py --task \"Open Chrome\"")
        print("2. Monitor GPU usage during first run")
        print("3. Adjust cuda_config settings if needed")
        return 0
    else:
        print("\n⚠ Some components need attention. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
