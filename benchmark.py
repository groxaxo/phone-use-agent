#!/usr/bin/env python3
"""
Benchmark Script for Phone Use Agent

This script helps measure and compare performance with different
CUDA configurations.
"""

import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime


def run_benchmark(config_path="config.json", task="Open Chrome", cycles=3):
    """
    Run a benchmark of the phone agent with the current configuration.
    
    Args:
        config_path (str): Path to configuration file
        task (str): Task to benchmark
        cycles (int): Number of cycles to run
        
    Returns:
        dict: Benchmark results
    """
    from phone_agent import PhoneAgent
    
    print(f"\n{'='*60}")
    print(f"Phone Use Agent Benchmark")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Max cycles: {cycles}")
    print(f"  Model: {config.get('qwen_model_path', 'N/A')}")
    
    if config.get('use_external_provider'):
        print(f"  Mode: External Provider")
        ext_cfg = config.get('external_provider', {})
        print(f"  API Base: {ext_cfg.get('api_base', 'N/A')}")
    else:
        print(f"  Mode: Local vLLM")
        cuda_config = config.get('cuda_config', {})
        print(f"  GPU Memory Util: {cuda_config.get('gpu_memory_utilization', 0.90)*100:.0f}%")
        print(f"  Data Type: {cuda_config.get('dtype', 'bfloat16')}")
        print(f"  Flash Attention: {cuda_config.get('enable_flash_attention', True)}")
    
    # Initialize agent
    print(f"\nInitializing agent...")
    start_init = time.time()
    agent = PhoneAgent(config)
    init_time = time.time() - start_init
    print(f"  Initialization time: {init_time:.2f}s")
    
    # Benchmark GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU Memory (before task):")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
    except Exception as e:
        print(f"  Could not read GPU memory: {e}")
    
    # Run benchmark
    print(f"\nRunning task benchmark...")
    print(f"{'='*60}")
    
    cycle_times = []
    start_task = time.time()
    
    try:
        result = agent.execute_task(task, max_cycles=cycles)
        task_time = time.time() - start_task
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results:")
        print(f"{'='*60}")
        print(f"  Total time: {task_time:.2f}s")
        print(f"  Cycles completed: {result.get('cycles', 0)}")
        print(f"  Average time per cycle: {task_time / max(result.get('cycles', 1), 1):.2f}s")
        print(f"  Success: {result.get('success', False)}")
        
        # GPU memory after
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\nGPU Memory (after task):")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    print(f"  GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
        except Exception:
            pass
        
        # Save benchmark results
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": config.get('qwen_model_path'),
                "use_external_provider": config.get('use_external_provider', False),
                "cuda_config": config.get('cuda_config', {}),
            },
            "results": {
                "init_time": init_time,
                "task_time": task_time,
                "cycles": result.get('cycles', 0),
                "avg_cycle_time": task_time / max(result.get('cycles', 1), 1),
                "success": result.get('success', False),
            }
        }
        
        benchmark_file = Path(f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\nBenchmark data saved to: {benchmark_file}")
        
        return benchmark_data
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_configs(config1_path, config2_path, task="Open Chrome", cycles=3):
    """
    Compare performance between two different configurations.
    
    Args:
        config1_path (str): Path to first config
        config2_path (str): Path to second config
        task (str): Task to benchmark
        cycles (int): Number of cycles
    """
    print(f"\n{'='*60}")
    print(f"Configuration Comparison")
    print(f"{'='*60}")
    
    print(f"\nRunning benchmark with config 1: {config1_path}")
    result1 = run_benchmark(config1_path, task, cycles)
    
    print(f"\n\nRunning benchmark with config 2: {config2_path}")
    result2 = run_benchmark(config2_path, task, cycles)
    
    if result1 and result2:
        print(f"\n{'='*60}")
        print(f"Comparison Summary")
        print(f"{'='*60}")
        
        time1 = result1["results"]["task_time"]
        time2 = result2["results"]["task_time"]
        
        speedup = time1 / time2 if time2 > 0 else 0
        
        print(f"\nConfig 1: {time1:.2f}s")
        print(f"Config 2: {time2:.2f}s")
        
        if speedup > 1:
            print(f"\n✓ Config 2 is {speedup:.2f}x faster")
        elif speedup < 1:
            print(f"\n✓ Config 1 is {1/speedup:.2f}x faster")
        else:
            print(f"\nPerformance is similar")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Phone Use Agent")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--task", type=str, default="Open Chrome",
                       help="Task to benchmark")
    parser.add_argument("--cycles", type=int, default=3,
                       help="Number of cycles to run")
    parser.add_argument("--compare", type=str, default=None,
                       help="Compare with another config file")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configs(args.config, args.compare, args.task, args.cycles)
    else:
        run_benchmark(args.config, args.task, args.cycles)


if __name__ == "__main__":
    main()
