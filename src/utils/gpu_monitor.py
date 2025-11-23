"""
GPU monitoring and cost tracking utilities for Vast.ai instances.
"""
import torch
import subprocess
import os
from typing import Dict, Optional
from datetime import datetime


def get_gpu_info() -> Dict[str, any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary with GPU information including name, memory, utilization
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'name': 'No GPU',
            'total_memory_gb': 0,
            'free_memory_gb': 0,
            'used_memory_gb': 0,
            'utilization_percent': 0
        }
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    # Get memory info
    total_memory = props.total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved
    
    # Try to get utilization from nvidia-smi
    utilization = 0
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            utilization = int(result.stdout.strip())
    except:
        pass
    
    return {
        'available': True,
        'name': props.name,
        'total_memory_gb': total_memory / 1e9,
        'free_memory_gb': free_memory / 1e9,
        'used_memory_gb': reserved / 1e9,
        'allocated_memory_gb': allocated / 1e9,
        'utilization_percent': utilization,
        'compute_capability': f"{props.major}.{props.minor}",
        'multiprocessor_count': props.multi_processor_count
    }


def estimate_batch_size(
    model_params: int = 1e6,
    sequence_length: int = 512,
    dtype_bytes: int = 4,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate maximum batch size based on available GPU memory.
    
    Args:
        model_params: Number of model parameters
        sequence_length: Sequence length (for transformers) or input size
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
        safety_factor: Safety factor to leave memory headroom (0.8 = 80% of memory)
    
    Returns:
        Estimated maximum batch size
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info['available']:
        return 1
    
    # Rough estimation:
    # - Model parameters: model_params * dtype_bytes
    # - Activations: batch_size * sequence_length * hidden_size * dtype_bytes
    # - Gradients: same as parameters
    # - Optimizer states: 2x parameters (for Adam)
    
    available_memory_bytes = gpu_info['free_memory_gb'] * 1e9 * safety_factor
    
    # Model memory (parameters + gradients + optimizer states)
    model_memory = model_params * dtype_bytes * 4  # params + grads + 2x optimizer states
    
    # Available for activations
    activation_memory = available_memory_bytes - model_memory
    
    # Rough estimate: assume hidden_size ~= sqrt(model_params) for transformers
    hidden_size = int(model_params ** 0.5)
    
    # Memory per sample (very rough estimate)
    memory_per_sample = sequence_length * hidden_size * dtype_bytes * 4  # 4x for activations
    
    if memory_per_sample <= 0:
        return 1
    
    batch_size = int(activation_memory / memory_per_sample)
    return max(1, min(batch_size, 128))  # Cap at 128 for safety


def get_instance_cost(
    hourly_rate: float,
    runtime_hours: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate instance cost information.
    
    Args:
        hourly_rate: Cost per hour in USD
        runtime_hours: Optional runtime in hours (if None, calculates per-hour metrics)
    
    Returns:
        Dictionary with cost information
    """
    gpu_info = get_gpu_info()
    
    # Estimate TFLOPs (very rough, based on GPU name)
    tflops = estimate_tflops(gpu_info.get('name', ''))
    
    cost_info = {
        'hourly_rate_usd': hourly_rate,
        'estimated_tflops': tflops,
        'usd_per_tflop': hourly_rate / tflops if tflops > 0 else 0
    }
    
    if runtime_hours is not None:
        cost_info['total_cost_usd'] = hourly_rate * runtime_hours
        cost_info['runtime_hours'] = runtime_hours
    
    return cost_info


def estimate_tflops(gpu_name: str) -> float:
    """
    Rough estimate of TFLOPs based on GPU name.
    This is a simplified lookup - actual TFLOPs depend on precision and workload.
    """
    gpu_name_lower = gpu_name.lower()
    
    # Rough estimates for common GPUs (FP32)
    tflop_estimates = {
        'rtx 4090': 83,
        'rtx 3090': 36,
        'rtx 3080': 30,
        'rtx 4080': 49,
        'a100': 19.5,  # FP32, much higher for FP16/TF32
        'a6000': 38.7,
        'v100': 15.7,
        't4': 8.1,
        'p100': 9.3,
        'k80': 2.9
    }
    
    for key, tflops in tflop_estimates.items():
        if key in gpu_name_lower:
            return tflops
    
    # Default estimate based on memory (very rough)
    if 'gb' in gpu_name_lower:
        # Try to extract memory size
        import re
        match = re.search(r'(\d+)\s*gb', gpu_name_lower)
        if match:
            memory_gb = int(match.group(1))
            # Rough estimate: ~2-3 TFLOPs per GB
            return memory_gb * 2.5
    
    return 10.0  # Default fallback


def print_gpu_status():
    """Print current GPU status to console"""
    gpu_info = get_gpu_info()
    
    print("\n" + "="*60)
    print("GPU Status")
    print("="*60)
    
    if not gpu_info['available']:
        print("No GPU available")
        return
    
    print(f"GPU: {gpu_info['name']}")
    print(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"Free Memory: {gpu_info['free_memory_gb']:.2f} GB")
    print(f"Used Memory: {gpu_info['used_memory_gb']:.2f} GB")
    print(f"Utilization: {gpu_info['utilization_percent']}%")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print("="*60 + "\n")


def recommend_instance_scale(
    current_hourly_rate: float,
    current_tflops: float,
    target_training_time_hours: float,
    max_budget: float
) -> Dict[str, any]:
    """
    Recommend whether to scale up or down based on budget and time constraints.
    
    Args:
        current_hourly_rate: Current instance hourly rate
        current_tflops: Current instance TFLOPs
        target_training_time_hours: Desired training completion time
        max_budget: Maximum budget for training
    
    Returns:
        Dictionary with recommendations
    """
    current_cost = current_hourly_rate * target_training_time_hours
    
    recommendation = {
        'current_cost': current_cost,
        'within_budget': current_cost <= max_budget,
        'recommendation': 'keep_current',
        'reason': ''
    }
    
    if current_cost > max_budget:
        # Need cheaper instance or longer time
        recommendation['recommendation'] = 'scale_down'
        recommendation['reason'] = f'Current cost ${current_cost:.2f} exceeds budget ${max_budget:.2f}'
    elif current_cost < max_budget * 0.5:
        # Could afford faster instance
        max_affordable_rate = max_budget / target_training_time_hours
        recommendation['recommendation'] = 'scale_up'
        recommendation['reason'] = f'Could afford up to ${max_affordable_rate:.2f}/hr for faster training'
        recommendation['max_affordable_rate'] = max_affordable_rate
    
    return recommendation


if __name__ == '__main__':
    # Test GPU monitoring
    print_gpu_status()
    
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        batch_size = estimate_batch_size(model_params=100e6, sequence_length=512)
        print(f"Recommended batch size: {batch_size}")
        
        cost_info = get_instance_cost(hourly_rate=0.50)
        print(f"\nCost Info:")
        print(f"  Hourly Rate: ${cost_info['hourly_rate_usd']:.2f}/hr")
        print(f"  Estimated TFLOPs: {cost_info['estimated_tflops']:.1f}")
        print(f"  $/TFLOP: ${cost_info['usd_per_tflop']:.4f}")

