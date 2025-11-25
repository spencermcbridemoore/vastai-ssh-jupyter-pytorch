#!/usr/bin/env python3
"""
Helper script to find and connect to Vast.ai instances.

This script helps you:
1. Search for available Vast.ai instances
2. Get connection details
3. Generate SSH config entries
4. Launch instances (if using Vast.ai API)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request


def search_instances(
    min_gpu_memory: int = 8,
    max_price: float = 1.0,
    gpu_name_filter: Optional[str] = None
) -> List[Dict]:
    """
    Search for available Vast.ai instances.
    
    Note: This requires the Vast.ai API or CLI tool.
    For now, this is a placeholder that shows the structure.
    """
    print("Searching for Vast.ai instances...")
    print(f"  Min GPU Memory: {min_gpu_memory} GB")
    print(f"  Max Price: ${max_price}/hr")
    if gpu_name_filter:
        print(f"  GPU Filter: {gpu_name_filter}")
    
    # TODO: Implement actual Vast.ai API integration
    # Example using vastai CLI (if installed):
    # result = subprocess.run(
    #     ['vastai', 'search', 'offers', '--raw'],
    #     capture_output=True,
    #     text=True
    # )
    # instances = json.loads(result.stdout)
    
    print("\nNote: Vast.ai API integration not implemented.")
    print("Please use the Vast.ai web interface to find instances.")
    print("Then use the SSH details to update your ~/.ssh/config")
    
    return []


def generate_ssh_config(
    instance_id: str,
    hostname: str,
    port: int,
    username: str = "root",
    local_tensorboard_port: int = 8080,
    local_jupyter_port: int = 8888
) -> str:
    """
    Generate SSH config entry for a Vast.ai instance.
    """
    config = f"""
Host vastai-{instance_id}
    HostName {hostname}
    Port {port}
    User {username}
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/control-%h-%p-%r
    ControlPersist 10m
    LocalForward {local_tensorboard_port} localhost:6006
    LocalForward {local_jupyter_port} localhost:8888
    ServerAliveInterval 60
    ServerAliveCountMax 3
"""
    return config


def test_connection(host: str) -> bool:
    """
    Test SSH connection to a host.
    """
    print(f"Testing connection to {host}...")
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', host, 'echo "Connection successful"'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Connection to {host} successful")
            return True
        else:
            print(f"✗ Connection to {host} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Connection to {host} timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing connection: {e}")
        return False


def load_env(env_path: Path) -> Dict[str, str]:
    """Simple .env parser (avoids external dependency)."""
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_vast_api_key(repo_root: Path) -> Optional[str]:
    """Return VASTAI_API_KEY from environment or .env file."""
    api_key = os.getenv("VASTAI_API_KEY")
    if api_key:
        return api_key

    env_file = repo_root / ".env"
    env_data = load_env(env_file)
    return env_data.get("VASTAI_API_KEY")


def list_instances(api_key: str) -> List[Dict]:
    """Query Vast.ai for the authenticated user's instances."""
    url = "https://console.vast.ai/api/v0/instances/"
    req = request.Request(url, headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"})
    with request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    return payload.get("instances", [])


def print_instances(instances: List[Dict], include_all: bool = False) -> None:
    """Format instance list for terminal output."""
    if not instances:
        print("No instances found for this account.")
        return

    running = [inst for inst in instances if inst.get("actual_status") == "running"]
    targets = instances if include_all else running

    if not targets:
        print("No running instances. Use --all to see stopped instances.")
        return

    print(f"Found {len(targets)} instance(s) ({'all' if include_all else 'running only'}):")
    for inst in targets:
        status = inst.get("actual_status")
        gpu = inst.get("gpu_name")
        inst_id = inst.get("id")
        hourly = inst.get("instance", {}).get("totalHour") or inst.get("total_cost_per_hour")
        host = inst.get("public_ipaddr") or inst.get("ssh_host")
        ssh_port = inst.get("ssh_port")
        label = inst.get("label") or "n/a"
        print("-" * 70)
        print(f"ID: {inst_id} | Status: {status} | Label: {label}")
        print(f"GPU: {gpu} ({inst.get('cuda_max_good')} compute) | RAM: {inst.get('gpu_ram')} MB")
        print(f"Region: {inst.get('geolocation')} | Provider: {inst.get('host_id')}")
        print(f"Hourly rate: ${hourly:.4f}" if isinstance(hourly, (float, int)) else f"Hourly rate: {hourly}")
        print(f"SSH: ssh -p {ssh_port} root@{host}")
        print(f"Started: {inst.get('start_date')}")


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for Vast.ai instance management"
    )
    parser.add_argument(
        'action',
        choices=['search', 'config', 'test', 'status'],
        help='Action to perform'
    )
    parser.add_argument(
        '--instance-id',
        help='Instance ID for config/test actions'
    )
    parser.add_argument(
        '--hostname',
        help='Hostname/IP for SSH config generation'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=22,
        help='SSH port (default: 22)'
    )
    parser.add_argument(
        '--min-memory',
        type=int,
        default=8,
        help='Minimum GPU memory in GB for search'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=1.0,
        help='Maximum price per hour for search'
    )
    parser.add_argument(
        '--host',
        help='Host alias to test connection'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Include non-running instances when using the status action'
    )
    
    args = parser.parse_args()
    
    if args.action == 'search':
        instances = search_instances(
            min_gpu_memory=args.min_memory,
            max_price=args.max_price
        )
        if instances:
            print(f"\nFound {len(instances)} instances:")
            for inst in instances:
                print(f"  - {inst}")
    
    elif args.action == 'config':
        if not args.instance_id or not args.hostname:
            print("Error: --instance-id and --hostname required for config action")
            sys.exit(1)
        
        config = generate_ssh_config(
            instance_id=args.instance_id,
            hostname=args.hostname,
            port=args.port
        )
        print("\nSSH Config Entry:")
        print(config)
        print("\nAdd this to your ~/.ssh/config file")
    
    elif args.action == 'test':
        if not args.host:
            print("Error: --host required for test action")
            sys.exit(1)
        
        success = test_connection(args.host)
        sys.exit(0 if success else 1)
    
    elif args.action == 'status':
        repo_root = Path(__file__).resolve().parents[1]
        api_key = get_vast_api_key(repo_root)
        if not api_key:
            print("Error: VASTAI_API_KEY not set. Add it to your environment or .env file.")
            sys.exit(1)
        try:
            instances = list_instances(api_key)
            print_instances(instances, include_all=args.all)
        except Exception as exc:
            print(f"Failed to query Vast.ai API: {exc}")
            sys.exit(1)


if __name__ == '__main__':
    main()

