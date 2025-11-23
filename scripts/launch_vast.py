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
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


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


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for Vast.ai instance management"
    )
    parser.add_argument(
        'action',
        choices=['search', 'config', 'test'],
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


if __name__ == '__main__':
    main()

