"""
Checkpoint management with local and remote (S3) storage support.
"""
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import torch


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and syncing to persistent storage.
    
    Supports:
    - Local checkpoint storage
    - S3/remote storage upload/download
    - Automatic checkpoint versioning
    - Best model tracking
    """
    
    def __init__(
        self,
        local_dir: Path,
        remote_bucket: Optional[str] = None,
        remote_prefix: str = 'checkpoints',
        keep_last_n: int = 5
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            local_dir: Local directory for checkpoints
            remote_bucket: S3 bucket name (optional)
            remote_prefix: Prefix for remote checkpoints
            keep_last_n: Number of checkpoints to keep locally
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.remote_bucket = remote_bucket
        self.remote_prefix = remote_prefix
        self.keep_last_n = keep_last_n
        
        # Metadata file
        self.metadata_file = self.local_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {'checkpoints': [], 'best_checkpoint': None}
        return {'checkpoints': [], 'best_checkpoint': None}
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(
        self,
        checkpoint: Dict[str, Any],
        step: int,
        is_best: bool = False,
        suffix: Optional[str] = None
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            step: Training step number
            is_best: Whether this is the best model so far
            suffix: Optional suffix for checkpoint name
        
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint filename
        if suffix:
            filename = f"checkpoint_step_{step}_{suffix}.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"
        
        checkpoint_path = self.local_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            'path': str(checkpoint_path),
            'step': step,
            'epoch': checkpoint.get('epoch', 0),
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best,
            'val_loss': checkpoint.get('best_val_loss', None)
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Update best checkpoint
        if is_best:
            self.metadata['best_checkpoint'] = checkpoint_info
        
        # Keep only last N checkpoints
        if len(self.metadata['checkpoints']) > self.keep_last_n:
            # Remove oldest checkpoint
            oldest = self.metadata['checkpoints'].pop(0)
            oldest_path = Path(oldest['path'])
            if oldest_path.exists() and not oldest['is_best']:
                oldest_path.unlink()
        
        self._save_metadata()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load a checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        if not self.metadata['checkpoints']:
            return None
        
        latest = self.metadata['checkpoints'][-1]
        checkpoint_path = Path(latest['path'])
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint file not found: {checkpoint_path}")
            return None
        
        return self.load(checkpoint_path)
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        if self.metadata['best_checkpoint'] is None:
            return None
        
        best_path = Path(self.metadata['best_checkpoint']['path'])
        
        if not best_path.exists():
            print(f"Warning: Best checkpoint file not found: {best_path}")
            return None
        
        return self.load(best_path)
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists"""
        return len(self.metadata['checkpoints']) > 0
    
    def upload_latest(self) -> bool:
        """
        Upload latest checkpoint to remote storage (S3).
        
        Returns:
            True if upload successful, False otherwise
        """
        if self.remote_bucket is None:
            print("No remote bucket configured, skipping upload")
            return False
        
        if not self.has_checkpoint():
            print("No checkpoint to upload")
            return False
        
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            print("boto3 not installed. Install with: pip install boto3")
            return False
        
        latest = self.metadata['checkpoints'][-1]
        checkpoint_path = Path(latest['path'])
        
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            s3_client = boto3.client('s3')
            
            # Upload checkpoint
            remote_key = f"{self.remote_prefix}/{checkpoint_path.name}"
            s3_client.upload_file(
                str(checkpoint_path),
                self.remote_bucket,
                remote_key
            )
            
            # Upload metadata
            metadata_key = f"{self.remote_prefix}/checkpoint_metadata.json"
            s3_client.upload_file(
                str(self.metadata_file),
                self.remote_bucket,
                metadata_key
            )
            
            print(f"Checkpoint uploaded to s3://{self.remote_bucket}/{remote_key}")
            return True
        
        except ClientError as e:
            print(f"Failed to upload checkpoint to S3: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error uploading checkpoint: {e}")
            return False
    
    def download_latest(self) -> bool:
        """
        Download latest checkpoint from remote storage (S3).
        
        Returns:
            True if download successful, False otherwise
        """
        if self.remote_bucket is None:
            print("No remote bucket configured, skipping download")
            return False
        
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            print("boto3 not installed. Install with: pip install boto3")
            return False
        
        try:
            s3_client = boto3.client('s3')
            
            # List checkpoints in remote
            response = s3_client.list_objects_v2(
                Bucket=self.remote_bucket,
                Prefix=self.remote_prefix
            )
            
            if 'Contents' not in response:
                print("No checkpoints found in remote storage")
                return False
            
            # Find latest checkpoint
            checkpoints = [obj for obj in response['Contents'] 
                          if obj['Key'].endswith('.pt')]
            
            if not checkpoints:
                print("No checkpoint files found in remote storage")
                return False
            
            latest = max(checkpoints, key=lambda x: x['LastModified'])
            remote_key = latest['Key']
            local_path = self.local_dir / Path(remote_key).name
            
            # Download checkpoint
            s3_client.download_file(
                self.remote_bucket,
                remote_key,
                str(local_path)
            )
            
            # Download metadata if available
            metadata_key = f"{self.remote_prefix}/checkpoint_metadata.json"
            try:
                s3_client.download_file(
                    self.remote_bucket,
                    metadata_key,
                    str(self.metadata_file)
                )
                self.metadata = self._load_metadata()
            except:
                pass
            
            print(f"Checkpoint downloaded from s3://{self.remote_bucket}/{remote_key}")
            return True
        
        except ClientError as e:
            print(f"Failed to download checkpoint from S3: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        return self.metadata['checkpoints'].copy()
    
    def cleanup_old_checkpoints(self, keep_n: Optional[int] = None):
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_n: Number of checkpoints to keep (defaults to self.keep_last_n)
        """
        if keep_n is None:
            keep_n = self.keep_last_n
        
        checkpoints = self.metadata['checkpoints']
        
        if len(checkpoints) <= keep_n:
            return
        
        # Keep best checkpoint and latest N-1
        best_path = None
        if self.metadata['best_checkpoint']:
            best_path = Path(self.metadata['best_checkpoint']['path'])
        
        # Sort by step
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x['step'])
        
        # Remove old checkpoints (except best)
        removed = 0
        for checkpoint_info in checkpoints_sorted[:-keep_n]:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and checkpoint_path != best_path:
                checkpoint_path.unlink()
                removed += 1
        
        # Update metadata
        self.metadata['checkpoints'] = checkpoints_sorted[-keep_n:]
        self._save_metadata()
        
        print(f"Cleaned up {removed} old checkpoints")

