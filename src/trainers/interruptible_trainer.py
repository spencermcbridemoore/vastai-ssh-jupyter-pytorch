"""
Interruptible Trainer for Vast.ai preemptible instances.

Handles graceful interruption, checkpointing, and resumption of training.
"""
import torch
import torch.nn as nn
import signal
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import time
from datetime import datetime


class InterruptibleTrainer:
    """
    Trainer that handles preemption and interruption gracefully.
    
    Features:
    - Automatic checkpointing every N steps
    - Signal handling for SIGTERM (preemption)
    - State restoration from checkpoints
    - Automatic upload to persistent storage
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_manager,
        config: Dict[str, Any],
        loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.loss_fn = loss_fn
        
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Interruption handling
        self.interrupted = False
        self.checkpoint_interval = config.get('checkpoint_interval', 1000)
        self.save_best = config.get('save_best', True)
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def _handle_interrupt(self, signum, frame):
        """Handle interruption signal (SIGTERM from preemption or Ctrl+C)"""
        print(f"\n{'='*60}")
        print(f"Received interrupt signal {signum}")
        print(f"Current epoch: {self.current_epoch}, Step: {self.global_step}")
        print(f"Saving checkpoint before exit...")
        print(f"{'='*60}\n")
        
        self.interrupted = True
        self._save_checkpoint(is_final=True)
        
        # Upload checkpoint if possible
        try:
            self.checkpoint_manager.upload_latest()
            print("Checkpoint uploaded to persistent storage")
        except Exception as e:
            print(f"Warning: Failed to upload checkpoint: {e}")
        
        sys.exit(0)
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer"""
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]):
        """Set the learning rate scheduler"""
        self.scheduler = scheduler
    
    def set_loss_fn(self, loss_fn: Callable):
        """Set the loss function"""
        self.loss_fn = loss_fn
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint"""
        if self.model is None or self.optimizer is None:
            print("Warning: Model or optimizer not set, skipping checkpoint")
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'is_final': is_final
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save(
            checkpoint,
            step=self.global_step,
            is_best=(self.best_val_loss == min(self.val_losses) if self.val_losses else False)
        )
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load training state from checkpoint"""
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")
    
    def train_step(self, batch) -> float:
        """
        Execute a single training step.
        
        Returns:
            loss value
        """
        if self.model is None or self.optimizer is None or self.loss_fn is None:
            raise ValueError("Model, optimizer, and loss_fn must be set before training")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        else:
            batch = batch.to(self.device)
        
        # Forward pass
        outputs = self.model(batch[0] if isinstance(batch, (list, tuple)) else batch)
        
        # Compute loss
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            targets = batch[1]
        elif isinstance(batch, dict) and 'target' in batch:
            targets = batch['target']
        else:
            # Assume outputs contain everything needed for loss
            targets = None
        
        loss = self.loss_fn(outputs, targets) if targets is not None else self.loss_fn(outputs)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if specified
        if self.config.get('grad_clip', None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
        
        self.optimizer.step()
        
        self.global_step += 1
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """
        Run validation loop.
        
        Returns:
            average validation loss
        """
        if self.model is None or self.loss_fn is None:
            raise ValueError("Model and loss_fn must be set before validation")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch[0] if isinstance(batch, (list, tuple)) else batch)
                
                # Compute loss
                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    targets = batch[1]
                elif isinstance(batch, dict) and 'target' in batch:
                    targets = batch['target']
                else:
                    targets = None
                
                loss = self.loss_fn(outputs, targets) if targets is not None else self.loss_fn(outputs)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self,
        train_loader,
        val_loader: Optional[Any] = None,
        start_epoch: int = 0,
        max_epochs: int = 100,
        checkpoint_interval: Optional[int] = None,
        log_interval: int = 100
    ):
        """
        Main training loop with interruption handling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            start_epoch: Starting epoch (for resumption)
            max_epochs: Maximum number of epochs
            checkpoint_interval: Steps between checkpoints (overrides config)
            log_interval: Steps between logging
        """
        if checkpoint_interval is not None:
            self.checkpoint_interval = checkpoint_interval
        
        self.current_epoch = start_epoch
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch}")
        print(f"Checkpoint interval: {self.checkpoint_interval} steps")
        print(f"Max epochs: {max_epochs}")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(start_epoch, max_epochs):
                if self.interrupted:
                    print("Training interrupted, exiting...")
                    break
                
                self.current_epoch = epoch
                epoch_loss = 0.0
                num_batches = 0
                
                print(f"\nEpoch {epoch + 1}/{max_epochs}")
                epoch_start_time = time.time()
                
                for batch_idx, batch in enumerate(train_loader):
                    if self.interrupted:
                        break
                    
                    # Training step
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Logging
                    if self.global_step % log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                        print(f"Step {self.global_step}: Loss={avg_loss:.4f}, LR={lr:.2e}")
                        
                        # Log to wandb if available
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({
                                    'train/loss': avg_loss,
                                    'train/learning_rate': lr,
                                    'train/epoch': epoch,
                                    'train/step': self.global_step
                                })
                        except:
                            pass
                    
                    # Periodic checkpointing
                    if self.global_step % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                
                # End of epoch
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                self.train_losses.append(avg_epoch_loss)
                
                # Validation
                if val_loader is not None:
                    val_loss = self.validate(val_loader)
                    self.val_losses.append(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss and self.save_best:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                    
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch + 1} complete: Train Loss={avg_epoch_loss:.4f}, "
                          f"Val Loss={val_loss:.4f}, Time={epoch_time:.2f}s")
                    
                    # Log to wandb
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'train/epoch_loss': avg_epoch_loss,
                                'val/loss': val_loss,
                                'val/epoch': epoch,
                                'val/best_loss': self.best_val_loss
                            })
                    except:
                        pass
                else:
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch + 1} complete: Loss={avg_epoch_loss:.4f}, "
                          f"Time={epoch_time:.2f}s")
                
                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # End of epoch checkpoint
                self._save_checkpoint()
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            self._handle_interrupt(signal.SIGINT, None)
        
        finally:
            # Final checkpoint
            if not self.interrupted:
                print("\nTraining complete, saving final checkpoint...")
                self._save_checkpoint(is_final=True)
                
                # Upload final checkpoint
                try:
                    self.checkpoint_manager.upload_latest()
                    print("Final checkpoint uploaded to persistent storage")
                except Exception as e:
                    print(f"Warning: Failed to upload final checkpoint: {e}")

