"""Training utilities for explainable GNN models."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from ..utils.device import get_device, set_seed, save_checkpoint, load_checkpoint
from ..utils.config import Config
from ..data import GraphDataset
from ..models import create_model
from ..eval import ClassificationMetrics, ExplainabilityMetrics


class Trainer:
    """Trainer class for GNN models."""
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        dataset: GraphDataset,
        device: torch.device,
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
            model: GNN model.
            dataset: Graph dataset.
            device: Device to run on.
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # Get data
        self.data = dataset.get_data().to(device)
        self.train_mask, self.val_mask, self.test_mask = dataset.get_splits()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("training.lr", 0.01),
            weight_decay=config.get("training.weight_decay", 5e-4),
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.classification_metrics = ClassificationMetrics(
            dataset.num_classes, task="multiclass"
        )
        self.explainability_metrics = ExplainabilityMetrics(device)
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.patience_counter = 0
        
    def _setup_logging(self) -> None:
        """Setup logging (TensorBoard and WandB)."""
        # TensorBoard
        if self.config.get("logging.use_tensorboard", True):
            log_dir = self.config.get("logging.log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # WandB
        if self.config.get("logging.use_wandb", False):
            wandb.init(
                project=self.config.get("experiment.name", "explainable_gnn"),
                config=self.config.to_dict(),
            )
        else:
            wandb.init(mode="disabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.train_mask], self.data.y[self.train_mask])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.get("training.grad_clip_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("training.grad_clip_norm")
            )
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred = out[self.train_mask].argmax(dim=1)
            true = self.data.y[self.train_mask]
            prob = torch.softmax(out[self.train_mask], dim=1)
            
            metrics = self.classification_metrics.compute(pred, true, prob)
            metrics["loss"] = loss.item()
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.val_mask], self.data.y[self.val_mask])
            
            pred = out[self.val_mask].argmax(dim=1)
            true = self.data.y[self.val_mask]
            prob = torch.softmax(out[self.val_mask], dim=1)
            
            metrics = self.classification_metrics.compute(pred, true, prob)
            metrics["loss"] = loss.item()
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model.
        
        Returns:
            Dictionary of test metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.test_mask], self.data.y[self.test_mask])
            
            pred = out[self.test_mask].argmax(dim=1)
            true = self.data.y[self.test_mask]
            prob = torch.softmax(out[self.test_mask], dim=1)
            
            metrics = self.classification_metrics.compute(pred, true, prob)
            metrics["loss"] = loss.item()
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Dictionary containing training results.
        """
        epochs = self.config.get("training.epochs", 200)
        patience = self.config.get("training.patience", 50)
        min_delta = self.config.get("training.min_delta", 1e-4)
        log_interval = self.config.get("logging.log_interval", 10)
        save_interval = self.config.get("logging.save_interval", 50)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Dataset: {self.dataset.name}")
        print(f"Device: {self.device}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Logging
            if epoch % log_interval == 0:
                self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self._save_checkpoint(train_metrics, val_metrics)
            
            # Early stopping
            if val_metrics["loss"] < self.best_val_loss - min_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_metrics = val_metrics.copy()
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(train_metrics, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final test
        test_metrics = self.test()
        
        training_time = time.time() - start_time
        
        print("-" * 80)
        print("Training completed!")
        print(f"Training time: {training_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Test metrics: {test_metrics}")
        
        # Close logging
        if self.writer:
            self.writer.close()
        wandb.finish()
        
        return {
            "best_val_metrics": self.best_val_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "epochs_trained": epoch + 1,
        }
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Log metrics to TensorBoard and WandB.
        
        Args:
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            epoch: Current epoch.
        """
        # TensorBoard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
        
        # WandB
        log_dict = {}
        for key, value in train_metrics.items():
            log_dict[f"train/{key}"] = value
        for key, value in val_metrics.items():
            log_dict[f"val/{key}"] = value
        log_dict["epoch"] = epoch
        
        wandb.log(log_dict)
        
        # Console output
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
    
    def _save_checkpoint(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            is_best: Whether this is the best model.
        """
        checkpoint_dir = self.config.get("paths.checkpoints", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            filepath = os.path.join(checkpoint_dir, "best_model.pth")
        else:
            filepath = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pth")
        
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            val_metrics["loss"],
            val_metrics,
            filepath,
            train_metrics=train_metrics,
            config=self.config.to_dict(),
        )
    
    def load_best_model(self) -> None:
        """Load the best model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.get("paths.checkpoints", "checkpoints"),
            "best_model.pth"
        )
        
        if os.path.exists(checkpoint_path):
            load_checkpoint(self.model, self.optimizer, checkpoint_path, self.device)
            print(f"Loaded best model from {checkpoint_path}")
        else:
            print("No best model checkpoint found")


def train_model(
    config_path: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Train a GNN model with the given configuration.
    
    Args:
        config_path: Path to configuration file.
        **overrides: Configuration overrides.
        
    Returns:
        Training results.
    """
    # Load configuration
    config = Config(config_path)
    config.update(overrides)
    
    # Set seed for reproducibility
    set_seed(config.get("experiment.seed", 42))
    
    # Get device
    device = get_device()
    
    # Load dataset
    dataset = GraphDataset(
        name=config.get("data.dataset", "cora"),
        root=config.get("data.data_dir", "data/raw"),
        transform=config.get("data.transform", ["normalize_features", "add_self_loops"]),
    )
    
    # Create model
    model = create_model(
        model_type=config.get("model.type", "gcn"),
        in_channels=dataset.num_features,
        hidden_channels=config.get("model.hidden_dim", 64),
        out_channels=dataset.num_classes,
        num_layers=config.get("model.num_layers", 2),
        dropout=config.get("model.dropout", 0.5),
        activation=config.get("model.activation", "relu"),
        use_batch_norm=config.get("model.use_batch_norm", True),
        use_residual=config.get("model.use_residual", False),
    ).to(device)
    
    # Print model summary
    from ..utils.device import print_model_summary
    print_model_summary(model, (dataset.num_features,))
    
    # Create trainer
    trainer = Trainer(config, model, dataset, device)
    
    # Train model
    results = trainer.train()
    
    return results
