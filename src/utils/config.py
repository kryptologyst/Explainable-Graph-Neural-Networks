"""Configuration management using OmegaConf."""

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the explainable GNN project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self._config: Optional[DictConfig] = None
        self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        default_config = {
            "experiment": {
                "name": "explainable_gnn",
                "seed": 42,
                "device": "auto",  # auto, cuda, mps, cpu
            },
            "data": {
                "dataset": "cora",
                "data_dir": "data/raw",
                "train_split": 0.6,
                "val_split": 0.2,
                "test_split": 0.2,
                "normalize_features": True,
                "add_self_loops": True,
            },
            "model": {
                "type": "gcn",  # gcn, gat, gin, gcn2
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.5,
                "activation": "relu",
                "use_batch_norm": True,
                "use_residual": False,
            },
            "gat": {
                "num_heads": 8,
                "concat": True,
                "negative_slope": 0.2,
            },
            "gin": {
                "eps": 0.0,
                "train_eps": True,
            },
            "training": {
                "epochs": 200,
                "lr": 0.01,
                "weight_decay": 5e-4,
                "patience": 50,
                "min_delta": 1e-4,
                "batch_size": 1,  # For full-batch training
                "grad_clip_norm": None,
            },
            "explainability": {
                "methods": ["gnn_explainer", "pg_explainer", "integrated_gradients"],
                "gnn_explainer": {
                    "epochs": 200,
                    "lr": 0.01,
                    "return_type": "log_prob",
                    "feat_mask_type": "feature",
                    "allow_edge_mask": True,
                    "allow_node_mask": False,
                },
                "pg_explainer": {
                    "epochs": 100,
                    "lr": 0.003,
                    "coff_size": 0.01,
                    "coff_ent": 0.01,
                    "t0": 5.0,
                    "t1": 1.0,
                },
                "integrated_gradients": {
                    "steps": 50,
                    "multiply_by_inputs": True,
                },
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_macro", "f1_micro", "auc"],
                "explainability_metrics": ["fidelity", "sparsity", "stability"],
                "num_explanations": 100,
                "explanation_nodes": "test",  # train, val, test, all
            },
            "logging": {
                "log_dir": "logs",
                "use_wandb": False,
                "use_tensorboard": True,
                "log_interval": 10,
                "save_interval": 50,
            },
            "paths": {
                "checkpoints": "checkpoints",
                "assets": "assets",
                "plots": "assets/plots",
                "explanations": "assets/explanations",
            },
        }
        
        self._config = OmegaConf.create(default_config)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file.
        """
        file_config = OmegaConf.load(config_path)
        self._config = OmegaConf.merge(self._config, file_config)
        self.config_path = config_path
    
    def save_config(self, save_path: str) -> None:
        """Save current configuration to file.
        
        Args:
            save_path: Path to save configuration.
        """
        OmegaConf.save(self._config, save_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            updates: Dictionary of updates.
        """
        self._config = OmegaConf.merge(self._config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying DictConfig object.
        
        Returns:
            DictConfig object.
        """
        return self._config
