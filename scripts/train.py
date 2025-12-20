#!/usr/bin/env python3
"""Training script for explainable GNN models."""

import argparse
import os
import sys
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.train import train_model
from src.utils.config import Config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train explainable GNN models")
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["cora", "citeseer", "pubmed", "corafull", "coauthor_cs", "coauthor_physics", 
                "amazon_computers", "amazon_photo", "wikics", "actor", "twitch_pt", "twitch_de", 
                "twitch_en", "twitch_ru"],
        help="Dataset to use"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["gcn", "gat", "gin", "gcn2"],
        help="Model type"
    )
    
    # Training options
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden dimension"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate"
    )
    
    # Experiment options
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name"
    )
    
    # Logging options
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="Use TensorBoard logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override configuration with command line arguments
    overrides = {}
    
    if args.dataset:
        overrides["data.dataset"] = args.dataset
    if args.model:
        overrides["model.type"] = args.model
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.hidden_dim:
        overrides["model.hidden_dim"] = args.hidden_dim
    if args.num_layers:
        overrides["model.num_layers"] = args.num_layers
    if args.dropout:
        overrides["model.dropout"] = args.dropout
    if args.seed:
        overrides["experiment.seed"] = args.seed
    if args.name:
        overrides["experiment.name"] = args.name
    if args.use_wandb:
        overrides["logging.use_wandb"] = True
    if args.use_tensorboard:
        overrides["logging.use_tensorboard"] = True
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Dataset: {overrides.get('data.dataset', config.get('data.dataset'))}")
    print(f"Model: {overrides.get('model.type', config.get('model.type'))}")
    print(f"Epochs: {overrides.get('training.epochs', config.get('training.epochs'))}")
    print(f"Learning rate: {overrides.get('training.lr', config.get('training.lr'))}")
    print(f"Hidden dim: {overrides.get('model.hidden_dim', config.get('model.hidden_dim'))}")
    print(f"Number of layers: {overrides.get('model.num_layers', config.get('model.num_layers'))}")
    print(f"Dropout: {overrides.get('model.dropout', config.get('model.dropout'))}")
    print(f"Seed: {overrides.get('experiment.seed', config.get('experiment.seed'))}")
    print(f"Experiment name: {overrides.get('experiment.name', config.get('experiment.name'))}")
    print("=" * 80)
    
    # Train model
    try:
        results = train_model(config_path=args.config, **overrides)
        
        print("\n" + "=" * 80)
        print("Training Results")
        print("=" * 80)
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Epochs trained: {results['epochs_trained']}")
        print(f"Best validation loss: {results['best_val_metrics']['loss']:.4f}")
        print(f"Best validation accuracy: {results['best_val_metrics']['accuracy']:.4f}")
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1 macro: {results['test_metrics']['f1_macro']:.4f}")
        print(f"Test F1 micro: {results['test_metrics']['f1_micro']:.4f}")
        if 'auc' in results['test_metrics']:
            print(f"Test AUC: {results['test_metrics']['auc']:.4f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
