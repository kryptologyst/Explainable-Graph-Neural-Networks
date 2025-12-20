"""Command-line interface for explainable GNN models."""

import argparse
import os
import sys
from typing import Optional

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.device import get_device, set_seed
from src.utils.config import Config
from src.data import GraphDataset
from src.models import create_model
from src.train import train_model
from src.explain import create_explainer
from src.eval import ClassificationMetrics, ExplainabilityMetrics


def train_command(args) -> None:
    """Train a GNN model."""
    print("Training GNN model...")
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    if args.dataset:
        config.set("data.dataset", args.dataset)
    if args.model:
        config.set("model.type", args.model)
    if args.epochs:
        config.set("training.epochs", args.epochs)
    if args.lr:
        config.set("training.lr", args.lr)
    
    # Train model
    results = train_model(config_path=args.config)
    
    print("Training completed!")
    print(f"Best validation metrics: {results['best_val_metrics']}")
    print(f"Test metrics: {results['test_metrics']}")


def explain_command(args) -> None:
    """Generate explanations for a trained model."""
    print("Generating explanations...")
    
    # Load configuration
    config = Config(args.config)
    
    # Set seed
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
    
    # Load best model
    checkpoint_path = os.path.join(
        config.get("paths.checkpoints", "checkpoints"),
        "best_model.pth"
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No trained model found. Please train a model first.")
        return
    
    # Get data
    data = dataset.get_data().to(device)
    train_mask, val_mask, test_mask = dataset.get_splits()
    
    # Select nodes to explain
    if args.nodes == "test":
        node_indices = torch.where(test_mask)[0].tolist()
    elif args.nodes == "val":
        node_indices = torch.where(val_mask)[0].tolist()
    elif args.nodes == "train":
        node_indices = torch.where(train_mask)[0].tolist()
    else:
        node_indices = [int(args.nodes)]
    
    # Limit number of explanations
    if args.num_explanations:
        node_indices = node_indices[:args.num_explanations]
    
    print(f"Explaining {len(node_indices)} nodes...")
    
    # Generate explanations
    methods = config.get("explainability.methods", ["gnn_explainer"])
    all_explanations = {}
    
    for method in methods:
        print(f"Generating explanations using {method}...")
        
        # Create explainer
        explainer_config = config.get(f"explainability.{method}", {})
        explainer = create_explainer(method, model, device, **explainer_config)
        
        # Generate explanations
        explanations = []
        for node_idx in node_indices:
            explanation = explainer.explain_node(node_idx, data.x, data.edge_index)
            explanations.append(explanation)
        
        all_explanations[method] = explanations
        
        # Save explanations
        explanations_dir = config.get("paths.explanations", "assets/explanations")
        os.makedirs(explanations_dir, exist_ok=True)
        
        explanations_path = os.path.join(explanations_dir, f"{method}_explanations.pt")
        torch.save({
            "method": method,
            "node_indices": node_indices,
            "explanations": explanations,
            "config": explainer_config,
        }, explanations_path)
        
        print(f"Saved explanations to {explanations_path}")
    
    # Evaluate explanations
    print("Evaluating explanations...")
    
    explainability_metrics = ExplainabilityMetrics(device)
    
    for method, explanations in all_explanations.items():
        print(f"Evaluating {method} explanations...")
        
        # Create explainer for evaluation
        explainer_config = config.get(f"explainability.{method}", {})
        explainer = create_explainer(method, model, device, **explainer_config)
        
        # Compute metrics
        metrics = explainability_metrics.compute_all(
            model, data.x, data.edge_index, data.y,
            explanations, explainer, node_indices
        )
        
        print(f"{method} metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("Explanation generation completed!")


def evaluate_command(args) -> None:
    """Evaluate a trained model."""
    print("Evaluating model...")
    
    # Load configuration
    config = Config(args.config)
    
    # Set seed
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
    
    # Load best model
    checkpoint_path = os.path.join(
        config.get("paths.checkpoints", "checkpoints"),
        "best_model.pth"
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No trained model found. Please train a model first.")
        return
    
    # Get data
    data = dataset.get_data().to(device)
    train_mask, val_mask, test_mask = dataset.get_splits()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        pred = out[test_mask].argmax(dim=1)
        true = data.y[test_mask]
        prob = torch.softmax(out[test_mask], dim=1)
        
        # Compute metrics
        classification_metrics = ClassificationMetrics(dataset.num_classes, task="multiclass")
        metrics = classification_metrics.compute(pred, true, prob)
    
    print("Test set evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def demo_command(args) -> None:
    """Launch the interactive demo."""
    print("Launching interactive demo...")
    
    import subprocess
    import sys
    
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "app.py")
    
    if os.path.exists(demo_path):
        subprocess.run([sys.executable, "-m", "streamlit", "run", demo_path])
    else:
        print("Demo not found. Please ensure demo/app.py exists.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Explainable GNN Models")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a GNN model")
    train_parser.add_argument("--config", "-c", default="configs/default.yaml", help="Configuration file")
    train_parser.add_argument("--dataset", "-d", help="Dataset name")
    train_parser.add_argument("--model", "-m", help="Model type")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Generate explanations")
    explain_parser.add_argument("--config", "-c", default="configs/default.yaml", help="Configuration file")
    explain_parser.add_argument("--nodes", default="test", help="Nodes to explain (train/val/test)")
    explain_parser.add_argument("--num-explanations", type=int, help="Number of explanations to generate")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", "-c", default="configs/default.yaml", help="Configuration file")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Launch interactive demo")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "explain":
        explain_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "demo":
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
