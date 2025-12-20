#!/usr/bin/env python3
"""Example script demonstrating the explainable GNN toolkit."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.device import get_device, set_seed
from src.utils.config import Config
from src.data import GraphDataset, create_synthetic_dataset
from src.models import create_model
from src.train import train_model
from src.explain import create_explainer
from src.eval import ClassificationMetrics, ExplainabilityMetrics


def main():
    """Main example function."""
    print("=" * 80)
    print("Explainable GNN Models - Example Script")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic dataset for demonstration
    print("\nCreating synthetic dataset...")
    data = create_synthetic_dataset(
        num_nodes=1000,
        num_classes=3,
        num_features=50,
        edge_prob=0.1,
        seed=42
    )
    
    print(f"Dataset created:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Classes: {data.num_classes}")
    
    # Create model
    print("\nCreating GCN model...")
    model = create_model(
        model_type="gcn",
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=data.num_classes,
        num_layers=2,
        dropout=0.5,
        use_batch_norm=True
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Move data to device
    data = data.to(device)
    
    # Quick training (just a few epochs for demo)
    print("\nTraining model (quick demo)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1)
        true = data.y[data.test_mask]
        prob = torch.softmax(out[data.test_mask], dim=1)
        
        # Compute metrics
        classification_metrics = ClassificationMetrics(data.num_classes, task="multiclass")
        metrics = classification_metrics.compute(pred, true, prob)
        
        print("Test set performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Generate explanations
    print("\nGenerating explanations...")
    
    # Select a few test nodes to explain
    test_nodes = torch.where(data.test_mask)[0][:5].tolist()
    print(f"Explaining nodes: {test_nodes}")
    
    # Create explainer
    explainer = create_explainer(
        method="gnn_explainer",
        model=model,
        device=device,
        epochs=50  # Quick demo
    )
    
    explanations = []
    for node_idx in test_nodes:
        explanation = explainer.explain_node(node_idx, data.x, data.edge_index)
        explanations.append(explanation)
        print(f"  Node {node_idx}: Edge mask shape {explanation['edge_mask'].shape}")
    
    # Evaluate explanations
    print("\nEvaluating explanations...")
    explainability_metrics = ExplainabilityMetrics(device)
    
    metrics = explainability_metrics.compute_all(
        model, data.x, data.edge_index, data.y,
        explanations, explainer, test_nodes
    )
    
    print("Explanation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize one explanation
    print("\nVisualizing explanation for node", test_nodes[0])
    
    node_idx = test_nodes[0]
    explanation = explanations[0]
    edge_mask = explanation["edge_mask"]
    
    # Get edges connected to the node
    node_edges = []
    node_edge_importance = []
    
    for i, (src, tgt) in enumerate(data.edge_index.t().cpu().numpy()):
        if src == node_idx or tgt == node_idx:
            node_edges.append(f"{src}-{tgt}")
            node_edge_importance.append(edge_mask[i].item())
    
    if node_edges:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(node_edges)), node_edge_importance)
        plt.xlabel("Edges")
        plt.ylabel("Importance")
        plt.title(f"Edge Importance for Node {node_idx}")
        plt.xticks(range(len(node_edges)), node_edges, rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("assets/plots", exist_ok=True)
        plt.savefig("assets/plots/example_explanation.png", dpi=150, bbox_inches="tight")
        print("  Saved explanation plot to assets/plots/example_explanation.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run the full training: python scripts/train.py --dataset cora --model gcn")
    print("2. Generate explanations: python -m src.cli explain --config configs/default.yaml")
    print("3. Launch interactive demo: python -m src.cli demo")
    print("4. Explore the code in src/ directory")


if __name__ == "__main__":
    main()
