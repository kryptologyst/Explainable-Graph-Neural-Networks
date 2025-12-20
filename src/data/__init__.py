"""Data loading and preprocessing utilities for graph datasets."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import (
    Planetoid,
    CoraFull,
    CiteSeer,
    PubMed,
    Coauthor,
    Amazon,
    WikiCS,
    Actor,
    Twitch,
)
from torch_geometric.transforms import (
    NormalizeFeatures,
    AddSelfLoops,
    ToUndirected,
    RandomNodeSplit,
)
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx


class GraphDataset:
    """Wrapper for graph datasets with standardized interface."""
    
    def __init__(
        self,
        name: str,
        root: str = "data/raw",
        transform: Optional[List] = None,
        pre_transform: Optional[List] = None,
    ):
        """Initialize dataset.
        
        Args:
            name: Dataset name.
            root: Root directory for data.
            transform: Transforms to apply.
            pre_transform: Pre-transforms to apply.
        """
        self.name = name.lower()
        self.root = root
        self.transform = transform or []
        self.pre_transform = pre_transform or []
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the specified dataset."""
        dataset_map = {
            "cora": Planetoid,
            "citeseer": CiteSeer,
            "pubmed": PubMed,
            "corafull": CoraFull,
            "coauthor_cs": lambda root, **kwargs: Coauthor(root, name="CS", **kwargs),
            "coauthor_physics": lambda root, **kwargs: Coauthor(root, name="Physics", **kwargs),
            "amazon_computers": lambda root, **kwargs: Amazon(root, name="Computers", **kwargs),
            "amazon_photo": lambda root, **kwargs: Amazon(root, name="Photo", **kwargs),
            "wikics": WikiCS,
            "actor": Actor,
            "twitch_pt": lambda root, **kwargs: Twitch(root, name="PT", **kwargs),
            "twitch_de": lambda root, **kwargs: Twitch(root, name="DE", **kwargs),
            "twitch_en": lambda root, **kwargs: Twitch(root, name="EN", **kwargs),
            "twitch_ru": lambda root, **kwargs: Twitch(root, name="RU", **kwargs),
        }
        
        if self.name not in dataset_map:
            raise ValueError(f"Dataset '{self.name}' not supported. Available: {list(dataset_map.keys())}")
        
        dataset_class = dataset_map[self.name]
        
        # Apply transforms
        transforms = []
        if "normalize_features" in self.transform:
            transforms.append(NormalizeFeatures())
        if "add_self_loops" in self.transform:
            transforms.append(AddSelfLoops())
        if "to_undirected" in self.transform:
            transforms.append(ToUndirected())
        if "random_split" in self.transform:
            transforms.append(RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2))
        
        transform = transforms if transforms else None
        
        self.dataset = dataset_class(
            root=self.root,
            transform=transform,
            pre_transform=self.pre_transform,
        )
        
        self.data = self.dataset[0]
        self.num_nodes = self.data.num_nodes
        self.num_edges = self.data.num_edges
        self.num_features = self.data.num_node_features
        self.num_classes = self.dataset.num_classes
    
    def get_data(self) -> Data:
        """Get the graph data.
        
        Returns:
            PyTorch Geometric Data object.
        """
        return self.data
    
    def get_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get train/validation/test splits.
        
        Returns:
            Tuple of (train_mask, val_mask, test_mask).
        """
        if hasattr(self.data, "train_mask"):
            return self.data.train_mask, self.data.val_mask, self.data.test_mask
        else:
            # Create random splits if not available
            return self._create_random_splits()
    
    def _create_random_splits(
        self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random train/val/test splits.
        
        Args:
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        num_nodes = self.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get dataset statistics.
        
        Returns:
            Dictionary of dataset statistics.
        """
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "density": self.num_edges / (self.num_nodes * (self.num_nodes - 1)),
            "avg_degree": 2 * self.num_edges / self.num_nodes,
        }
        
        # Add class distribution
        if hasattr(self.data, "y"):
            class_counts = torch.bincount(self.data.y)
            stats["class_distribution"] = class_counts.tolist()
            stats["class_balance"] = (class_counts.min() / class_counts.max()).item()
        
        return stats
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph.
        
        Returns:
            NetworkX graph.
        """
        return to_networkx(self.data, to_undirected=True)
    
    def visualize_graph(
        self,
        max_nodes: int = 1000,
        node_color: Optional[str] = None,
        edge_color: str = "gray",
        node_size: int = 50,
        width: float = 0.5,
    ) -> None:
        """Visualize the graph using NetworkX and matplotlib.
        
        Args:
            max_nodes: Maximum number of nodes to visualize.
            node_color: Node color (can be 'label' for class-based coloring).
            edge_color: Edge color.
            node_size: Node size.
            width: Edge width.
        """
        import matplotlib.pyplot as plt
        
        G = self.to_networkx()
        
        # Sample nodes if graph is too large
        if len(G.nodes()) > max_nodes:
            nodes_to_keep = np.random.choice(list(G.nodes()), max_nodes, replace=False)
            G = G.subgraph(nodes_to_keep)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        if node_color == "label" and hasattr(self.data, "y"):
            node_colors = [self.data.y[node].item() for node in G.nodes()]
            nx.draw(
                G,
                pos,
                node_color=node_colors,
                cmap=plt.cm.Set3,
                node_size=node_size,
                edge_color=edge_color,
                width=width,
                with_labels=False,
            )
        else:
            nx.draw(
                G,
                pos,
                node_color="lightblue",
                node_size=node_size,
                edge_color=edge_color,
                width=width,
                with_labels=False,
            )
        
        plt.title(f"Graph Visualization: {self.name.upper()}")
        plt.show()


def create_synthetic_dataset(
    num_nodes: int = 1000,
    num_classes: int = 3,
    num_features: int = 50,
    edge_prob: float = 0.1,
    seed: int = 42,
) -> Data:
    """Create a synthetic graph dataset for testing.
    
    Args:
        num_nodes: Number of nodes.
        num_classes: Number of classes.
        num_features: Number of features per node.
        edge_prob: Probability of edge existence.
        seed: Random seed.
        
    Returns:
        PyTorch Geometric Data object.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create random features
    x = torch.randn(num_nodes, num_features)
    
    # Create random labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, int(num_nodes * num_nodes * edge_prob)))
    
    # Remove self-loops and duplicates
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
