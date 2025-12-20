"""Explainability methods for Graph Neural Networks."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from captum.attr import IntegratedGradients


class BaseExplainer:
    """Base class for explainability methods."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize explainer.
        
        Args:
            model: Trained GNN model.
            device: Device to run on.
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for a single node.
        
        Args:
            node_idx: Index of node to explain.
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        raise NotImplementedError
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for entire graph.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        raise NotImplementedError


class GNNExplainerWrapper(BaseExplainer):
    """Wrapper for PyTorch Geometric's GNNExplainer."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epochs: int = 200,
        lr: float = 0.01,
        return_type: str = "log_prob",
        feat_mask_type: str = "feature",
        allow_edge_mask: bool = True,
        allow_node_mask: bool = False,
    ):
        """Initialize GNNExplainer.
        
        Args:
            model: Trained GNN model.
            device: Device to run on.
            epochs: Number of training epochs.
            lr: Learning rate.
            return_type: Return type ('log_prob', 'prob', 'reg').
            feat_mask_type: Feature mask type ('feature', 'individual_feature').
            allow_edge_mask: Whether to allow edge masking.
            allow_node_mask: Whether to allow node masking.
        """
        super().__init__(model, device)
        
        self.explainer = GNNExplainer(
            model,
            epochs=epochs,
            lr=lr,
            return_type=return_type,
            feat_mask_type=feat_mask_type,
            allow_edge_mask=allow_edge_mask,
            allow_node_mask=allow_node_mask,
        )
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for a single node.
        
        Args:
            node_idx: Index of node to explain.
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        node_feat_mask, edge_mask = self.explainer.explain_node(
            node_idx, x, edge_index, **kwargs
        )
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for entire graph.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        node_feat_mask, edge_mask = self.explainer.explain_graph(
            x, edge_index, **kwargs
        )
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }


class PGExplainer(BaseExplainer):
    """Parameterized Graph Explainer."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epochs: int = 100,
        lr: float = 0.003,
        coff_size: float = 0.01,
        coff_ent: float = 0.01,
        t0: float = 5.0,
        t1: float = 1.0,
    ):
        """Initialize PGExplainer.
        
        Args:
            model: Trained GNN model.
            device: Device to run on.
            epochs: Number of training epochs.
            lr: Learning rate.
            coff_size: Size coefficient.
            coff_ent: Entropy coefficient.
            t0: Initial temperature.
            t1: Final temperature.
        """
        super().__init__(model, device)
        
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        
        # Initialize explainer network
        self.explainer_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.explainer_net.parameters(), lr=lr)
    
    def _get_edge_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get edge embeddings for explanation.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Edge embeddings.
        """
        # Get node embeddings from the model
        with torch.no_grad():
            node_embeddings = self.model.get_embeddings(x, edge_index)
        
        # Create edge embeddings by concatenating source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        tgt_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)
        
        return edge_embeddings
    
    def _sample_edge_mask(
        self, edge_embeddings: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Sample edge mask using Gumbel-Softmax.
        
        Args:
            edge_embeddings: Edge embeddings.
            temperature: Temperature for Gumbel-Softmax.
            
        Returns:
            Edge mask.
        """
        logits = self.explainer_net(edge_embeddings).squeeze()
        
        # Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        logits_with_noise = (logits + gumbel_noise) / temperature
        
        edge_mask = torch.sigmoid(logits_with_noise)
        
        return edge_mask
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for a single node.
        
        Args:
            node_idx: Index of node to explain.
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        edge_embeddings = self._get_edge_embeddings(x, edge_index)
        
        # Sample edge mask
        edge_mask = self._sample_edge_mask(edge_embeddings, self.t1)
        
        return {
            "edge_mask": edge_mask,
            "node_feat_mask": None,  # PGExplainer doesn't provide node feature masks
        }
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for entire graph.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        return self.explain_node(0, x, edge_index, **kwargs)


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer for GNNs."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        steps: int = 50,
        multiply_by_inputs: bool = True,
    ):
        """Initialize Integrated Gradients explainer.
        
        Args:
            model: Trained GNN model.
            device: Device to run on.
            steps: Number of integration steps.
            multiply_by_inputs: Whether to multiply by inputs.
        """
        super().__init__(model, device)
        
        self.steps = steps
        self.multiply_by_inputs = multiply_by_inputs
        
        self.ig = IntegratedGradients(self._forward_fn)
    
    def _forward_fn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward function for Integrated Gradients.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Model predictions.
        """
        return self.model(x, edge_index)
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for a single node using Integrated Gradients.
        
        Args:
            node_idx: Index of node to explain.
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        # Create baseline (zero features)
        baseline_x = torch.zeros_like(x)
        
        # Compute integrated gradients for node features
        node_feat_attr = self.ig.attribute(
            x,
            baselines=baseline_x,
            target=node_idx,
            additional_forward_args=(edge_index,),
            n_steps=self.steps,
            internal_batch_size=1,
        )
        
        # Normalize to get importance scores
        node_feat_mask = torch.abs(node_feat_attr[node_idx])
        
        # For edge importance, we can approximate by looking at feature gradients
        # This is a simplified approach - more sophisticated methods exist
        edge_mask = torch.ones(edge_index.size(1), device=self.device)
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for entire graph.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        # Create baseline
        baseline_x = torch.zeros_like(x)
        
        # Compute integrated gradients for all nodes
        node_feat_attr = self.ig.attribute(
            x,
            baselines=baseline_x,
            additional_forward_args=(edge_index,),
            n_steps=self.steps,
            internal_batch_size=1,
        )
        
        # Normalize to get importance scores
        node_feat_mask = torch.abs(node_feat_attr)
        
        # Edge mask (simplified)
        edge_mask = torch.ones(edge_index.size(1), device=self.device)
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }


class AttentionExplainer(BaseExplainer):
    """Attention-based explainer for GAT models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize attention explainer.
        
        Args:
            model: Trained GAT model.
            device: Device to run on.
        """
        super().__init__(model, device)
        
        if not hasattr(model, 'convs'):
            raise ValueError("Model must have 'convs' attribute for attention explanation")
        
        # Check if model has attention layers
        self.has_attention = any(
            hasattr(conv, 'att_src') or hasattr(conv, 'alpha') 
            for conv in model.convs
        )
        
        if not self.has_attention:
            raise ValueError("Model must be a GAT model with attention mechanisms")
    
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction using attention weights.
        
        Args:
            node_idx: Index of node to explain.
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        # Forward pass to get attention weights
        self.model.eval()
        with torch.no_grad():
            # Get attention weights from the last layer
            last_conv = self.model.convs[-1]
            
            if hasattr(last_conv, 'alpha'):
                # GATConv with alpha attribute
                _, alpha = last_conv(x, edge_index, return_attention_weights=True)
                edge_mask = alpha.mean(dim=1)  # Average across heads
            else:
                # Fallback: use uniform attention
                edge_mask = torch.ones(edge_index.size(1), device=self.device)
        
        # Node feature mask (simplified - use feature magnitudes)
        node_feat_mask = torch.abs(x[node_idx])
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }
    
    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Explain prediction for entire graph using attention.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing explanation masks.
        """
        # Get attention weights for all edges
        self.model.eval()
        with torch.no_grad():
            last_conv = self.model.convs[-1]
            
            if hasattr(last_conv, 'alpha'):
                _, alpha = last_conv(x, edge_index, return_attention_weights=True)
                edge_mask = alpha.mean(dim=1)
            else:
                edge_mask = torch.ones(edge_index.size(1), device=self.device)
        
        # Node feature mask for all nodes
        node_feat_mask = torch.abs(x)
        
        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
        }


def create_explainer(
    method: str,
    model: nn.Module,
    device: torch.device,
    **kwargs
) -> BaseExplainer:
    """Create an explainer instance.
    
    Args:
        method: Explanation method ('gnn_explainer', 'pg_explainer', 'integrated_gradients', 'attention').
        model: Trained GNN model.
        device: Device to run on.
        **kwargs: Additional explainer parameters.
        
    Returns:
        Explainer instance.
    """
    explainer_map = {
        "gnn_explainer": GNNExplainerWrapper,
        "pg_explainer": PGExplainer,
        "integrated_gradients": IntegratedGradientsExplainer,
        "attention": AttentionExplainer,
    }
    
    if method not in explainer_map:
        raise ValueError(f"Explanation method '{method}' not supported. Available: {list(explainer_map.keys())}")
    
    explainer_class = explainer_map[method]
    return explainer_class(model, device, **kwargs)
