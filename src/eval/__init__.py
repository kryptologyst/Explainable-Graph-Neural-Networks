"""Evaluation metrics for explainable GNN models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from torchmetrics import (
    Accuracy,
    F1Score,
    AUROC,
    Precision,
    Recall,
)
from torch_geometric.utils import to_networkx
import networkx as nx


class ClassificationMetrics:
    """Classification metrics for node classification tasks."""
    
    def __init__(self, num_classes: int, task: str = "multiclass"):
        """Initialize classification metrics.
        
        Args:
            num_classes: Number of classes.
            task: Task type ('multiclass', 'multilabel', 'binary').
        """
        self.num_classes = num_classes
        self.task = task
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        self.f1_macro = F1Score(task=task, num_classes=num_classes, average="macro")
        self.f1_micro = F1Score(task=task, num_classes=num_classes, average="micro")
        self.f1_weighted = F1Score(task=task, num_classes=num_classes, average="weighted")
        
        if task == "binary":
            self.auroc = AUROC(task="binary")
        else:
            self.auroc = AUROC(task="multiclass", num_classes=num_classes)
    
    def compute(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute all classification metrics.
        
        Args:
            y_pred: Predicted labels.
            y_true: True labels.
            y_prob: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
        metrics["f1_macro"] = f1_score(y_true_np, y_pred_np, average="macro")
        metrics["f1_micro"] = f1_score(y_true_np, y_pred_np, average="micro")
        metrics["f1_weighted"] = f1_score(y_true_np, y_pred_np, average="weighted")
        
        # AUC metrics
        if y_prob is not None:
            y_prob_np = y_prob.cpu().numpy()
            
            if self.task == "binary":
                metrics["auc"] = roc_auc_score(y_true_np, y_prob_np)
            else:
                metrics["auc"] = roc_auc_score(
                    y_true_np, y_prob_np, multi_class="ovr", average="macro"
                )
        
        return metrics


class ExplainabilityMetrics:
    """Metrics for evaluating explainability methods."""
    
    def __init__(self, device: torch.device):
        """Initialize explainability metrics.
        
        Args:
            device: Device to run on.
        """
        self.device = device
    
    def fidelity(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y_true: torch.Tensor,
        edge_mask: torch.Tensor,
        node_idx: Optional[int] = None,
        threshold: float = 0.5,
    ) -> float:
        """Compute fidelity metric.
        
        Fidelity measures how well the explanation subgraph preserves the original prediction.
        
        Args:
            model: Trained GNN model.
            x: Node features.
            edge_index: Edge indices.
            y_true: True labels.
            edge_mask: Edge importance mask.
            node_idx: Node index for node-level explanation.
            threshold: Threshold for edge selection.
            
        Returns:
            Fidelity score.
        """
        model.eval()
        
        with torch.no_grad():
            # Original prediction
            if node_idx is not None:
                orig_pred = model(x, edge_index)[node_idx]
                orig_pred_class = orig_pred.argmax().item()
            else:
                orig_pred = model(x, edge_index)
                orig_pred_class = orig_pred.argmax(dim=1)
            
            # Select important edges
            important_edges = edge_mask > threshold
            if important_edges.sum() == 0:
                return 0.0
            
            filtered_edge_index = edge_index[:, important_edges]
            
            # Prediction with filtered graph
            if node_idx is not None:
                filtered_pred = model(x, filtered_edge_index)[node_idx]
                filtered_pred_class = filtered_pred.argmax().item()
            else:
                filtered_pred = model(x, filtered_edge_index)
                filtered_pred_class = filtered_pred.argmax(dim=1)
            
            # Compute fidelity
            if node_idx is not None:
                fidelity = 1.0 if orig_pred_class == filtered_pred_class else 0.0
            else:
                fidelity = (orig_pred_class == filtered_pred_class).float().mean().item()
        
        return fidelity
    
    def sparsity(
        self,
        edge_mask: torch.Tensor,
        node_feat_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute sparsity metric.
        
        Sparsity measures how concise the explanation is.
        
        Args:
            edge_mask: Edge importance mask.
            node_feat_mask: Node feature importance mask (optional).
            threshold: Threshold for feature selection.
            
        Returns:
            Dictionary containing sparsity scores.
        """
        metrics = {}
        
        # Edge sparsity
        edge_sparsity = (edge_mask > threshold).float().mean().item()
        metrics["edge_sparsity"] = edge_sparsity
        
        # Node feature sparsity
        if node_feat_mask is not None:
            if node_feat_mask.dim() == 1:
                # Single node
                feat_sparsity = (node_feat_mask > threshold).float().mean().item()
            else:
                # Multiple nodes
                feat_sparsity = (node_feat_mask > threshold).float().mean().item()
            metrics["feature_sparsity"] = feat_sparsity
        
        return metrics
    
    def stability(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        explainer,
        node_idx: int,
        num_runs: int = 5,
        noise_std: float = 0.01,
    ) -> float:
        """Compute stability metric.
        
        Stability measures how consistent explanations are across multiple runs.
        
        Args:
            model: Trained GNN model.
            x: Node features.
            edge_index: Edge indices.
            explainer: Explanation method.
            node_idx: Node index to explain.
            num_runs: Number of runs for stability test.
            noise_std: Standard deviation of noise to add.
            
        Returns:
            Stability score.
        """
        explanations = []
        
        for _ in range(num_runs):
            # Add small noise to features
            noisy_x = x + torch.randn_like(x) * noise_std
            
            # Get explanation
            explanation = explainer.explain_node(node_idx, noisy_x, edge_index)
            explanations.append(explanation["edge_mask"])
        
        # Compute pairwise similarities
        similarities = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                sim = torch.cosine_similarity(
                    explanations[i].unsqueeze(0),
                    explanations[j].unsqueeze(0)
                ).item()
                similarities.append(sim)
        
        # Stability is the average similarity
        stability = np.mean(similarities) if similarities else 0.0
        
        return stability
    
    def contrastivity(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        node_idx: int,
        threshold: float = 0.5,
    ) -> float:
        """Compute contrastivity metric.
        
        Contrastivity measures how much the explanation changes when the prediction changes.
        
        Args:
            model: Trained GNN model.
            x: Node features.
            edge_index: Edge indices.
            edge_mask: Edge importance mask.
            node_idx: Node index to explain.
            threshold: Threshold for edge selection.
            
        Returns:
            Contrastivity score.
        """
        model.eval()
        
        with torch.no_grad():
            # Original prediction
            orig_pred = model(x, edge_index)[node_idx]
            orig_pred_class = orig_pred.argmax().item()
            
            # Get important edges
            important_edges = edge_mask > threshold
            
            # Create modified graph by removing important edges
            if important_edges.sum() > 0:
                modified_edge_index = edge_index[:, ~important_edges]
                
                if modified_edge_index.size(1) > 0:
                    modified_pred = model(x, modified_edge_index)[node_idx]
                    modified_pred_class = modified_pred.argmax().item()
                    
                    # Contrastivity is 1 if prediction changes, 0 otherwise
                    contrastivity = 1.0 if orig_pred_class != modified_pred_class else 0.0
                else:
                    contrastivity = 0.0
            else:
                contrastivity = 0.0
        
        return contrastivity
    
    def compute_all(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y_true: torch.Tensor,
        explanations: List[Dict[str, torch.Tensor]],
        explainer,
        node_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compute all explainability metrics.
        
        Args:
            model: Trained GNN model.
            x: Node features.
            edge_index: Edge indices.
            y_true: True labels.
            explanations: List of explanations.
            explainer: Explanation method.
            node_indices: List of node indices (optional).
            
        Returns:
            Dictionary of all metrics.
        """
        metrics = {}
        
        if not explanations:
            return metrics
        
        # Compute metrics for each explanation
        fidelities = []
        sparsities = []
        stabilities = []
        contrastivities = []
        
        for i, explanation in enumerate(explanations):
            node_idx = node_indices[i] if node_indices else i
            
            # Fidelity
            fidelity = self.fidelity(
                model, x, edge_index, y_true,
                explanation["edge_mask"], node_idx
            )
            fidelities.append(fidelity)
            
            # Sparsity
            sparsity = self.sparsity(
                explanation["edge_mask"],
                explanation.get("node_feat_mask")
            )
            sparsities.append(sparsity)
            
            # Stability
            stability = self.stability(
                model, x, edge_index, explainer, node_idx
            )
            stabilities.append(stability)
            
            # Contrastivity
            contrastivity = self.contrastivity(
                model, x, edge_index, explanation["edge_mask"], node_idx
            )
            contrastivities.append(contrastivity)
        
        # Average metrics
        metrics["fidelity"] = np.mean(fidelities)
        metrics["fidelity_std"] = np.std(fidelities)
        metrics["sparsity"] = np.mean([s["edge_sparsity"] for s in sparsities])
        metrics["sparsity_std"] = np.std([s["edge_sparsity"] for s in sparsities])
        metrics["stability"] = np.mean(stabilities)
        metrics["stability_std"] = np.std(stabilities)
        metrics["contrastivity"] = np.mean(contrastivities)
        metrics["contrastivity_std"] = np.std(contrastivities)
        
        return metrics


class GraphMetrics:
    """Graph-level metrics for evaluating explanations."""
    
    def __init__(self):
        """Initialize graph metrics."""
        pass
    
    def modularity(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Compute modularity of explanation subgraph.
        
        Args:
            edge_index: Edge indices.
            edge_mask: Edge importance mask.
            threshold: Threshold for edge selection.
            
        Returns:
            Modularity score.
        """
        # Select important edges
        important_edges = edge_mask > threshold
        
        if important_edges.sum() == 0:
            return 0.0
        
        # Create subgraph
        subgraph_edge_index = edge_index[:, important_edges]
        
        # Convert to NetworkX for modularity computation
        G = nx.Graph()
        G.add_edges_from(subgraph_edge_index.t().cpu().numpy())
        
        # Compute modularity
        try:
            modularity = nx.algorithms.community.modularity(
                G, nx.algorithms.community.greedy_modularity_communities(G)
            )
        except:
            modularity = 0.0
        
        return modularity
    
    def clustering_coefficient(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Compute clustering coefficient of explanation subgraph.
        
        Args:
            edge_index: Edge indices.
            edge_mask: Edge importance mask.
            threshold: Threshold for edge selection.
            
        Returns:
            Clustering coefficient.
        """
        # Select important edges
        important_edges = edge_mask > threshold
        
        if important_edges.sum() == 0:
            return 0.0
        
        # Create subgraph
        subgraph_edge_index = edge_index[:, important_edges]
        
        # Convert to NetworkX
        G = nx.Graph()
        G.add_edges_from(subgraph_edge_index.t().cpu().numpy())
        
        # Compute clustering coefficient
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0.0
        
        return clustering_coeff
