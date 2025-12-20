"""Graph Neural Network models for explainable GNN project."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    GCN2Conv,
    BatchNorm,
    LayerNorm,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


class BaseGNN(nn.Module):
    """Base class for Graph Neural Networks."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ):
        """Initialize base GNN.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_layers: Number of GNN layers.
            dropout: Dropout rate.
            activation: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                BatchNorm(hidden_channels) for _ in range(num_layers - 1)
            ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        raise NotImplementedError
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings (without final classification layer).
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        return self.forward(x, edge_index)


class GCN(BaseGNN):
    """Graph Convolutional Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ):
        """Initialize GCN.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            activation: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, activation, use_batch_norm, use_residual
        )
        
        # GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs[:-1]):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GAT(BaseGNN):
    """Graph Attention Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        activation: str = "elu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        """Initialize GAT.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            activation: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            concat: Whether to concatenate attention heads.
            negative_slope: Negative slope for LeakyReLU.
        """
        super().__init__(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, activation, use_batch_norm, use_residual
        )
        
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        # Calculate hidden dimension per head
        if concat:
            hidden_dim_per_head = hidden_channels // num_heads
        else:
            hidden_dim_per_head = hidden_channels
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            in_channels, hidden_dim_per_head, heads=num_heads,
            dropout=dropout, concat=concat, negative_slope=negative_slope
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels, hidden_dim_per_head, heads=num_heads,
                dropout=dropout, concat=concat, negative_slope=negative_slope
            ))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels, out_channels, heads=1,
                dropout=dropout, concat=False, negative_slope=negative_slope
            ))
        else:
            self.convs.append(GATConv(
                in_channels, out_channels, heads=1,
                dropout=dropout, concat=False, negative_slope=negative_slope
            ))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs[:-1]):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GIN(BaseGNN):
    """Graph Isomorphism Network."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        """Initialize GIN.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_layers: Number of GIN layers.
            dropout: Dropout rate.
            activation: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            eps: Initial epsilon value.
            train_eps: Whether to train epsilon.
        """
        super().__init__(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, activation, use_batch_norm, use_residual
        )
        
        self.eps = eps
        self.train_eps = train_eps
        
        # MLPs for GIN layers
        self.mlps = nn.ModuleList()
        
        # First MLP
        self.mlps.append(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        ))
        
        # Hidden MLPs
        for _ in range(num_layers - 2):
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))
        
        # Output MLP
        if num_layers > 1:
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            ))
        else:
            self.mlps.append(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            ))
        
        # GIN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GINConv(self.mlps[i], eps=eps, train_eps=train_eps))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs[:-1]):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GCN2(BaseGNN):
    """GCN2 with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = True,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        """Initialize GCN2.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_layers: Number of GCN2 layers.
            dropout: Dropout rate.
            activation: Activation function.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            alpha: Alpha parameter for skip connections.
            theta: Theta parameter for skip connections.
        """
        super().__init__(
            in_channels, hidden_channels, out_channels, num_layers,
            dropout, activation, use_batch_norm, use_residual
        )
        
        self.alpha = alpha
        self.theta = theta
        
        # GCN2 layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCN2Conv(
            in_channels, hidden_channels, alpha=alpha, theta=theta
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCN2Conv(
                hidden_channels, hidden_channels, alpha=alpha, theta=theta
            ))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCN2Conv(
                hidden_channels, out_channels, alpha=alpha, theta=theta
            ))
        else:
            self.convs.append(GCN2Conv(
                in_channels, out_channels, alpha=alpha, theta=theta
            ))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs[:-1]):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout_layer(x)
            
            if residual is not None:
                x = x + residual
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


def create_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    **kwargs
) -> BaseGNN:
    """Create a GNN model.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin', 'gcn2').
        in_channels: Input feature dimension.
        hidden_channels: Hidden feature dimension.
        out_channels: Output feature dimension.
        **kwargs: Additional model parameters.
        
    Returns:
        GNN model.
    """
    model_map = {
        "gcn": GCN,
        "gat": GAT,
        "gin": GIN,
        "gcn2": GCN2,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Model type '{model_type}' not supported. Available: {list(model_map.keys())}")
    
    model_class = model_map[model_type]
    return model_class(in_channels, hidden_channels, out_channels, **kwargs)
