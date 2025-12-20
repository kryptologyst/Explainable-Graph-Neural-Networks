"""Tests for GNN models."""

import pytest
import torch
import torch.nn as nn

from src.models import GCN, GAT, GIN, GCN2, create_model
from src.utils.device import get_device


class TestGCN:
    """Test GCN model."""
    
    def test_gcn_creation(self):
        """Test GCN model creation."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        assert isinstance(model, nn.Module)
        assert len(model.convs) == 2
        assert model.in_channels == 10
        assert model.out_channels == 5
    
    def test_gcn_forward(self):
        """Test GCN forward pass."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 5)
        assert not torch.isnan(output).any()
    
    def test_gcn_with_batch_norm(self):
        """Test GCN with batch normalization."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            use_batch_norm=True
        )
        
        assert len(model.batch_norms) == 1  # num_layers - 1
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        assert output.shape == (100, 5)


class TestGAT:
    """Test GAT model."""
    
    def test_gat_creation(self):
        """Test GAT model creation."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            num_heads=4
        )
        
        assert isinstance(model, nn.Module)
        assert len(model.convs) == 2
        assert model.num_heads == 4
    
    def test_gat_forward(self):
        """Test GAT forward pass."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            num_heads=4
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 5)
        assert not torch.isnan(output).any()


class TestGIN:
    """Test GIN model."""
    
    def test_gin_creation(self):
        """Test GIN model creation."""
        model = GIN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        assert isinstance(model, nn.Module)
        assert len(model.convs) == 2
        assert len(model.mlps) == 2
    
    def test_gin_forward(self):
        """Test GIN forward pass."""
        model = GIN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 5)
        assert not torch.isnan(output).any()


class TestGCN2:
    """Test GCN2 model."""
    
    def test_gcn2_creation(self):
        """Test GCN2 model creation."""
        model = GCN2(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        assert isinstance(model, nn.Module)
        assert len(model.convs) == 2
    
    def test_gcn2_forward(self):
        """Test GCN2 forward pass."""
        model = GCN2(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 5)
        assert not torch.isnan(output).any()


class TestCreateModel:
    """Test model creation function."""
    
    def test_create_gcn(self):
        """Test creating GCN model."""
        model = create_model(
            model_type="gcn",
            in_channels=10,
            hidden_channels=32,
            out_channels=5
        )
        
        assert isinstance(model, GCN)
    
    def test_create_gat(self):
        """Test creating GAT model."""
        model = create_model(
            model_type="gat",
            in_channels=10,
            hidden_channels=32,
            out_channels=5
        )
        
        assert isinstance(model, GAT)
    
    def test_create_gin(self):
        """Test creating GIN model."""
        model = create_model(
            model_type="gin",
            in_channels=10,
            hidden_channels=32,
            out_channels=5
        )
        
        assert isinstance(model, GIN)
    
    def test_create_gcn2(self):
        """Test creating GCN2 model."""
        model = create_model(
            model_type="gcn2",
            in_channels=10,
            hidden_channels=32,
            out_channels=5
        )
        
        assert isinstance(model, GCN2)
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model(
                model_type="invalid",
                in_channels=10,
                hidden_channels=32,
                out_channels=5
            )


class TestModelParameters:
    """Test model parameter counting."""
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = GCN(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2
        )
        
        # Count parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable


if __name__ == "__main__":
    pytest.main([__file__])
