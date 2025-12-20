"""Tests for explainability methods."""

import pytest
import torch
import torch.nn as nn

from src.explain import (
    GNNExplainerWrapper,
    PGExplainer,
    IntegratedGradientsExplainer,
    AttentionExplainer,
    create_explainer,
)
from src.models import GCN, GAT
from src.utils.device import get_device


class TestGNNExplainerWrapper:
    """Test GNNExplainer wrapper."""
    
    def test_gnn_explainer_creation(self):
        """Test GNNExplainer creation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = GNNExplainerWrapper(model, device)
        
        assert isinstance(explainer, GNNExplainerWrapper)
        assert explainer.model == model
        assert explainer.device == device
    
    def test_gnn_explainer_explain_node(self):
        """Test GNNExplainer node explanation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = GNNExplainerWrapper(model, device, epochs=1)  # Quick test
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explanation = explainer.explain_node(0, x, edge_index)
        
        assert "node_feat_mask" in explanation
        assert "edge_mask" in explanation
        assert explanation["edge_mask"].shape == (200,)
        assert explanation["node_feat_mask"].shape == (10,)


class TestPGExplainer:
    """Test PGExplainer."""
    
    def test_pg_explainer_creation(self):
        """Test PGExplainer creation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = PGExplainer(model, device)
        
        assert isinstance(explainer, PGExplainer)
        assert explainer.model == model
        assert explainer.device == device
    
    def test_pg_explainer_explain_node(self):
        """Test PGExplainer node explanation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = PGExplainer(model, device)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explanation = explainer.explain_node(0, x, edge_index)
        
        assert "edge_mask" in explanation
        assert explanation["edge_mask"].shape == (200,)
        assert explanation["node_feat_mask"] is None  # PGExplainer doesn't provide this


class TestIntegratedGradientsExplainer:
    """Test Integrated Gradients explainer."""
    
    def test_ig_explainer_creation(self):
        """Test Integrated Gradients explainer creation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = IntegratedGradientsExplainer(model, device)
        
        assert isinstance(explainer, IntegratedGradientsExplainer)
        assert explainer.model == model
        assert explainer.device == device
    
    def test_ig_explainer_explain_node(self):
        """Test Integrated Gradients node explanation."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = IntegratedGradientsExplainer(model, device, steps=5)  # Quick test
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explanation = explainer.explain_node(0, x, edge_index)
        
        assert "node_feat_mask" in explanation
        assert "edge_mask" in explanation
        assert explanation["node_feat_mask"].shape == (10,)
        assert explanation["edge_mask"].shape == (200,)


class TestAttentionExplainer:
    """Test Attention explainer."""
    
    def test_attention_explainer_creation(self):
        """Test Attention explainer creation."""
        device = get_device()
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            num_heads=4
        )
        
        explainer = AttentionExplainer(model, device)
        
        assert isinstance(explainer, AttentionExplainer)
        assert explainer.model == model
        assert explainer.device == device
    
    def test_attention_explainer_explain_node(self):
        """Test Attention explainer node explanation."""
        device = get_device()
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            num_heads=4
        )
        
        explainer = AttentionExplainer(model, device)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explanation = explainer.explain_node(0, x, edge_index)
        
        assert "node_feat_mask" in explanation
        assert "edge_mask" in explanation
        assert explanation["node_feat_mask"].shape == (10,)
        assert explanation["edge_mask"].shape == (200,)
    
    def test_attention_explainer_invalid_model(self):
        """Test Attention explainer with invalid model."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        with pytest.raises(ValueError):
            AttentionExplainer(model, device)


class TestCreateExplainer:
    """Test explainer creation function."""
    
    def test_create_gnn_explainer(self):
        """Test creating GNNExplainer."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = create_explainer("gnn_explainer", model, device)
        
        assert isinstance(explainer, GNNExplainerWrapper)
    
    def test_create_pg_explainer(self):
        """Test creating PGExplainer."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = create_explainer("pg_explainer", model, device)
        
        assert isinstance(explainer, PGExplainer)
    
    def test_create_integrated_gradients(self):
        """Test creating Integrated Gradients explainer."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        explainer = create_explainer("integrated_gradients", model, device)
        
        assert isinstance(explainer, IntegratedGradientsExplainer)
    
    def test_create_attention_explainer(self):
        """Test creating Attention explainer."""
        device = get_device()
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            num_heads=4
        )
        
        explainer = create_explainer("attention", model, device)
        
        assert isinstance(explainer, AttentionExplainer)
    
    def test_invalid_explainer_method(self):
        """Test invalid explainer method raises error."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        with pytest.raises(ValueError):
            create_explainer("invalid_method", model, device)


class TestExplainerConsistency:
    """Test explainer consistency."""
    
    def test_explanation_shapes(self):
        """Test that all explainers return consistent shapes."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explainers = [
            GNNExplainerWrapper(model, device, epochs=1),
            PGExplainer(model, device),
            IntegratedGradientsExplainer(model, device, steps=5),
        ]
        
        for explainer in explainers:
            explanation = explainer.explain_node(0, x, edge_index)
            
            assert "edge_mask" in explanation
            assert explanation["edge_mask"].shape == (200,)
            
            if explanation["node_feat_mask"] is not None:
                assert explanation["node_feat_mask"].shape == (10,)
    
    def test_explanation_values(self):
        """Test that explanations return valid values."""
        device = get_device()
        model = GCN(in_channels=10, hidden_channels=32, out_channels=5)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        explainer = GNNExplainerWrapper(model, device, epochs=1)
        explanation = explainer.explain_node(0, x, edge_index)
        
        # Check that values are not NaN or infinite
        assert not torch.isnan(explanation["edge_mask"]).any()
        assert not torch.isinf(explanation["edge_mask"]).any()
        
        if explanation["node_feat_mask"] is not None:
            assert not torch.isnan(explanation["node_feat_mask"]).any()
            assert not torch.isinf(explanation["node_feat_mask"]).any()


if __name__ == "__main__":
    pytest.main([__file__])
