"""Interactive Streamlit demo for explainable GNN models."""

import os
import sys
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.device import get_device, set_seed
from src.utils.config import Config
from src.data import GraphDataset
from src.models import create_model
from src.explain import create_explainer
from src.eval import ClassificationMetrics, ExplainabilityMetrics


@st.cache_data
def load_dataset(dataset_name: str) -> GraphDataset:
    """Load dataset with caching."""
    return GraphDataset(
        name=dataset_name,
        root="data/raw",
        transform=["normalize_features", "add_self_loops"],
    )


@st.cache_data
def load_model(config: Config, dataset: GraphDataset) -> torch.nn.Module:
    """Load trained model with caching."""
    device = get_device()
    
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
    
    return model


def visualize_graph(
    data: torch.Tensor,
    edge_index: torch.Tensor,
    node_colors: Optional[torch.Tensor] = None,
    edge_colors: Optional[torch.Tensor] = None,
    max_nodes: int = 500,
) -> go.Figure:
    """Create interactive graph visualization."""
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Sample nodes if graph is too large
    if len(G.nodes()) > max_nodes:
        nodes_to_keep = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        G = G.subgraph(nodes_to_keep)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract coordinates
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge color based on importance
        if edge_colors is not None:
            edge_idx = None
            for i, (src, tgt) in enumerate(data.edge_index.t().cpu().numpy()):
                if (src == edge[0] and tgt == edge[1]) or (src == edge[1] and tgt == edge[0]):
                    edge_idx = i
                    break
            
            if edge_idx is not None:
                importance = edge_colors[edge_idx].item()
                edge_info.append(f"Edge {edge[0]}-{edge[1]}: Importance = {importance:.3f}")
            else:
                edge_info.append(f"Edge {edge[0]}-{edge[1]}")
        else:
            edge_info.append(f"Edge {edge[0]}-{edge[1]}")
    
    # Create node traces
    node_info = []
    for node in G.nodes():
        if node_colors is not None and node < len(node_colors):
            color = node_colors[node].item()
            node_info.append(f"Node {node}: Class = {int(color)}")
        else:
            node_info.append(f"Node {node}")
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(0,0,0,0.2)'),
        hoverinfo='none',
        mode='lines',
        name='Edges'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            size=10,
            color=node_colors[:len(G.nodes())].cpu().numpy() if node_colors is not None else 'lightblue',
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node Class"),
        ),
        text=node_info,
        hoverinfo='text',
        name='Nodes'
    ))
    
    fig.update_layout(
        title="Graph Visualization",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive graph visualization",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Explainable GNN Models",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 Explainable Graph Neural Networks")
    st.markdown("Interactive exploration of GNN explanations and model predictions")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["cora", "citeseer", "pubmed", "corafull"],
        index=0
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["gcn", "gat", "gin", "gcn2"],
        index=0
    )
    
    # Explanation method
    explain_method = st.sidebar.selectbox(
        "Explanation Method",
        ["gnn_explainer", "pg_explainer", "integrated_gradients", "attention"],
        index=0
    )
    
    # Load configuration
    config_path = f"configs/{model_type}.yaml"
    if not os.path.exists(config_path):
        config_path = "configs/default.yaml"
    
    config = Config(config_path)
    config.set("data.dataset", dataset_name)
    config.set("model.type", model_type)
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset = load_dataset(dataset_name)
        data = dataset.get_data()
        train_mask, val_mask, test_mask = dataset.get_splits()
    
    # Display dataset statistics
    st.sidebar.subheader("Dataset Statistics")
    stats = dataset.get_statistics()
    st.sidebar.metric("Nodes", stats["num_nodes"])
    st.sidebar.metric("Edges", stats["num_edges"])
    st.sidebar.metric("Features", stats["num_features"])
    st.sidebar.metric("Classes", stats["num_classes"])
    st.sidebar.metric("Density", f"{stats['density']:.4f}")
    
    # Load model
    with st.spinner("Loading model..."):
        device = get_device()
        model = load_model(config, dataset)
        model.eval()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Graph Overview", "Node Analysis", "Explanation Comparison", "Model Performance"])
    
    with tab1:
        st.header("Graph Overview")
        
        # Graph visualization
        fig = visualize_graph(data, data.edge_index, data.y)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Dataset**: {dataset_name.upper()}")
            st.write(f"**Number of nodes**: {stats['num_nodes']:,}")
            st.write(f"**Number of edges**: {stats['num_edges']:,}")
            st.write(f"**Number of features**: {stats['num_features']}")
            st.write(f"**Number of classes**: {stats['num_classes']}")
        
        with col2:
            st.subheader("Class Distribution")
            if "class_distribution" in stats:
                class_counts = stats["class_distribution"]
                fig_pie = px.pie(
                    values=class_counts,
                    names=[f"Class {i}" for i in range(len(class_counts))],
                    title="Node Class Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.header("Node Analysis")
        
        # Node selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            node_set = st.selectbox("Node Set", ["Test", "Validation", "Training"], index=0)
            if node_set == "Test":
                available_nodes = torch.where(test_mask)[0].tolist()
            elif node_set == "Validation":
                available_nodes = torch.where(val_mask)[0].tolist()
            else:
                available_nodes = torch.where(train_mask)[0].tolist()
            
            node_idx = st.selectbox("Node Index", available_nodes[:100], index=0)
        
        with col2:
            # Get node prediction
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred_prob = torch.softmax(out[node_idx], dim=0)
                pred_class = pred_prob.argmax().item()
                true_class = data.y[node_idx].item()
            
            st.subheader(f"Node {node_idx} Analysis")
            
            col_pred, col_true = st.columns(2)
            with col_pred:
                st.metric("Predicted Class", pred_class)
                st.metric("Confidence", f"{pred_prob[pred_class]:.3f}")
            
            with col_true:
                st.metric("True Class", true_class)
                st.metric("Correct", "✅" if pred_class == true_class else "❌")
            
            # Prediction probabilities
            fig_prob = px.bar(
                x=[f"Class {i}" for i in range(len(pred_prob))],
                y=pred_prob.cpu().numpy(),
                title=f"Prediction Probabilities for Node {node_idx}"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Generate explanation
        if st.button("Generate Explanation"):
            with st.spinner("Generating explanation..."):
                # Create explainer
                explainer_config = config.get(f"explainability.{explain_method}", {})
                explainer = create_explainer(explain_method, model, device, **explainer_config)
                
                # Generate explanation
                explanation = explainer.explain_node(node_idx, data.x, data.edge_index)
                
                # Display explanation
                st.subheader(f"Explanation using {explain_method}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Edge Importance**")
                    edge_mask = explanation["edge_mask"]
                    
                    # Get edges connected to the node
                    node_edges = []
                    node_edge_importance = []
                    
                    for i, (src, tgt) in enumerate(data.edge_index.t().cpu().numpy()):
                        if src == node_idx or tgt == node_idx:
                            node_edges.append(f"{src}-{tgt}")
                            node_edge_importance.append(edge_mask[i].item())
                    
                    if node_edges:
                        fig_edges = px.bar(
                            x=node_edges,
                            y=node_edge_importance,
                            title=f"Edge Importance for Node {node_idx}"
                        )
                        st.plotly_chart(fig_edges, use_container_width=True)
                    else:
                        st.write("No edges found for this node.")
                
                with col2:
                    st.write("**Feature Importance**")
                    if explanation["node_feat_mask"] is not None:
                        feat_mask = explanation["node_feat_mask"]
                        if feat_mask.dim() == 1:
                            feat_mask = feat_mask.unsqueeze(0)
                        
                        fig_features = px.bar(
                            x=[f"Feature {i}" for i in range(len(feat_mask[0]))],
                            y=feat_mask[0].cpu().numpy(),
                            title=f"Feature Importance for Node {node_idx}"
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                    else:
                        st.write("Feature importance not available for this method.")
                
                # Visualize explanation subgraph
                st.subheader("Explanation Subgraph")
                
                # Create subgraph with important edges
                threshold = st.slider("Edge Importance Threshold", 0.0, 1.0, 0.5)
                important_edges = edge_mask > threshold
                
                if important_edges.sum() > 0:
                    subgraph_edge_index = data.edge_index[:, important_edges]
                    
                    # Get nodes in subgraph
                    subgraph_nodes = torch.unique(subgraph_edge_index).cpu().numpy()
                    
                    # Create subgraph visualization
                    fig_subgraph = visualize_graph(
                        data,
                        subgraph_edge_index,
                        data.y,
                        edge_mask,
                        max_nodes=100
                    )
                    fig_subgraph.update_layout(title=f"Explanation Subgraph (threshold={threshold:.2f})")
                    st.plotly_chart(fig_subgraph, use_container_width=True)
                else:
                    st.write("No edges meet the importance threshold.")
    
    with tab3:
        st.header("Explanation Comparison")
        
        # Select multiple nodes for comparison
        num_nodes = st.slider("Number of nodes to compare", 1, 20, 5)
        
        # Get test nodes
        test_nodes = torch.where(test_mask)[0].tolist()[:num_nodes]
        
        if st.button("Compare Explanations"):
            with st.spinner("Generating explanations..."):
                methods = ["gnn_explainer", "integrated_gradients"]
                if model_type == "gat":
                    methods.append("attention")
                
                all_explanations = {}
                
                for method in methods:
                    explainer_config = config.get(f"explainability.{method}", {})
                    explainer = create_explainer(method, model, device, **explainer_config)
                    
                    explanations = []
                    for node_idx in test_nodes:
                        explanation = explainer.explain_node(node_idx, data.x, data.edge_index)
                        explanations.append(explanation)
                    
                    all_explanations[method] = explanations
                
                # Compare edge importance
                st.subheader("Edge Importance Comparison")
                
                for i, node_idx in enumerate(test_nodes):
                    st.write(f"**Node {node_idx}**")
                    
                    # Get edges for this node
                    node_edges = []
                    edge_importance = {method: [] for method in methods}
                    
                    for j, (src, tgt) in enumerate(data.edge_index.t().cpu().numpy()):
                        if src == node_idx or tgt == node_idx:
                            node_edges.append(f"{src}-{tgt}")
                            for method in methods:
                                edge_importance[method].append(
                                    all_explanations[method][i]["edge_mask"][j].item()
                                )
                    
                    if node_edges:
                        # Create comparison plot
                        fig_comparison = go.Figure()
                        
                        for method in methods:
                            fig_comparison.add_trace(go.Bar(
                                name=method,
                                x=node_edges,
                                y=edge_importance[method]
                            ))
                        
                        fig_comparison.update_layout(
                            title=f"Edge Importance Comparison for Node {node_idx}",
                            xaxis_title="Edges",
                            yaxis_title="Importance",
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab4:
        st.header("Model Performance")
        
        # Evaluate model
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # Test set evaluation
            test_pred = out[test_mask].argmax(dim=1)
            test_true = data.y[test_mask]
            test_prob = torch.softmax(out[test_mask], dim=1)
            
            # Compute metrics
            classification_metrics = ClassificationMetrics(dataset.num_classes, task="multiclass")
            test_metrics = classification_metrics.compute(test_pred, test_true, test_prob)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1 Macro", f"{test_metrics['f1_macro']:.4f}")
        with col3:
            st.metric("F1 Micro", f"{test_metrics['f1_micro']:.4f}")
        with col4:
            st.metric("AUC", f"{test_metrics['auc']:.4f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_true.cpu().numpy(), test_pred.cpu().numpy())
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="True"),
            x=[f"Class {i}" for i in range(dataset.num_classes)],
            y=[f"Class {i}" for i in range(dataset.num_classes)]
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Class-wise performance
        st.subheader("Class-wise Performance")
        
        class_metrics = []
        for i in range(dataset.num_classes):
            class_mask = test_true == i
            if class_mask.sum() > 0:
                class_acc = (test_pred[class_mask] == test_true[class_mask]).float().mean().item()
                class_metrics.append({
                    "Class": f"Class {i}",
                    "Accuracy": class_acc,
                    "Count": class_mask.sum().item()
                })
        
        if class_metrics:
            df_metrics = pd.DataFrame(class_metrics)
            st.dataframe(df_metrics, use_container_width=True)


if __name__ == "__main__":
    main()
