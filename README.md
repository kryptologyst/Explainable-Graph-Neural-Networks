# Explainable Graph Neural Networks

A comprehensive toolkit for training, evaluating, and explaining Graph Neural Network models with multiple explainability methods.

## Features

- **Multiple GNN Architectures**: GCN, GAT, GIN, GCN2 with modern implementations
- **Explainability Methods**: GNNExplainer, PGExplainer, Integrated Gradients, Attention-based explanations
- **Comprehensive Evaluation**: Classification metrics and explainability-specific metrics (fidelity, sparsity, stability)
- **Interactive Demo**: Streamlit-based web interface for exploring explanations
- **Production Ready**: Clean code structure, configuration management, logging, and testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Explainable-Graph-Neural-Networks.git
cd Explainable-Graph-Neural-Networks

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

1. **Train a model**:
```bash
python -m src.cli train --config configs/default.yaml --dataset cora --model gcn
```

2. **Generate explanations**:
```bash
python -m src.cli explain --config configs/default.yaml --nodes test --num-explanations 50
```

3. **Launch interactive demo**:
```bash
python -m src.cli demo
```

## Project Structure

```
explainable-gnn-models/
├── src/                    # Source code
│   ├── models/            # GNN model implementations
│   ├── explain/           # Explainability methods
│   ├── data/              # Data loading and preprocessing
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation metrics
│   ├── utils/             # Utility functions
│   └── cli.py             # Command-line interface
├── configs/               # Configuration files
├── data/                  # Data directory
├── checkpoints/           # Model checkpoints
├── assets/                # Generated assets
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── notebooks/             # Jupyter notebooks
```

## Supported Models

### Graph Convolutional Network (GCN)
- Standard GCN with optional batch normalization and residual connections
- Configuration: `configs/default.yaml`

### Graph Attention Network (GAT)
- Multi-head attention mechanism
- Configurable number of heads and attention parameters
- Configuration: `configs/gat.yaml`

### Graph Isomorphism Network (GIN)
- Powerful for molecular and structural graphs
- Configurable epsilon parameter
- Configuration: `configs/gin.yaml`

### GCN2
- Skip connections and improved training dynamics
- Configurable alpha and theta parameters
- Configuration: `configs/gcn2.yaml`

## Explainability Methods

### GNNExplainer
- Identifies important subgraphs and node features
- Configurable training epochs and learning rate
- Supports both edge and node feature masking

### PGExplainer
- Parameterized explainer with neural network
- Faster inference after training
- Configurable size and entropy coefficients

### Integrated Gradients
- Gradient-based attribution method
- Configurable number of integration steps
- Works with any differentiable model

### Attention-based Explanations
- Extracts attention weights from GAT models
- Visualizes attention patterns
- No additional training required

## Datasets

The toolkit supports multiple graph datasets:

- **Citation Networks**: Cora, CiteSeer, PubMed, CoraFull
- **Co-authorship**: CS, Physics
- **E-commerce**: Amazon Computers, Amazon Photo
- **Social Networks**: Actor, Twitch (PT, DE, EN, RU)
- **Wikipedia**: WikiCS

### Dataset Schema

All datasets follow a standardized format:
- `nodes.csv`: Node features and labels
- `edges.csv`: Edge connections
- `graph_splits.json`: Train/validation/test splits

## Configuration

The project uses YAML configuration files with OmegaConf for flexible parameter management:

```yaml
experiment:
  name: "explainable_gnn_cora"
  seed: 42
  device: "auto"

data:
  dataset: "cora"
  normalize_features: true
  add_self_loops: true

model:
  type: "gcn"
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5

training:
  epochs: 200
  lr: 0.01
  patience: 50

explainability:
  methods: ["gnn_explainer", "integrated_gradients"]
```

## Evaluation Metrics

### Classification Metrics
- Accuracy, F1-Score (macro/micro/weighted)
- Area Under ROC Curve (AUC)
- Precision, Recall

### Explainability Metrics
- **Fidelity**: How well the explanation preserves the original prediction
- **Sparsity**: How concise the explanation is
- **Stability**: Consistency across multiple runs
- **Contrastivity**: How much the explanation changes with prediction changes

## Interactive Demo

The Streamlit demo provides an intuitive interface for:

- **Graph Visualization**: Interactive network plots with node/edge coloring
- **Node Analysis**: Detailed analysis of individual node predictions
- **Explanation Comparison**: Side-by-side comparison of different methods
- **Model Performance**: Comprehensive evaluation metrics and visualizations

Launch the demo:
```bash
streamlit run demo/app.py
```

## API Reference

### Core Classes

#### `GraphDataset`
```python
dataset = GraphDataset(name="cora", root="data/raw")
data = dataset.get_data()
train_mask, val_mask, test_mask = dataset.get_splits()
```

#### `create_model`
```python
model = create_model(
    model_type="gcn",
    in_channels=1433,
    hidden_channels=64,
    out_channels=7
)
```

#### `create_explainer`
```python
explainer = create_explainer(
    method="gnn_explainer",
    model=model,
    device=device
)
explanation = explainer.explain_node(node_idx, x, edge_index)
```

### Training Pipeline

```python
from src.train import train_model

results = train_model(
    config_path="configs/default.yaml",
    dataset="cora",
    model="gcn"
)
```

## Advanced Usage

### Custom Datasets

To use your own dataset, create a CSV file following the schema:

```python
# nodes.csv
node_id,feature_1,feature_2,...,label
0,0.1,0.2,...,0
1,0.3,0.4,...,1
...

# edges.csv
src,dst
0,1
1,2
...
```

### Custom Models

Extend the base model class:

```python
from src.models import BaseGNN

class CustomGNN(BaseGNN):
    def __init__(self, ...):
        super().__init__(...)
        # Your custom implementation
    
    def forward(self, x, edge_index):
        # Your forward pass
        return output
```

### Custom Explainers

Implement the base explainer interface:

```python
from src.explain import BaseExplainer

class CustomExplainer(BaseExplainer):
    def explain_node(self, node_idx, x, edge_index, **kwargs):
        # Your explanation logic
        return {"edge_mask": edge_mask, "node_feat_mask": feat_mask}
```

## Performance Optimization

### Device Support
- **CUDA**: Automatic detection and usage
- **Apple Silicon**: MPS support for M1/M2 chips
- **CPU**: Fallback for all operations

### Memory Optimization
- Gradient checkpointing for large models
- Mixed precision training (optional)
- Efficient data loading

### Scalability
- Neighbor sampling for large graphs
- Distributed training support
- Model parallelism

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_models.py -v
pytest tests/test_explain.py -v
pytest tests/test_data.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** hooks for automated checks

Install pre-commit hooks:
```bash
pre-commit install
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{explainable_gnn_models,
  title={Explainable Graph Neural Networks: A Comprehensive Toolkit},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Explainable-Graph-Neural-Networks}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- Captum team for gradient-based attribution methods
- Streamlit team for the interactive demo framework
- The open-source community for various contributions

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Import errors**: Ensure all dependencies are installed
3. **Dataset not found**: Check data directory and download datasets
4. **Model checkpoint not found**: Train a model first before generating explanations

## Roadmap

- [ ] Support for heterogeneous graphs
- [ ] Temporal graph neural networks
- [ ] More explainability methods (LIME, SHAP)
- [ ] Graph-level explanations
- [ ] Model compression and quantization
- [ ] Web-based deployment
- [ ] Integration with popular ML platforms
# Explainable-Graph-Neural-Networks
