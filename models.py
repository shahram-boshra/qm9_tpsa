# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_mean_pool,
    MessagePassing,
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    GraphConv,
    TransformerConv,
)
import logging
from exceptions import ModelLayerInitializationError
from typing import Tuple, Optional, List
import torch_geometric.data

logger = logging.getLogger(__name__)

class CustomMPLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.lin(x_j)

class MGModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        layer_types: List[str],
        hidden_channels: int,
        dropout_rate: float,
        gat_heads: int = 1,
        transformer_heads: int = 1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.identities = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        for i in range(num_layers):
            layer_type = layer_types[i % len(layer_types)]
            in_dim = in_channels if i == 0 else hidden_channels
            try:
                if layer_type == "custom_mp":
                    conv = CustomMPLayer(in_dim, hidden_channels)
                elif layer_type == "gcn":
                    conv = GCNConv(in_dim, hidden_channels)
                elif layer_type == "gat":
                    conv = GATConv(in_dim, hidden_channels, heads=gat_heads, dropout=dropout_rate)
                elif layer_type == "sage":
                    conv = SAGEConv(in_dim, hidden_channels)
                elif layer_type == "gin":
                    conv = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
                elif layer_type == "graph_conv":
                    conv = GraphConv(in_dim, hidden_channels)
                elif layer_type == "transformer_conv":
                    conv = TransformerConv(in_dim, hidden_channels, heads=transformer_heads)
                else:
                    raise ValueError(f"Unsupported layer type: {layer_type}")
                self.convs.append(conv)
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            except Exception as e:
                logger.error(f"Error initializing layer {i+1}: {e}")
                raise ModelLayerInitializationError(f"Failed to initialize layer {i+1}: {e}")

        self.linout = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, edge_index=None, batch=None) -> Tuple[torch.Tensor, float]:
        if isinstance(data, torch_geometric.data.Data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x = data
            if edge_index is None or batch is None:
                raise ValueError("If 'data' is not a Data object, 'edge_index' and 'batch' must be provided.")

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self.identities[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.linout(x)

        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.abs(param).sum()
        l1_reg = 0

        return x, l1_reg
