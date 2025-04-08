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
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CustomMPLayer(MessagePassing):
    """
    Custom Message Passing Layer.

    This layer defines a custom message passing mechanism using a linear transformation.
    It aggregates messages using the mean aggregation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format of shape [2, num_edges].

        Returns:
            torch.Tensor: Output node feature matrix of shape [num_nodes, out_channels].
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        Message passing function.

        Args:
            x_j (torch.Tensor): Source node features of shape [num_edges, in_channels].

        Returns:
            torch.Tensor: Transformed source node features of shape [num_edges, out_channels].
        """
        return self.lin(x_j)

class MGModel(torch.nn.Module):
    """
    Multi-Graph Model.

    This model consists of multiple graph convolutional layers, batch normalization,
    and dropout. It supports various graph convolutional layer types.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        first_layer_type (str): Type of the first graph convolutional layer.
        second_layer_type (str): Type of the second graph convolutional layer.
        hidden_channels (int): Number of hidden units.
        dropout_rate (float): Dropout probability.
        gat_heads (int, optional): Number of attention heads for GATConv. Defaults to 1.
        transformer_heads (int, optional): Number of attention heads for TransformerConv. Defaults to 1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        first_layer_type: str,
        second_layer_type: str,
        hidden_channels: int,
        dropout_rate: float,
        gat_heads: int = 1,
        transformer_heads: int = 1
    ):
        super(MGModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.gat_heads = gat_heads
        self.transformer_heads = transformer_heads

        try:
            if first_layer_type == "custom_mp":
                self.conv1 = CustomMPLayer(in_channels, hidden_channels)
            elif first_layer_type == "gcn":
                self.conv1 = GCNConv(in_channels, hidden_channels)
            elif first_layer_type == "gat":
                self.conv1 = GATConv(in_channels, hidden_channels, heads=gat_heads, dropout=dropout_rate)
            elif first_layer_type == "sage":
                self.conv1 = SAGEConv(in_channels, hidden_channels)
            elif first_layer_type == "gin":
                self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
            elif first_layer_type == "graph_conv":
                self.conv1 = GraphConv(in_channels, hidden_channels)
            elif first_layer_type == "transformer_conv":
                self.conv1 = TransformerConv(in_channels, hidden_channels, heads=transformer_heads)
            else:
                self.conv1 = GCNConv(in_channels, hidden_channels) # default case
        except Exception as e:
            logger.error(f"Error initializing first layer: {e}")
            raise ModelLayerInitializationError(f"Failed to initialize first layer: {e}")

        self.bcn1 = nn.BatchNorm1d(hidden_channels)

        try:
            if second_layer_type == "custom_mp":
                self.conv_gcn = CustomMPLayer(hidden_channels, hidden_channels)
            elif second_layer_type == "gcn":
                self.conv_gcn = GCNConv(hidden_channels, hidden_channels)
            elif second_layer_type == "gat":
                self.conv_gcn = GATConv(hidden_channels, hidden_channels, heads=gat_heads, dropout=dropout_rate)
            elif second_layer_type == "sage":
                self.conv_gcn = SAGEConv(hidden_channels, hidden_channels)
            elif second_layer_type == "gin":
                self.conv_gcn = GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
            elif second_layer_type == "graph_conv":
                self.conv_gcn = GraphConv(hidden_channels, hidden_channels)
            elif second_layer_type == "transformer_conv":
                self.conv_gcn = TransformerConv(hidden_channels, hidden_channels, heads=transformer_heads)
            else:
                self.conv_gcn = GCNConv(hidden_channels, hidden_channels) # Default case
        except Exception as e:
            logger.error(f"Error initializing second layer: {e}")
            raise ModelLayerInitializationError(f"Failed to initialize second layer: {e}")

        self.bcn_gcn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = CustomMPLayer(hidden_channels, hidden_channels * 2)
        self.bcn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.linout = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format of shape [2, num_edges].
            batch (torch.Tensor): Batch vector of shape [num_nodes].

        Returns:
            Tuple[torch.Tensor, float]: Output tensor and L1 regularization term.
        """
        x = self.conv1(x, edge_index)
        x = self.bcn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_gcn(x, edge_index)
        x = self.bcn_gcn(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bcn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.linout(x)

        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.abs(param).sum()
        l1_reg = 0 

        return x, l1_reg

    
