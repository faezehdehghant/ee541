
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i+1]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))

        # Additional layers as per your original architecture
        self.convs.append(GCNConv(hidden_dims[-1], hidden_dims[-1]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[-1]))

        # Output layers
        self.convs.append(GCNConv(hidden_dims[-1], hidden_dims[-1]))
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.fc2 = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply graph convolution layers with batch normalization and ReLU activation
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        # Final graph convolution layer without ReLU
        # x = self.convs[-1](x, edge_index)

        # Fully connected layers with ReLU
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Output layer with sigmoid activation to ensure output is in the range (0, 1)
        x = torch.sigmoid(x)

        return x



#####code for #GAT

# GNN_Model.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv

# class GAT(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, heads=1):
#         """
#         A multi-layer Graph Attention Network (GAT).
        
#         Args:
#             input_dim (int): Dimensionality of node features.
#             hidden_dims (list of int): List of hidden layer dimensions.
#             output_dim (int): Dimensionality of the output layer.
#             heads (int): Number of attention heads for each GAT layer.
#                          For multi-head attention, the output of each layer
#                          can be expanded.  If heads > 1 and 'concat=True', 
#                          make sure to account for the multiplied dimensions
#                          in subsequent layers.
#         """
#         super(GAT, self).__init__()
        
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()

#         # --- Input Layer ---
#         # If heads > 1 and concat=True, output dimension becomes hidden_dims[0]*heads.
#         # For simplicity, we set concat=False below to keep dimension = hidden_dims[0].
#         self.convs.append(GATConv(input_dim, hidden_dims[0], heads=heads, concat=False))
#         self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

#         # --- Hidden Layers ---
#         for i in range(len(hidden_dims) - 1):
#             self.convs.append(GATConv(hidden_dims[i], hidden_dims[i+1], heads=heads, concat=False))
#             self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))

#         # --- Additional Graph Layers (matching your original architecture) ---
#         self.convs.append(GATConv(hidden_dims[-1], hidden_dims[-1], heads=heads, concat=False))
#         self.bns.append(nn.BatchNorm1d(hidden_dims[-1]))

#         # --- Output Layers ---
#         # If you want another GAT layer before fully connected layers:
#         self.convs.append(GATConv(hidden_dims[-1], hidden_dims[-1], heads=heads, concat=False))

#         self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
#         self.fc2 = nn.Linear(hidden_dims[-1] // 2, output_dim)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         # --- Apply GAT layers with BatchNorm and ReLU ---
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = self.bns[i](x)
#             x = F.relu(x)

#         # You could apply the final GAT layer directly or similarly with BN/activation:
#         # x = self.convs[-1](x, edge_index)
#         # x = F.relu(x)  # Optional, depending on your design

#         # --- Fully Connected Layers ---
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         # --- Sigmoid for final output ---
#         x = torch.sigmoid(x)

#         return x
