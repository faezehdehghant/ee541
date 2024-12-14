# predict.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
from GNN_Architecture import GCN  # Import the GCN model from model.py

# Initialize the model with the same dimensions used during training
input_dim = 3  # transmission_prob_log, transmission_prob_1m_log, theoretical_throughput
hidden_dims = [64, 128, 256, 128, 64]  # Same as training
output_dim = 1  # Predicting saturation throughput per node

model = GCN(input_dim, hidden_dims, output_dim)

# Load the trained model weights
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=True))
model.eval()  # Set the model to evaluation mode

def preprocess_input(adj_matrix, transmission_prob, theoretical_throughput):
    """
    Preprocesses the input adjacency matrix and transmission probabilities.
    """
    # Ensure adjacency matrix is a numpy array
    adj_matrix = np.array(adj_matrix)
    
    # Number of nodes
    num_nodes = adj_matrix.shape[0]
    
    # Handle log(0) by setting to a large negative number to avoid -inf
    transmission_prob_log = [np.log(p) if p > 0 else -1e10 for p in transmission_prob]
    transmission_prob_1m_log = [np.log(1 - p) if p < 1 else -1e10 for p in transmission_prob]
    
    # Combine transmission_prob_log, transmission_prob_1m_log, and theoretical_throughput into node features
    transmission_prob_log = np.array(transmission_prob_log).reshape(-1, 1)
    transmission_prob_1m_log = np.array(transmission_prob_1m_log).reshape(-1, 1)
    theoretical_throughput = np.array(theoretical_throughput).reshape(-1, 1)
    x = np.hstack((transmission_prob_log, transmission_prob_1m_log, theoretical_throughput))
    x = torch.tensor(x, dtype=torch.float)
    
    # Create edge_index from adjacency matrix
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                edge_index.append([i, j])
    
    if len(edge_index) == 0:
        raise ValueError("The adjacency matrix has no edges.")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create the Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data

def predict_saturation_throughput(model, adj_matrix, transmission_prob, theoretical_throughput):
    """
    Predicts the saturation throughput for each node in the graph.
    """
    # Preprocess the input
    data = preprocess_input(adj_matrix, transmission_prob, theoretical_throughput)
    
    # Move data to the same device as the model
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(data)
    
    # Move output to CPU and convert to numpy
    prediction = output.cpu().numpy().flatten()
    
    return prediction

if __name__ == "__main__":
    # Example input
    example_adj_matrix = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
    
    example_transmission_prob = [0.2, 0.3, 0.7]
    
    # Theoretical saturation throughput (must be provided)
    example_theoretical_throughput = [0.2581, 0.1319, 0.3529]
    
    # Make a prediction
    predicted_throughput = predict_saturation_throughput(model, example_adj_matrix, example_transmission_prob, example_theoretical_throughput)
    
    # Display the results
    print("\nPrediction for Example Input:")
    for node_idx, pred in enumerate(predicted_throughput, 1):
        print(f"Node {node_idx}: Predicted Saturation Throughput = {pred:.4f}")
