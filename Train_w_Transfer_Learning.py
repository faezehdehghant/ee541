import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from GNN_Architecture import GCN  # Ensure this file contains the GCN class as per your code

# Theoretical Saturation Throughput Function (same as your original code)
def theoretical_saturation_throughput_old(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * T
        denominator = (sigma * prod_other * (1 - p_i)
                       + (p_i * prod_other) * T
                       + (1 - ((prod_other * (1 - p_i)) + p_i * prod_other)) * T)
        S[i] = numerator / denominator
    return S.tolist()

# Function to create Data object from adjacency matrix and transmission probabilities (same as original)
def create_data(transmission_prob, transmission_prob_log, transmission_prob_1m_log, G, saturation_throughput, theoretical_throughput):
    edge_index = []

    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == 1:
                edge_index.append([i, j])

    if len(edge_index) == 0:
        return None  # Skip if there are no edges

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    transmission_prob = np.array(transmission_prob).reshape(-1, 1)
    transmission_prob_log = np.array(transmission_prob_log).reshape(-1, 1)
    transmission_prob_1m_log = np.array(transmission_prob_1m_log).reshape(-1, 1)
    theoretical_throughput = np.array(theoretical_throughput).reshape(-1, 1)

    # Features: [transmission_prob_log, transmission_prob_1m_log, theoretical_throughput]
    x = np.hstack((transmission_prob_log, transmission_prob_1m_log, theoretical_throughput))
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(saturation_throughput, dtype=torch.float).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Function to read the CSV and create data objects (same as original)
def read_csv_and_create_data(csv_path, T, sigma):
    df = pd.read_csv(csv_path)
    data_list = []
    adj_matrices = []
    transmission_probs = []
    transmission_probs_log = []
    transmission_probs_1m_log = []
    theoretical_throughputs = []
    for _, row in df.iterrows():
        transmission_prob_1m_log_temp = []
        transmission_prob_log_temp = []
        adj_matrix = ast.literal_eval(row['adj_matrix'])
        transmission_prob = ast.literal_eval(row['transmission_prob'])
        saturation_throughput = ast.literal_eval(row['saturation_throughput'])
        theoretical_throughput = theoretical_saturation_throughput_old(transmission_prob, T, sigma, adj_matrix)
        for j in range(len(transmission_prob)):
            transmission_prob_log_temp.append(np.log(transmission_prob[j]))
            transmission_prob_1m_log_temp.append(np.log(1-transmission_prob[j]))
        data = create_data(transmission_prob, transmission_prob_log_temp, transmission_prob_1m_log_temp, adj_matrix, saturation_throughput, theoretical_throughput)
        if data is not None:
            data_list.append(data)
            adj_matrices.append(adj_matrix)
            transmission_probs_log.append(transmission_prob_log_temp)
            transmission_probs_1m_log.append(transmission_prob_1m_log_temp)
            transmission_probs.append(transmission_prob)
            theoretical_throughputs.append(theoretical_throughput)
    return data_list, adj_matrices, transmission_probs, transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs

# Split data (same as original)
def split_data(dataset, adj_matrices, transmission_probs, transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs, test_size=0.2, val_size=0.1):
    train_val_data, test_data, train_val_adj, test_adj, train_val_trans, test_trans,\
    train_val_trans_log, test_trans_log, train_val_trans_1m_log, test_trans_1m_log, train_val_theo, test_theo = train_test_split(
        dataset, adj_matrices, transmission_probs, transmission_probs_log, transmission_probs_1m_log,
        theoretical_throughputs, test_size=test_size, random_state=42)
    
    train_data, val_data, train_adj, val_adj, train_trans, val_trans, train_trans_log, val_trans_log, train_trans_1m_log, val_trans_1m_log, train_theo, val_theo = train_test_split(
        train_val_data, train_val_adj, train_val_trans, train_val_trans_log, train_val_trans_1m_log, train_val_theo,
        test_size=val_size / (1 - test_size), random_state=42)
    return train_data, val_data, test_data, train_adj, val_adj, test_adj, train_trans, val_trans, test_trans,\
           train_trans_log, val_trans_log, test_trans_log, train_trans_1m_log, val_trans_1m_log, test_trans_1m_log,\
           train_theo, val_theo, test_theo

# Evaluation function (same as original)
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_error = 0
    theoretical_loss = 0
    theoretical_error = 0
    predicted = []
    actual = []
    theoretical = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            target = data.y
            theoretical_throughput = data.x[:, 2].view(-1, 1)  # index 2 is theoretical throughput
            loss = criterion(out, target)
            total_loss += loss.item()
            total_error += torch.abs(out - target).sum().item()
            theoretical_loss += criterion(theoretical_throughput, target).item()
            theoretical_error += torch.abs(theoretical_throughput - target).sum().item()
            predicted.append(out.cpu().numpy())
            actual.append(target.cpu().numpy())
            theoretical.append(theoretical_throughput.cpu().numpy())
    return (total_loss / len(loader), total_error / len(loader.dataset),
            theoretical_loss / len(loader), theoretical_error / len(loader.dataset),
            np.vstack(predicted), np.vstack(actual), np.vstack(theoretical))


# Pre-trained model weights (from 8-node training)
pretrained_weights_path = '8best_model.pt'  # Ensure this file exists

# Specify dataset details
csv_path_10 = '10Data.csv'  # Your 10-node dataset
T = 2
sigma = 1

# Read the 10-node dataset and create data objects
data_list_10, adj_matrices_10, transmission_probs_10, trans_probs_log_10, trans_probs_1m_log_10, theoretical_throughputs_10 = read_csv_and_create_data(csv_path_10, T, sigma)

# Split into train/val/test
train_data_10, val_data_10, test_data_10, train_adj_10, val_adj_10, test_adj_10, train_trans_10, val_trans_10, test_trans_10,\
train_trans_log_10, val_trans_log_10, test_trans_log_10, train_trans_1m_log_10, val_trans_1m_log_10, test_trans_1m_log_10,\
train_theo_10, val_theo_10, test_theo_10 = split_data(data_list_10, adj_matrices_10, transmission_probs_10,
                                                      trans_probs_log_10, trans_probs_1m_log_10, theoretical_throughputs_10)

train_loader_10 = DataLoader(train_data_10, batch_size=32, shuffle=True)
val_loader_10 = DataLoader(val_data_10, batch_size=32, shuffle=False)
test_loader_10 = DataLoader(test_data_10, batch_size=32, shuffle=False)

# Initialize the model with the same architecture
input_dim = 3  
hidden_dims = [64, 128, 256, 128, 64]
output_dim = 1
model_10 = GCN(input_dim, hidden_dims, output_dim)

# Load the pre-trained weights from the 8-node model
model_10.load_state_dict(torch.load(pretrained_weights_path), strict=True)

# Optionally, freeze some layers
for param in model_10.convs.parameters():
    param.requires_grad = False
for param in model_10.bns.parameters():
    param.requires_grad = False

optimizer_10 = torch.optim.Adam(model_10.parameters(), lr=0.001)
scheduler_10 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_10, mode='min', factor=0.5, patience=5)
criterion = torch.nn.MSELoss()

best_val_loss_10 = float('inf')

print("Fine-tuning on 10-node dataset...")
for epoch in range(230):
    model_10.train()
    total_loss = 0
    for batch in train_loader_10:
        optimizer_10.zero_grad()
        out = model_10(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer_10.step()
        total_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_loss / len(train_loader_10.dataset)
    val_loss, val_error, val_theoretical_loss, val_theoretical_error, _, _, _ = evaluate(model_10, val_loader_10, criterion)
    scheduler_10.step(val_loss)

    if val_loss < best_val_loss_10:
        best_val_loss_10 = val_loss
        torch.save(model_10.state_dict(), 'best_model_10_nodes.pt')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Validation Error: {val_error:.4f}')

# Load the best model after fine-tuning
model_10.load_state_dict(torch.load('best_model_10_nodes.pt'))

# Evaluation on test set
test_loss, test_error, test_theoretical_loss, test_theoretical_error, test_pred, test_actual, test_theoretical = evaluate(model_10, test_loader_10, criterion)
print(f'Test Loss (10-nodes): {test_loss:.4f}, Test Error (10-nodes): {test_error:.4f}')

# Plot comparison of actual vs predicted on 10-node dataset
plt.figure(figsize=(10, 6))
plt.plot(test_actual.flatten(), label='Actual Saturation Throughput')
plt.plot(test_pred.flatten(), label='Predicted Saturation Throughput', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Saturation Throughput')
plt.legend()
plt.title('Comparison of Actual vs. Predicted (10-nodes, after Transfer Learning)')
plt.show()

combined_results = list(zip(test_pred.flatten(), test_actual.flatten()))

# Select 10 random samples
random_samples = random.sample(combined_results, 10)

print("\nResults for 10 Random Samples:")
print("Sample\tPredicted\tActual")
for idx, (pred, actual) in enumerate(random_samples, 1):
    print(f"{idx}\t{pred:.4f}\t\t{actual:.4f}")

