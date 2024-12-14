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
from GNN_Architecture import GCN

# Theoretical Saturation Throughput Function
def theoretical_saturation_throughput_old(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * T
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1 - ((prod_other * (1 - p_i)) + p_i * prod_other)) * T
        S[i] = numerator / denominator
    return S.tolist()

num_nodes=6
# Function to create Data object from adjacency matrix and transmission probabilities
def create_data(transmission_prob, transmission_prob_log,transmission_prob_1m_log, G, saturation_throughput, theoretical_throughput):
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

    # Combine transmission_prob and theoretical throughput
    x = np.hstack((transmission_prob_log,transmission_prob_1m_log,theoretical_throughput))
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(saturation_throughput, dtype=torch.float).view(-1, 1)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Function to read the CSV and create data objects
def read_csv_and_create_data(csv_path, T, sigma):
    df = pd.read_csv(csv_path)
    data_list = []
    adj_matrices = []
    transmission_probs = []
    transmission_probs_log = []
    transmission_probs_1m_log = []
    theoretical_throughputs = []
    for _, row in df.iterrows():
        transmission_prob_1m_log = []
        transmission_prob_log  = []
        adj_matrix = ast.literal_eval(row['adj_matrix'])
        transmission_prob = ast.literal_eval(row['transmission_prob'])
        saturation_throughput = ast.literal_eval(row['saturation_throughput'])
        theoretical_throughput = theoretical_saturation_throughput_old(transmission_prob, T, sigma, adj_matrix)
        for j in range(len(transmission_prob)):
            transmission_prob_log.append(np.log(transmission_prob[j]))
            transmission_prob_1m_log.append(np.log(1-transmission_prob[j]))
        data = create_data(transmission_prob, transmission_prob_log, transmission_prob_1m_log,adj_matrix, saturation_throughput, theoretical_throughput)
        if data is not None:
            data_list.append(data)
            adj_matrices.append(adj_matrix)
            transmission_probs_log.append(transmission_prob_log)
            transmission_probs_1m_log.append(transmission_prob_1m_log)
            transmission_probs.append(transmission_prob)
            theoretical_throughputs.append(theoretical_throughput)
    return data_list, adj_matrices, transmission_probs, transmission_probs_log, transmission_probs_1m_log,theoretical_throughputs

# Path to the CSV file
csv_path = '6Data.csv'
T = 2  # Example value for T
sigma = 1  # Example value for sigma

# Reading the CSV file and creating data objects
data_list, adj_matrices, transmission_probs,transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs = read_csv_and_create_data(csv_path, T, sigma)
print(data_list[0])

# Splitting data into training, validation, and test sets
def split_data(dataset, adj_matrices, transmission_probs,transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs, test_size=0.2, val_size=0.1):
    train_val_data, test_data, train_val_adj, test_adj, train_val_trans,test_trans,train_val_trans_log,test_trans_log,train_val_trans_1m_log,test_trans_1m_log, train_val_theo, test_theo = train_test_split(
        dataset, adj_matrices, transmission_probs,transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs, test_size=test_size, random_state=42)
    train_data, val_data, train_adj, val_adj, train_trans, val_trans, train_trans_log, val_trans_log, train_trans_1m_log, val_trans_1m_log, train_theo, val_theo = train_test_split(
        train_val_data, train_val_adj, train_val_trans,train_val_trans_log,train_val_trans_1m_log, train_val_theo, test_size=val_size / (1 - test_size), random_state=42)
    return train_data, val_data, test_data, train_adj, val_adj, test_adj, train_trans, val_trans, test_trans,train_trans_log, val_trans_log, test_trans_log,train_trans_1m_log, val_trans_1m_log, test_trans_1m_log, train_theo, val_theo, test_theo

train_data, val_data, test_data, train_adj, val_adj, test_adj, train_trans, val_trans, test_trans,train_trans_log, val_trans_log, test_trans_log,train_trans_1m_log, val_trans_1m_log, test_trans_1m_log, train_theo, val_theo, test_theo = split_data(
    data_list, adj_matrices, transmission_probs,transmission_probs_log, transmission_probs_1m_log, theoretical_throughputs)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print(len(test_data))


# Define model, optimizer, and loss function
input_dim = 3  # Adjusted for the remaining features (transmission_prob and theoretical throughput)
hidden_dims = [64, 128, 256, 128, 64]  # Increased number of hidden layers and dimensions
output_dim = 1
model = GCN(input_dim, hidden_dims, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Learning rate scheduler
criterion = torch.nn.MSELoss()

# Evaluation function
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
            target = data.y  # Ensure target shape matches prediction shape
            theoretical_throughput = data.x[:, 2].view(-1, 1)  # Extract theoretical throughput from input features
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

# Training loop with gradient clipping
best_val_loss = float('inf')

for epoch in range(230):  # Increased the number of epochs
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_loss / len(train_loader.dataset)

    # Evaluate on validation set
    val_loss, val_error, val_theoretical_loss, val_theoretical_error, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)  # Update learning rate based on validation loss

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), (str(num_nodes)+ 'best_model.pt'))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Validation Error: {val_error:.4f}')

# Load the best model for evaluation
model.load_state_dict(torch.load(str(num_nodes) +'best_model.pt'))

# Evaluation on the training set
train_loss, train_error, train_theoretical_loss, train_theoretical_error, train_pred, train_actual, train_theoretical = evaluate(
    model, train_loader, criterion)
print(f'Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f}')

# Evaluation on the test set
test_loss, test_error, test_theoretical_loss, test_theoretical_error, test_pred, test_actual, test_theoretical = evaluate(
    model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Error: {test_error:.4f}')

# Plot comparison of exact, predicted, and theoretical saturation throughput
plt.figure(figsize=(10, 6))
plt.plot(test_actual.flatten(), label='Actual Saturation Throughput')
plt.plot(test_pred.flatten(), label='Predicted Saturation Throughput', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Saturation Throughput')
plt.legend()
plt.title('Comparison of Actual, Predicted, and Theoretical Saturation Throughput')
plt.show()

combined_results = list(zip(test_pred.flatten(), test_actual.flatten()))

# Select 10 random samples
random_samples = random.sample(combined_results, 10)

print("\nResults for 10 Random Samples:")
print("Sample\tPredicted\tActual")
for idx, (pred, actual) in enumerate(random_samples, 1):
    print(f"{idx}\t{pred:.4f}\t\t{actual:.4f}")