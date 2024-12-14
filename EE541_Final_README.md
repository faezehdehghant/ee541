# EE541_Final Repository
repository for training and evaluating Graph Neural Networks (GNNs) to predict saturation throughput in CSMA networks.

---

## Overview
- **Objective**: Predict per-node throughput in a CSMA network using GNNs.
- **Approach**: Train a GCN or GAT model on CSV datasets containing adjacency matrices, node transmission probabilities, and saturation throughput.
- **Key Feature**: Integrates a theoretical throughput formula to enhance predictions.
- **Note**: for transfer learnign you should train the model with **GNN_Architecture.py** and choose the correct variables for like 8 nodes and then use the generated model for the "Train_w_Transfer_Learning.py" for 10 nodes.
---

## Repository Structure

1. **GNN_Architecture.py**
   - Defines the GNN and GAT(comment out) architecture:
     - Stacked `GCNConv` layers from PyTorch Geometric.
     - Batch normalization and ReLU activation.
     - Final fully connected layers with a sigmoid output.

2. **GNN_Model.py**
   - Data Preparation & Theoretical Analysis: Reads a CSV, extracts graph features (adjacency matrix, transmission probabilities), and calculates theoretical saturation throughput for CSMA networks.
   - Model Training: Trains a GCN using node features and graph structure to predict per-node throughput, saving the best-performing model based on validation loss.
   - Evaluation & Visualization: Evaluates the model on test data, compares actual vs. predicted throughput, and plots results for analysis.

3. **Train_w_Transfer_Learning.py** (Main Script)
   - Data Preparation and Fine-Tuning: Reads a 10-node dataset (CSV), generates graph data with features like log-transformed transmission probabilities and theoretical throughput, and fine-tunes a pre-trained GCN model using transfer learning.
   - Training and Validation: Trains the model on the 10-node data, monitors loss and validation error, and saves the best model. Optionally freezes earlier layers for transfer learning efficiency.
   - Evaluation and Visualization: Evaluates the fine-tuned model on the test set, plots actual vs. predicted throughput, and outputs performance metrics and sample results for analysis.

4. **Test_Model_prediction.py** (Optional)
   - Evaluates a trained model on a test set.
   - Prints loss/error metrics and plots predictions vs. actual throughput.

5. **theoretical_saturation_throughput_old**
   - A function for calculating Renewal theoretical throughput in CSMA networks.

---

## Quick Start

### Install Dependencies
```bash
pip install torch torchvision torch_geometric numpy pandas scikit-learn matplotlib
```

### Prepare the Dataset
Create a CSV file (e.g., `6Data.csv`) with the following columns:
- **`adj_matrix`**: Stringified adjacency matrix (e.g., `"[[0,1],[1,0]]"`).
- **`transmission_prob`**: Node transmission probabilities (e.g., `"[0.1, 0.2]"`).
- **`saturation_throughput`**: Actual node-level throughput (e.g., `"[0.08, 0.15]"`).

**Test the model to Predict new data**:
```bash
python Test_Model_prediction.py
```
- These scripts load the saved model, compute metrics, and/or output new predictions.

---

## Model Architectures

### GCN:
- Multiple `GCNConv` layers with batch normalization and ReLU.
- Final fully connected layers and a sigmoid output.

### GAT (commented out in `GNN_Model.py`):
- Swap in `GATConv` layers for attention-based message passing.
- Similar structure (batch normalization, fully connected layers, sigmoid output).

---

## Theoretical Throughput
The function `theoretical_saturation_throughput_old()` is Renewal theory approximation which is fast but in accurate it is inculded to included to compute approximatation of throughput in CSMA networks. It uses:
- Node transmission probabilities.
- Network adjacency matrix.
- Parameters **T** and **σ**.

This theoretical value is also fed into the GNN as an additional feature, which can improve prediction performance.

---
