import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def build_adjacency_matrix(alpha=0.05):
    pkl_path = BASE_DIR / "outputs" / "causal_matrices" / "pcmci_results.pkl"
    print(f"Loading PCMCI+ results from: {pkl_path.name}")
    
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
        
    p_matrix = results['p_matrix']
    significant_links = p_matrix[:, :, 1:] < alpha
    adj_2d = significant_links.any(axis=2).astype(float)
    np.fill_diagonal(adj_2d, 1.0)
    
    return torch.tensor(adj_2d, dtype=torch.float32)

class CausalGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super().__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # x shape: (Batch, Time, Nodes, Features)
        # 1. Apply the learnable weight transformation
        h = self.linear(x)

        # 2. Route the data using Einstein Summation (einsum)
        # v, w = nodes; b = batch; t = time; f = features
        # This explicitly multiplies the node features by the strict causal map
        out = torch.einsum('vw,btwf->btvf', self.adj_matrix, h)
        
        return torch.relu(out)

if __name__ == "__main__":
    adj = build_adjacency_matrix()
    print(f"\n[SUCCESS] Causal Adjacency Matrix Extracted!")
    print(f"Matrix Shape: {adj.shape} (255 nodes x 255 nodes)")
    print(f"Total Causal Edges Enforced: {int(adj.sum().item())}")
    
    # Initialized specifically for the 4 multivariate features
    dummy_v1 = CausalGCNLayer(in_features=4, out_features=16, adj_matrix=adj)
    print("\n[SUCCESS] Model v1 (Rigid Mask) Layer Initialized and Ready!")