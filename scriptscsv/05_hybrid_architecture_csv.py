import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from mamba_ssm import Mamba

BASE_DIR = Path(__file__).resolve().parent.parent

# --- 1. Load the Locked Spatial Map ---
def load_adj_matrix():
    pkl_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "pcmci_results.pkl"
    with open(pkl_path, "rb") as f:
        p_matrix = pickle.load(f)['p_matrix']
    significant_links = p_matrix[:, :, 1:] < 0.05
    adj_2d = significant_links.any(axis=2).astype(float)
    
    np.fill_diagonal(adj_2d, 1.0)
    return torch.tensor(adj_2d, dtype=torch.float32)

# --- 2. Model v1: Rigid Spatial Layer ---
class GraphConvolution(nn.Module):
    """
    Standard GCN layer mapping spatial relationships between nodes.
    Math: Output = Adjacency_Matrix @ Input_Features @ Weights
    """
    def __init__(self, in_features, out_features, adj_matrix):
        super(GraphConvolution, self).__init__()
        self.register_buffer('adj_matrix', adj_matrix)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, text_features):
        support = torch.matmul(text_features, self.weight)
        output = torch.einsum('ij,bsjk->bsik', self.adj_matrix, support)
        return output + self.bias

# --- 3. Model v2: Gated Spatial Layer (CSV Version) ---
class GatedCausalGCNLayer(nn.Module):
    """
    Gated GCN layer with PROTECTED SELF-LOOPS.
    Updated to use nn.Linear to perfectly match the saved CSV v2 state_dict.
    """
    def __init__(self, in_features, out_features, adj_matrix):
        super().__init__()
        num_nodes = adj_matrix.shape[0]
        
        identity = torch.eye(num_nodes)
        self.register_buffer('self_loops', identity)                               
        self.register_buffer('neighbor_map', torch.clamp(adj_matrix - identity, min=0.0)) 
        
        # Using nn.Linear to match the saved keys: "linear.weight" and "linear.bias"
        self.linear = nn.Linear(in_features, out_features)
        
        self.edge_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, x):
        h = self.linear(x)
        
        gated_neighbors = self.neighbor_map * torch.sigmoid(self.edge_weights)
        dynamic_adj = gated_neighbors + self.self_loops
        
        out = torch.einsum('vw,btwf->btvf', dynamic_adj, h)
        return torch.relu(out)

# --- 4. The Hybrid Architecture ---
class SpatioTemporalCarbonModel(nn.Module):
    def __init__(self, adj_matrix, num_nodes=255, window_size=14):
        super().__init__()
        self.num_nodes = num_nodes
        
        # --- MODEL V1 (Rigid Baseline) ---
        #self.gcn1 = GraphConvolution(in_features=1, out_features=8, adj_matrix=adj_matrix)
        #self.gcn2 = GraphConvolution(in_features=8, out_features=16, adj_matrix=adj_matrix)
        
        # --- MODEL V2 (Gated Learnable) ---
        self.gcn1 = GatedCausalGCNLayer(in_features=1, out_features=8, adj_matrix=adj_matrix)
        self.gcn2 = GatedCausalGCNLayer(in_features=8, out_features=16, adj_matrix=adj_matrix)
        
        # =======================================================
        
        self.temporal_dim = self.num_nodes * 16
        
        self.attention = nn.MultiheadAttention(embed_dim=self.temporal_dim, num_heads=4, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(self.temporal_dim)
        self.temporal_memory = Mamba(
            d_model=self.temporal_dim, 
            d_state=16,                
            d_conv=4,                  
            expand=2,
            device='cuda'                  
        )
        
        self.predictor = nn.Linear(self.temporal_dim, self.num_nodes)

    def forward(self, x):
        # Add feature dimension: (Batch, 14, 255) -> (Batch, 14, 255, 1)
        x = x.unsqueeze(-1)
        
        # Because we standardized the __init__, the forward pass works for BOTH models automatically
        x_gcn = torch.relu(self.gcn1(x))
        x_gcn = torch.relu(self.gcn2(x_gcn))
        
        batch_size, seq_len, num_nodes, out_features = x_gcn.shape
        x_flat = x_gcn.reshape(batch_size, seq_len, -1) 
        
        attn_out, attn_weights = self.attention(x_flat, x_flat, x_flat)
        
        attn_normalized = self.layer_norm(attn_out)
        memory_out = self.temporal_memory(attn_normalized)
        
        seq_out = memory_out[:, -1, :]
        predictions = self.predictor(seq_out)
        
        return predictions, attn_weights

if __name__ == "__main__":
    adj = load_adj_matrix()
    model = SpatioTemporalCarbonModel(adj_matrix=adj).to('cuda')
    dummy_input = torch.randn(32, 14, 255).to('cuda')
    predictions, _ = model(dummy_input)
    
    print("\n[SUCCESS] Spatio-Temporal Architecture Compiled!")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {predictions.shape}")