import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

class CarbonDataset(Dataset):
    def __init__(self, data_array, window_size=14):
        self.data = data_array
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_test_loader(window_size=14, batch_size=32):
    val_path = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.csv"
    val_df = pd.read_csv(val_path, index_col=0)
    
    scaler_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "data_scaler_baseline.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    val_scaled = scaler.transform(val_df.values)
    val_dataset = CarbonDataset(val_scaled, window_size)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # We must return the scaler to inverse-transform the predictions later
    return val_loader, scaler


def load_adj_matrix():
    pkl_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "pcmci_results.pkl"
    with open(pkl_path, "rb") as f:
        p_matrix = pickle.load(f)['p_matrix']
    significant_links = p_matrix[:, :, 1:] < 0.05
    adj_2d = significant_links.any(axis=2).astype(float)
    np.fill_diagonal(adj_2d, 1.0)
    return torch.tensor(adj_2d, dtype=torch.float32)


class BaselineSTGCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes=255, window_size=14, hidden_dim=64):
        super(BaselineSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        A = adj_matrix.clone().detach().to(torch.float32)
        A = A + torch.eye(num_nodes)
        deg = A.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_A = torch.diag(deg_inv) @ A
        self.adj = nn.Parameter(norm_A, requires_grad=False)

        self.gcn_weight = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.gcn_weight)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, window_size, num_nodes = x.size()
        
        x_reshaped = x.view(batch_size * window_size, num_nodes, 1)
        ax = torch.matmul(self.adj, x_reshaped)
        spatial_out = torch.relu(torch.matmul(ax, self.gcn_weight))
        
        spatial_out = self.norm(spatial_out)

        spatial_out = spatial_out.view(batch_size, window_size, num_nodes, self.hidden_dim)
        temporal_in = spatial_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, window_size, self.hidden_dim)
        _, hidden_state = self.gru(temporal_in)
        hidden_state = hidden_state.squeeze(0).view(batch_size, num_nodes, self.hidden_dim)
        
        return self.fc(hidden_state).squeeze(-1)


def evaluate_csv_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating Baseline on Device: {device}\n")
    
    seeds = [42, 100, 2026, 333, 777]
    val_loader, scaler = prepare_test_loader(window_size=14, batch_size=32)
    adj_matrix = load_adj_matrix()

    rmse_scores = []
    mae_scores = []
    r2_scores = []
    acc_scores = []
    
    for seed in seeds:
        model = BaselineSTGCN(adj_matrix=adj_matrix, num_nodes=255, window_size=14).to(device)
        weight_path = BASE_DIR / "outputs" / "baseline_seed_csv" / f"best_baseline_csv_seed_{seed}.pth"
        
        if not weight_path.exists():
            print(f"[ERROR] Weight file missing for seed {seed}: {weight_path}")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Move batches to CPU memory
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
                
        # Combine batches into single matrices
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # INVERSE TRANSFORM: Map normalized predictions back to real-world carbon units
        preds_inv = scaler.inverse_transform(all_preds)
        targets_inv = scaler.inverse_transform(all_targets)
        
        # FLATTEN ARRAYS: Treat all spatio-temporal points as a single distribution
        preds_flat = preds_inv.flatten()
        targets_flat = targets_inv.flatten()
        
        # 1. Calculate Real-World MAE and RMSE globally
        mae = np.mean(np.abs(preds_flat - targets_flat))
        mse = np.mean((preds_flat - targets_flat) ** 2)
        rmse = math.sqrt(mse)
        
        # 2. Calculate Global R-Squared
        r2 = r2_score(targets_flat, preds_flat)
        
        # 3. Calculate Overall Accuracy based on the global mean
        global_mean = np.mean(targets_flat)
        error_percentage = (mae / global_mean) * 100
        accuracy = 100 - error_percentage
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        acc_scores.append(accuracy)
        
        print(f"Seed {seed:<5} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | Acc: {accuracy:.2f}%")

    if not rmse_scores:
        return

    # Final Manuscript Aggregation
    mean_rmse, std_rmse = np.mean(rmse_scores), np.std(rmse_scores)
    mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
    mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)
    mean_acc, std_acc = np.mean(acc_scores), np.std(acc_scores)

    print("\n" + "="*60)
    print("FINAL MANUSCRIPT METRICS (STGCN BASELINE - CSV)")
    print("="*60)
    print(f"Overall RMSE:     {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Overall MAE:      {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"Overall R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Overall Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    evaluate_csv_baseline()