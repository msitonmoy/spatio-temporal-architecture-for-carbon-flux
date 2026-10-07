import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
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
        y = self.data[idx + self.window_size, :, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_test_loader(window_size=14, batch_size=32):
    train_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.npy"
    test_path = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.npy"
    
    train_tensor = np.load(train_path)
    test_tensor = np.load(test_path)
    
    train_days, nodes, features = train_tensor.shape
    test_days = test_tensor.shape[0]
    
    # Fit scalers strictly on train data, then apply to test data
    carbon_scaler = RobustScaler()
    _ = carbon_scaler.fit_transform(train_tensor[:, :, 0])
    test_carbon = carbon_scaler.transform(test_tensor[:, :, 0])
    
    weather_scaler = RobustScaler()
    _ = weather_scaler.fit_transform(train_tensor[:, :, 1:].reshape(train_days, nodes * 3))
    test_weather_2d = weather_scaler.transform(test_tensor[:, :, 1:].reshape(test_days, nodes * 3))
    test_weather = test_weather_2d.reshape(test_days, nodes, 3)
    
    # Recombine test features
    test_scaled = np.concatenate([test_carbon[:, :, np.newaxis], test_weather], axis=2)
    test_dataset = CarbonDataset(test_scaled, window_size)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, carbon_scaler


def load_adj_matrix():
    pkl_path = BASE_DIR / "outputs" / "causal_matrices" / "pcmci_results.pkl"
    with open(pkl_path, "rb") as f:
        p_matrix = pickle.load(f)['p_matrix']
    significant_links = p_matrix[:, :, 1:] < 0.05
    adj_2d = significant_links.any(axis=2).astype(float)
    np.fill_diagonal(adj_2d, 1.0)
    return torch.tensor(adj_2d, dtype=torch.float32)


class BaselineSTGCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes=255, window_size=14, in_features=4, hidden_dim=64):
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

        self.linear = nn.Linear(in_features, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, window_size, num_nodes, num_features = x.size()
        
        support = self.linear(x) 
        spatial_out = torch.einsum('ij,bsjk->bsik', self.adj, support)
        spatial_out = torch.relu(spatial_out)
        
        spatial_out = self.norm(spatial_out)

        temporal_in = spatial_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, window_size, self.hidden_dim)
        _, hidden_state = self.gru(temporal_in)
        hidden_state = hidden_state.squeeze(0).view(batch_size, num_nodes, self.hidden_dim)
        
        return self.fc(hidden_state).squeeze(-1)


def evaluate_npy_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating NPY Baseline on Device: {device}\n")
    
    seeds = [42, 100, 2026, 333, 777]
    val_loader, carbon_scaler = prepare_test_loader(window_size=14, batch_size=32)
    adj_matrix = load_adj_matrix()

    rmse_scores, mae_scores, r2_scores, acc_scores = [], [], [], []
    
    for seed in seeds:
        model = BaselineSTGCN(adj_matrix=adj_matrix, num_nodes=255, window_size=14, in_features=4).to(device)
        weight_path = BASE_DIR / "outputs" / "baseline_seed_npy" / f"best_baseline_npy_seed_{seed}.pth"
        
        if not weight_path.exists():
            print(f"[ERROR] Weight file missing for seed {seed}: {weight_path}")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
                
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Inverse transform only the carbon predictions
        preds_inv = carbon_scaler.inverse_transform(all_preds)
        targets_inv = carbon_scaler.inverse_transform(all_targets)
        
        preds_flat = preds_inv.flatten()
        targets_flat = targets_inv.flatten()
        
        mae = np.mean(np.abs(preds_flat - targets_flat))
        mse = np.mean((preds_flat - targets_flat) ** 2)
        rmse = math.sqrt(mse)
        r2 = r2_score(targets_flat, preds_flat)
        
        global_mean = np.mean(targets_flat)
        error_percentage = (mae / global_mean) * 100
        accuracy = 100 - error_percentage
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        acc_scores.append(accuracy)
        
        print(f"Seed {seed:<5} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | Acc: {accuracy:.2f}%")

    if not rmse_scores: return

    print("\n" + "="*60)
    print("FINAL MANUSCRIPT METRICS (STGCN BASELINE - NPY)")
    print("="*60)
    print(f"Overall RMSE:     {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Overall MAE:      {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Overall R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Overall Accuracy: {np.mean(acc_scores):.2f}% ± {np.std(acc_scores):.2f}%")
    print("="*60)

if __name__ == "__main__":
    evaluate_npy_baseline()