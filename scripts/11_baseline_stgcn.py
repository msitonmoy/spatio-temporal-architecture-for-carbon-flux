import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

#Directory and Seed Setup

BASE_DIR = Path(__file__).resolve().parent.parent

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Dataset loading for 4 features

class CarbonDataset(Dataset):
    def __init__(self, data_array, window_size=14):
        self.data = data_array
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        # X: (14, 255, 4)
        x = self.data[idx : idx + self.window_size]
        # Y: (255,) - Only Carbon (Index 0)
        y = self.data[idx + self.window_size, :, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_dataloaders(window_size=14, batch_size=32):
    train_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.npy"
    val_path = BASE_DIR / "data" / "processed" / "validation" / "validation_tensor.npy"
    
    train_tensor = np.load(train_path)
    val_tensor = np.load(val_path)
    
    train_days, nodes, features = train_tensor.shape
    val_days = val_tensor.shape[0]
    
    # 1. Scale Carbon
    carbon_scaler = RobustScaler()
    train_carbon = carbon_scaler.fit_transform(train_tensor[:, :, 0])
    val_carbon = carbon_scaler.transform(val_tensor[:, :, 0])
    
    scaler_path = BASE_DIR / "outputs" / "causal_matrices" / "data_scaler_baseline.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(carbon_scaler, f)
        
    # 2. Scale Weather
    weather_scaler = RobustScaler()
    train_weather_2d = weather_scaler.fit_transform(train_tensor[:, :, 1:].reshape(train_days, nodes * 3))
    val_weather_2d = weather_scaler.transform(val_tensor[:, :, 1:].reshape(val_days, nodes * 3))
    
    train_weather = train_weather_2d.reshape(train_days, nodes, 3)
    val_weather = val_weather_2d.reshape(val_days, nodes, 3)
    
    # 3. Recombine
    train_scaled = np.concatenate([train_carbon[:, :, np.newaxis], train_weather], axis=2)
    val_scaled = np.concatenate([val_carbon[:, :, np.newaxis], val_weather], axis=2)
        
    train_dataset = CarbonDataset(train_scaled, window_size)
    val_dataset = CarbonDataset(val_scaled, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, carbon_scaler

#Adjacency Matrix

def load_adj_matrix():
    pkl_path = BASE_DIR / "outputs" / "causal_matrices" / "pcmci_results.pkl"
    with open(pkl_path, "rb") as f:
        p_matrix = pickle.load(f)['p_matrix']
    significant_links = p_matrix[:, :, 1:] < 0.05
    adj_2d = significant_links.any(axis=2).astype(float)
    np.fill_diagonal(adj_2d, 1.0)
    return torch.tensor(adj_2d, dtype=torch.float32)

#Baseline STGCN Architecture

class BaselineSTGCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes=255, window_size=14, in_features=4, hidden_dim=64):
        super(BaselineSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Normalized Adjacency
        A = adj_matrix.clone().detach().to(torch.float32)
        A = A + torch.eye(num_nodes)
        deg = A.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_A = torch.diag(deg_inv) @ A
        self.adj = nn.Parameter(norm_A, requires_grad=False)

        # Map 4 features to hidden_dim
        self.linear = nn.Linear(in_features, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, window_size, num_nodes, num_features = x.size()
        
        # Spatial Pass using multi-feature linear mapping
        support = self.linear(x) 
        spatial_out = torch.einsum('ij,bsjk->bsik', self.adj, support)
        spatial_out = torch.relu(spatial_out)
        
        # Stabilization
        spatial_out = self.norm(spatial_out)

        # Temporal Pass
        temporal_in = spatial_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, window_size, self.hidden_dim)
        _, hidden_state = self.gru(temporal_in)
        hidden_state = hidden_state.squeeze(0).view(batch_size, num_nodes, self.hidden_dim)
        
        return self.fc(hidden_state).squeeze(-1)

#Training Pipeline (500 epoch, 15 patience)

def train_npy_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running NPY Baseline on Device: {device}")
    
    seeds = [42, 100, 2026, 333, 777]
    epochs = 500
    patience = 15
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"[INFO] Initializing NPY Baseline Training for SEED: {seed}")
        print(f"{'='*50}")
        
        set_seed(seed)
        
        train_loader, val_loader, _ = prepare_dataloaders(window_size=14, batch_size=32)
        adj_matrix = load_adj_matrix()

        model = BaselineSTGCN(adj_matrix=adj_matrix, num_nodes=255, window_size=14, in_features=4).to(device)
        
        train_criterion = nn.SmoothL1Loss()
        val_criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        save_path = BASE_DIR / "outputs" / "baseline_seed_npy" / f"best_baseline_npy_seed_{seed}.pth"
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = train_criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = val_criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Seed {seed} | Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.5f}")
                break
        
        print(f"[SUCCESS] Seed {seed} completed. Best weights saved to {save_path.name}")

if __name__ == "__main__":
    train_npy_baseline()