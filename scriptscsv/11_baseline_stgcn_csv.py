import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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

# Dataset Loading

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

def prepare_dataloaders(window_size=14, batch_size=32):
    train_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.csv"
    val_path = BASE_DIR / "data" / "processed" / "validation" / "validation_tensor.csv"
    
    train_df = pd.read_csv(train_path, index_col=0)
    val_df = pd.read_csv(val_path, index_col=0)
    
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    
    scaler_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "data_scaler_baseline.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    train_dataset = CarbonDataset(train_scaled, window_size)
    val_dataset = CarbonDataset(val_scaled, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

#Adjacency Matrix

def load_adj_matrix():
    pkl_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "pcmci_results.pkl"
    with open(pkl_path, "rb") as f:
        p_matrix = pickle.load(f)['p_matrix']
    significant_links = p_matrix[:, :, 1:] < 0.05
    adj_2d = significant_links.any(axis=2).astype(float)
    np.fill_diagonal(adj_2d, 1.0)
    return torch.tensor(adj_2d, dtype=torch.float32)

#Baseline STGCN Architecture

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
        
        # New: LayerNorm stabilizes the spatial output before passing it to the GRU
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, window_size, num_nodes = x.size()
        
        x_reshaped = x.view(batch_size * window_size, num_nodes, 1)
        ax = torch.matmul(self.adj, x_reshaped)
        spatial_out = torch.relu(torch.matmul(ax, self.gcn_weight))
        
        # Apply normalization to prevent gradient explosion
        spatial_out = self.norm(spatial_out)

        spatial_out = spatial_out.view(batch_size, window_size, num_nodes, self.hidden_dim)
        temporal_in = spatial_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, window_size, self.hidden_dim)
        _, hidden_state = self.gru(temporal_in)
        hidden_state = hidden_state.squeeze(0).view(batch_size, num_nodes, self.hidden_dim)
        
        return self.fc(hidden_state).squeeze(-1)

#Training Pipeline 

def train_csv_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Baseline on Device: {device}")
    
    seeds = [42, 100, 2026, 333, 777]
    epochs = 500
    patience = 15
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"[INFO] Initializing CSV Baseline Training for SEED: {seed}")
        print(f"{'='*50}")
        
        set_seed(seed)
        
        train_loader, val_loader, _ = prepare_dataloaders(window_size=14, batch_size=32)
        adj_matrix = load_adj_matrix()

        model = BaselineSTGCN(adj_matrix=adj_matrix, num_nodes=255, window_size=14).to(device)
        
        # New: Huber Loss for robust training, standard MSE for pure validation
        train_criterion = nn.SmoothL1Loss()
        val_criterion = nn.MSELoss()
        
        # New: Lowered Learning Rate to 3e-4 to stop initial shocks
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        save_path = BASE_DIR / "outputs" / "baseline_seed_csv" / f"best_baseline_csv_seed_{seed}.pth"
        
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
    train_csv_baseline()