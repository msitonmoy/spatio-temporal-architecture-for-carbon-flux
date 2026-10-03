import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from pathlib import Path
import importlib.util
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Helper function to import previous scripts ---
def load_script_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

data_module = load_script_module("data_loader", BASE_DIR / "scriptscsv" / "03_data_loader_csv.py")
arch_module = load_script_module("hybrid_arch", BASE_DIR / "scriptscsv" / "05_hybrid_architecture_csv.py")

def set_seed(seed):
    """Locks all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(seed, epochs=100):
    # 1. Lock the seed
    set_seed(seed)
    print(f"\n======================================")
    print(f"   STARTING ENSEMBLE TRAINING: SEED {seed}")
    print(f"======================================")
    
    # 2. Initialize Data and Architecture
    train_loader, val_loader, scaler = data_module.prepare_dataloaders(window_size=14, batch_size=32)
    adj_matrix = arch_module.load_adj_matrix()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
    
    # 3. Define Optimizer, Loss Function, and Tracking Variables
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    criterion = nn.HuberLoss()
    
    best_val_loss = float('inf')
    
    # --- FIX: Dynamic Save Paths and Directory Creation ---
    output_dir = BASE_DIR / "outputs" / "seed_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"best_csv_model_seed_v2_{seed}.pth"
    log_path = output_dir / f"training_log_csv_seed_v2_{seed}.txt"
    
    patience = 15
    stagnant_epochs = 0 
    
    with open(log_path, "w") as f:
        f.write(f"Starting Training Loop for Seed {seed} (Max Epochs: {epochs})...\n")
    
    # 4. The Main Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_preds, _ = model(val_x)
                loss = criterion(val_preds, val_y)
                val_loss += loss.item() * val_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        log_msg = f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(log_msg)
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
            save_msg = f"  -> Model improved! Saved to {save_path.name}"
            print(save_msg)
            with open(log_path, "a") as f:
                f.write(save_msg + "\n")
                
            stagnant_epochs = 0  
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                stop_msg = f"\n[INFO] Early stopping triggered at epoch {epoch+1} to prevent overfitting!"
                print(stop_msg)
                with open(log_path, "a") as f:
                    f.write(stop_msg + "\n")
                break

    print(f"\n[SUCCESS] Seed {seed} Training Complete!")

if __name__ == "__main__":
    # --- FIX: Execute the 5-Seed Ensemble ---
    ensemble_seeds = [42, 100, 2026, 333, 777]
    for seed in ensemble_seeds:
        train_model(seed=seed, epochs=300)