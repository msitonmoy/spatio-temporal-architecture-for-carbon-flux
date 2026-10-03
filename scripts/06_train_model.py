import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
import importlib.util
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

def load_script_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

data_module = load_script_module("data_loader", BASE_DIR / "scripts" / "03_data_loader.py")
arch_module = load_script_module("hybrid_arch", BASE_DIR / "scripts" / "05_hybrid_architecture.py")

# 1. The Seed Locker
def set_fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(seed, epochs=300):
    # Lock the random state for this specific run
    set_fixed_seed(seed)
    
    train_loader, val_loader, scaler = data_module.prepare_dataloaders(window_size=14, batch_size=32)
    adj_matrix = arch_module.load_adj_matrix()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    criterion = nn.HuberLoss()
    
    output_dir = BASE_DIR / "outputs" / "seed_npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Dynamically name the output files based on the current seed
    save_path = output_dir / f"best_carbon_model_v2_weather_seed_{seed}.pth"
    log_path = output_dir / f"training_log_v2_weather_seed_{seed}.txt"
    
    best_val_loss = float('inf')
    patience = 15
    stagnant_epochs = 0 
    
    print(f"\n{'='*50}\nStarting Run for SEED: {seed}\n{'='*50}")
    with open(log_path, "w") as f:
        f.write(f"Starting Training Loop - Model v1 + Weather (Seed {seed})...\n")
    
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_preds, _ = model(val_x)
                loss = criterion(val_preds, val_y)
                val_loss += loss.item() * val_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    # 3. The Ensemble Loop
    ensemble_seeds = [42, 100, 2026, 333, 777]
    for current_seed in ensemble_seeds:
        train_model(seed=current_seed, epochs=300)
    
    print("\n[SUCCESS] 5-Seed Ensemble Training Complete!")