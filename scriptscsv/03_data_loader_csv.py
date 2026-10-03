import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent

class CarbonDataset(Dataset):
    def __init__(self, data_array, window_size=14):
        self.data = data_array
        self.window_size = window_size
        
    def __len__(self):
        # Total days minus the 14-day window
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        # X: 14 days of history
        x = self.data[idx : idx + self.window_size]
        # y: The 15th day (what we are predicting)
        y = self.data[idx + self.window_size]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_dataloaders(window_size=14, batch_size=32):
    train_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.csv"
    val_path = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.csv"
    
    print(f"Loading files to create {window_size}-day sliding windows...")
    train_df = pd.read_csv(train_path, index_col=0)
    val_df = pd.read_csv(val_path, index_col=0)
    
    # Neural Networks require normalized data (mean=0, variance=1)
    # We strictly fit the scaler ONLY on the training data to prevent leakage!
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    
    # Save the scaler so we can un-normalize predictions later
    scaler_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "data_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # Create the PyTorch Datasets
    train_dataset = CarbonDataset(train_scaled, window_size)
    val_dataset = CarbonDataset(val_scaled, window_size)
    
    # Wrap them in DataLoaders (this feeds the data to the GPU in batches of 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n[SUCCESS] PyTorch DataLoaders Initialized!")
    print(f"Training Batches: {len(train_loader)} (Size: {batch_size} samples each)")
    print(f"Validation Batches: {len(val_loader)} (Size: {batch_size} samples each)")
    print(f"-> Saved data scaler to: {scaler_path.name}")
    
    return train_loader, val_loader, scaler

if __name__ == "__main__":
    train_loader, val_loader, scaler = prepare_dataloaders(window_size=14)