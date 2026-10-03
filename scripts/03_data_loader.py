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
        # X: 14 days of history across all 255 nodes and 4 features. Shape: (14, 255, 4)
        x = self.data[idx : idx + self.window_size]
        
        # y: The 15th day across all 255 nodes, but ONLY feature index 0 (Carbon). Shape: (255,)
        y = self.data[idx + self.window_size, :, 0]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_dataloaders(window_size=14, batch_size=32):
    # If testing Model v1 vs Model v2 on the test set, change val_path to test_tensor.npy
    train_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.npy"
    val_path = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.npy"
    
    print(f"Loading native NumPy tensors to create {window_size}-day sliding windows...")
    train_tensor = np.load(train_path) # Shape: (Days, 255, 4)
    val_tensor = np.load(val_path)     # Shape: (Days, 255, 4)
    
    train_days, nodes, features = train_tensor.shape
    val_days = val_tensor.shape[0]
    
    # --- DUAL SCALING STRATEGY ---
    # 1. Scale Carbon (Index 0)
    carbon_scaler = RobustScaler()
    train_carbon = carbon_scaler.fit_transform(train_tensor[:, :, 0])
    val_carbon = carbon_scaler.transform(val_tensor[:, :, 0])
    
    # Save ONLY the carbon scaler so the evaluation script can unscale predictions back to metric tonnes
    scaler_path = BASE_DIR / "outputs" / "causal_matrices" / "data_scaler.pkl"
    # Ensure directory exists
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(carbon_scaler, f)
        
    # 2. Scale Weather (Indices 1, 2, 3)
    # We flatten the weather dimensions down to 2D for the scaler, then reshape back to 3D
    weather_scaler = RobustScaler()
    train_weather_2d = weather_scaler.fit_transform(train_tensor[:, :, 1:].reshape(train_days, nodes * 3))
    val_weather_2d = weather_scaler.transform(val_tensor[:, :, 1:].reshape(val_days, nodes * 3))
    
    train_weather = train_weather_2d.reshape(train_days, nodes, 3)
    val_weather = val_weather_2d.reshape(val_days, nodes, 3)
    
    # 3. Recombine scaled features
    # Add a dummy axis to carbon (Days, 255, 1) and concatenate with weather (Days, 255, 3)
    train_scaled = np.concatenate([train_carbon[:, :, np.newaxis], train_weather], axis=2)
    val_scaled = np.concatenate([val_carbon[:, :, np.newaxis], val_weather], axis=2)
        
    # Create the PyTorch Datasets
    train_dataset = CarbonDataset(train_scaled, window_size)
    val_dataset = CarbonDataset(val_scaled, window_size)
    
    # Wrap them in DataLoaders (this feeds the data to the GPU in batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n[SUCCESS] PyTorch DataLoaders Initialized!")
    print(f"Training Batches: {len(train_loader)} (Size: {batch_size} samples each)")
    print(f"Validation Batches: {len(val_loader)} (Size: {batch_size} samples each)")
    print(f"-> Saved CARBON data scaler to: {scaler_path.name}")
    
    return train_loader, val_loader, carbon_scaler

if __name__ == "__main__":
    train_loader, val_loader, scaler = prepare_dataloaders(window_size=14)