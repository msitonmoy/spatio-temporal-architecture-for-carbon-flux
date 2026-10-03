import pandas as pd
import numpy as np
from pathlib import Path

# 1. Dynamically find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def physically_isolate_data():
    raw_file_path = BASE_DIR / "data" / "raw" / "master_multivariate_dataset_2023_2025.csv"
    
    # Load and sort chronologically and alphabetically to guarantee node order
    df = pd.read_csv(raw_file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'state'])
    
    sectors = ['power_mt', 'transport_mt', 'aviation_mt', 'industry_mt', 'residential_mt']
    states = sorted(df['state'].unique())
    dates = sorted(df['date'].unique())
    
    num_days = len(dates)
    num_nodes = len(states) * len(sectors) # 51 * 5 = 255
    num_features = 4 # [Carbon, Temperature, Wind, Precipitation]
    
    # Initialize the empty 3D tensor: (Days, 255, 4)
    dataset_tensor = np.zeros((num_days, num_nodes, num_features), dtype=np.float32)
    
    # Build the 3D tensor
    for t_idx, date in enumerate(dates):
        day_data = df[df['date'] == date]
        
        node_idx = 0
        for state in states:
            state_data = day_data[day_data['state'] == state].iloc[0]
            for sector in sectors:
                dataset_tensor[t_idx, node_idx, 0] = state_data[sector]
                dataset_tensor[t_idx, node_idx, 1] = state_data['temperature']
                dataset_tensor[t_idx, node_idx, 2] = state_data['wind_speed']
                dataset_tensor[t_idx, node_idx, 3] = state_data['precipitation']
                node_idx += 1

    # 2. Execute the 3-Way Chronological Split
    date_series = pd.Series(dates)
    train_mask = date_series <= pd.to_datetime('2025-03-31')
    val_mask = (date_series > pd.to_datetime('2025-03-31')) & (date_series <= pd.to_datetime('2025-07-31'))
    test_mask = date_series > pd.to_datetime('2025-07-31')
    
    train_tensor = dataset_tensor[train_mask]
    val_tensor = dataset_tensor[val_mask]
    test_tensor = dataset_tensor[test_mask]
    
    # 3. Define secure output paths
    train_out = BASE_DIR / "data" / "processed" / "train" / "train_tensor.npy"
    val_out = BASE_DIR / "data" / "processed" / "validation" / "validation_tensor.npy"
    test_out = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.npy"
    
    # 4. Save the files as native NumPy binaries
    np.save(train_out, train_tensor)
    np.save(val_out, val_tensor)
    np.save(test_out, test_tensor)
    
    print("\n[SUCCESS] Physical Isolation Complete! (Native 3D NumPy Arrays)")
    print(f"TRAIN      -> {train_out.name} Shape: {train_tensor.shape}")
    print(f"VALIDATION -> {val_out.name} Shape: {val_tensor.shape}")
    print(f"TEST       -> {test_out.name} Shape: {test_tensor.shape} - LOCKED")

if __name__ == "__main__":
    physically_isolate_data()