import pandas as pd
from pathlib import Path

# 1. Dynamically find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def physically_isolate_data():
    raw_file_path = BASE_DIR / "data" / "raw" / "master_carbon_emission_dataset_2023_2025.csv"
    print(f"Loading raw dataset from: {raw_file_path}")
    
    # Load and reshape
    df = pd.read_csv(raw_file_path)
    df['date'] = pd.to_datetime(df['date'])
    pivot_df = df.pivot(index='date', columns='state', 
                        values=['power_mt', 'transport_mt', 'aviation_mt', 'industry_mt', 'residential_mt'])
    pivot_df.columns = [f"{state}_{sector}" for sector, state in pivot_df.columns]
    
    # 2. Execute the 3-Way Chronological Split
    train_df = pivot_df[pivot_df.index <= '2025-03-31']
    val_df = pivot_df[(pivot_df.index > '2025-03-31') & (pivot_df.index <= '2025-07-31')]
    test_df = pivot_df[pivot_df.index > '2025-07-31']
    
    # 3. Define secure output paths
    train_out = BASE_DIR / "data" / "processed" / "train" / "train_tensor.csv"
    val_out = BASE_DIR / "data" / "processed" / "validation" / "validation_tensor.csv"
    test_out = BASE_DIR / "data" / "processed" / "test_HIDDEN" / "test_tensor.csv"
    
    # 4. Save the files
    train_df.to_csv(train_out)
    val_df.to_csv(val_out)
    test_df.to_csv(test_out)
    
    print("\n[SUCCESS] Physical Isolation Complete!")
    print(f"TRAIN      -> {train_out.name} ({len(train_df)} days)")
    print(f"VALIDATION -> {val_out.name} ({len(val_df)} days)")
    print(f"TEST       -> {test_out.name} ({len(test_df)} days) - LOCKED")

if __name__ == "__main__":
    physically_isolate_data()