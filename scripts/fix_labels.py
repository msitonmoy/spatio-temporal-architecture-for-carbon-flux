import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
raw_data_path = BASE_DIR / "data" / "raw" / "master_carbon_emission_dataset_2023_2025.csv"
labels_path = BASE_DIR / "data" / "raw" / "node_labels.json"

# 1. Load data to extract unique states
print("Reading raw data...")
df = pd.read_csv(raw_data_path)
states = df['state'].unique().tolist()

# Sort alphabetically to match standard Graph Neural Network node generation
states.sort() 

# The 5 specific sectors
sectors = ["power_mt", "transport_mt", "aviation_mt", "industry_mt", "residential_mt"]

# 2. Generate exactly 255 combinations
node_names = []
for state in states:
    for sector in sectors:
        node_names.append(f"{state} ({sector})")

# 3. Save to JSON
labels_path.parent.mkdir(parents=True, exist_ok=True)
with open(labels_path, "w") as f:
    json.dump(node_names, f)

print(f"\n[SUCCESS] Generated exactly {len(node_names)} node labels!")
print(f"First 3 nodes: {node_names[:3]}")
print(f"Last 3 nodes: {node_names[-3:]}")