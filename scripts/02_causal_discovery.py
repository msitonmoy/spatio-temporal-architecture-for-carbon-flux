import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent

def load_secure_training_data():
    train_npy_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.npy"
    print(f"Fetching strictly isolated training data from: {train_npy_path}")
    
    # 1. Load the 3D NumPy array (Days, 255, 4)
    train_tensor = np.load(train_npy_path)
    
    # 2. Extract Carbon feature (index 0) across all days and nodes -> (Days, 255)
    carbon_data = train_tensor[:, :, 0].astype(np.float64)
    
    # 3. Generate deterministic node names matching 01_data_isolation order
    sectors = ['power_mt', 'transport_mt', 'aviation_mt', 'industry_mt', 'residential_mt']
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
        "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
        "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
        "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ]
    
    var_names = [f"{state}_{sector}" for state in sorted(states) for sector in sectors]
    
    # 4. Initialize Tigramite DataFrame
    dataframe = pp.DataFrame(
        carbon_data, 
        var_names=var_names
    )
    
    return dataframe, var_names, carbon_data.shape

if __name__ == "__main__":
    tigramite_df, variable_names, raw_shape = load_secure_training_data()
    
    print(f"\n[SUCCESS] Loaded {raw_shape[0]} days across {raw_shape[1]} carbon nodes.")
    print("Initializing PCMCI+ Engine...")
    
    # 1. Define the conditional independence test (Partial Correlation)
    parcorr = ParCorr(significance='analytic')
    
    # 2. Initialize the PCMCI object
    pcmci = PCMCI(
        dataframe=tigramite_df, 
        cond_ind_test=parcorr,
        verbosity=0
    )
    
    # 3. Execute PCMCI+
    print("Executing causal discovery across 255 nodes (tau_min=1, tau_max=2)...")
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=2, pc_alpha=0.05)
    
    # Ensure directory exists
    output_dir = BASE_DIR / "outputs" / "causal_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "pcmci_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    print("\n[SUCCESS] Phase 2 Complete: Causal Adjacency Matrix Generated!")
    print(f"-> Saved causal map to: {output_path}")