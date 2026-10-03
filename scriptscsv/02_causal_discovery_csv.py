import pandas as pd
import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent

def load_secure_training_data():
    train_csv_path = BASE_DIR / "data" / "processed" / "train" / "train_tensor.csv"
    print(f"Fetching strictly isolated training data from: {train_csv_path}")
    
    train_df = pd.read_csv(train_csv_path, index_col=0)
    var_names = train_df.columns.tolist()
    
    # Initialize Tigramite
    dataframe = pp.DataFrame(
        train_df.values, 
        var_names=var_names
    )
    
    # Return the dataframe and the raw Pandas shape for our print statement
    return dataframe, var_names, train_df.shape

if __name__ == "__main__":
    tigramite_df, variable_names, raw_shape = load_secure_training_data()
    
    print(f"\n[SUCCESS] Loaded {raw_shape[0]} days and {raw_shape[1]} variables.")
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
    # tau_min=1 and tau_max=2 means we are testing if emissions on Day 1 cause spikes on Day 2 or Day 3
    print("Executing causal discovery across 255 nodes (this may take a few minutes)...")
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=2, pc_alpha=0.05)
    
    output_path = BASE_DIR / "outputs" / "causal_matrices_csv" / "pcmci_results.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    print("\n[SUCCESS] Phase 2 Complete: Causal Adjacency Matrix Generated!")
    print(f"-> Saved causal map to: outputs/causal_matrices_csv/{output_path.name}")