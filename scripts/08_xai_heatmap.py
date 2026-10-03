import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_script_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import the multivariate data loader and architecture
data_module = load_script_module("data_loader", BASE_DIR / "scripts" / "03_data_loader.py")
arch_module = load_script_module("hybrid_arch", BASE_DIR / "scripts" / "05_hybrid_architecture.py")

def generate_weather_xai_heatmap():

    MODEL_VERSION = "v2" # Swap to "v2" for the Gated Architecture
    SEED_TO_VISUALIZE = 42 # Pick one of your 5 seeds (42, 100, 2026, 333, 777)

    print(f"\n[INFO] Generating XAI Heatmap for Weather Model {MODEL_VERSION.upper()} (Seed {SEED_TO_VISUALIZE})")
    
    # 1. Load Data (Ensuring it pulls from test_HIDDEN if set in data_loader)
    _, test_loader, _ = data_module.prepare_dataloaders(window_size=14, batch_size=32)
    adj_matrix = arch_module.load_adj_matrix()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize Model and Load Weights
    # If your architecture accepts model_version, use it. Otherwise, rely on your manual swap in 05.
    model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
    
    # Adjust this path string to exactly match how you saved your weather models!
    model_path = BASE_DIR / "outputs" / "seed_npy" / f"best_carbon_model_{MODEL_VERSION}_weather_seed_{SEED_TO_VISUALIZE}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Run a single batch to extract attention
    with torch.no_grad():
        # Get one batch of test data
        test_x, _ = next(iter(test_loader))
        test_x = test_x.to(device)
        
        # Extract the attention weights
        _, attn_weights = model(test_x)
        
        # Average the attention weights across the entire batch
        # PyTorch attn_weights shape: (Batch, Seq_Len, Seq_Len) -> (32, 14, 14)
        avg_attention = attn_weights.mean(dim=0).cpu().numpy()
        
    # 4. Plot the XAI Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, cmap="viridis", annot=False, fmt=".3f", 
                xticklabels=[f"Day {i+1}" for i in range(14)],
                yticklabels=[f"Day {i+1}" for i in range(14)])
    
    plt.title(f"Temporal Attention Weights (Weather Dataset - Model {MODEL_VERSION.upper()})", pad=15)
    plt.xlabel("Key (Information Source Day)", labelpad=10)
    plt.ylabel("Query (Information Target Day)", labelpad=10)
    
    # 5. Save the output
    output_dir = BASE_DIR / "outputs" / "plotsnpy" / "xai_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"xai_heatmap_weather_{MODEL_VERSION}_seed_{SEED_TO_VISUALIZE}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Heatmap saved to: {save_path}")

if __name__ == "__main__":
    generate_weather_xai_heatmap()