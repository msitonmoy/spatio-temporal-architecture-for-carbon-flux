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

data_module = load_script_module("data_loader", BASE_DIR / "scriptscsv" / "03_data_loader_csv.py")
arch_module = load_script_module("hybrid_arch", BASE_DIR / "scriptscsv" / "05_hybrid_architecture_csv.py")

def generate_csv_xai_heatmap():
    MODEL_VERSION = "v2" # Swap to "v2" after you change 05_hybrid_architecture.py
    SEED_TO_VISUALIZE = 42 

    print(f"\n[INFO] Generating XAI Heatmap for CSV Model {MODEL_VERSION.upper()} (Seed {SEED_TO_VISUALIZE})")
    
    _, test_loader, _ = data_module.prepare_dataloaders(window_size=14, batch_size=32)
    adj_matrix = arch_module.load_adj_matrix()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
    
    # Path explicitly maps to the training script output we just verified
    model_path = BASE_DIR / "outputs" / "seed_csv" / f"best_csv_model_seed_v2_{SEED_TO_VISUALIZE}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        test_x, _ = next(iter(test_loader))
        test_x = test_x.to(device)
        _, attn_weights = model(test_x)
        avg_attention = attn_weights.mean(dim=0).cpu().numpy()
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, cmap="magma", annot=False, 
                xticklabels=[f"Day {i+1}" for i in range(14)],
                yticklabels=[f"Day {i+1}" for i in range(14)])
    
    plt.title(f"Temporal Attention Weights (CSV Dataset - Model {MODEL_VERSION.upper()})", pad=15)
    plt.xlabel("Key (Information Source Day)", labelpad=10)
    plt.ylabel("Query (Information Target Day)", labelpad=10)
    
    output_dir = BASE_DIR / "outputs" / "plotscsv" / "xai_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"xai_heatmap_csv_{MODEL_VERSION}_seed_{SEED_TO_VISUALIZE}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Heatmap saved to: {save_path}")

if __name__ == "__main__":
    generate_csv_xai_heatmap()