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

arch_module = load_script_module("hybrid_arch", BASE_DIR / "scripts" / "05_hybrid_architecture.py")

def generate_weather_spatial_xai():

    MODEL_VERSION = "v1" # Toggle between "v1" (Static) and "v2" (Learned)
    SEED_TO_VISUALIZE = 42
    
    print(f"\n[INFO] Extracting Spatial Map for Weather Model {MODEL_VERSION.upper()} (Seed {SEED_TO_VISUALIZE})")
    
    # 1. Load the base PCMCI+ Adjacency Matrix
    adj_matrix = arch_module.load_adj_matrix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODEL_VERSION == "v1":
        # Model v1 uses the rigid matrix. No weights to extract.
        final_spatial_map = adj_matrix.numpy()
        title_text = "Static Causal Map (Weather Dataset - Rigid Model V1)"
        
    elif MODEL_VERSION == "v2":
        # 2. Initialize Model v2 and Load Weights
        model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
        
        # Ensure this matches your 06_train_model.py output naming for weather models
        model_path = BASE_DIR / "outputs" / "seed_npy" / f"best_carbon_model_{MODEL_VERSION}_weather_seed_{SEED_TO_VISUALIZE}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # 3. Extract the Learned Edge Weights
        with torch.no_grad():
            raw_edge_weights = model.gcn1.edge_weights.cpu()
            learned_gates = torch.sigmoid(raw_edge_weights).numpy()
            static_neighbor_map = model.gcn1.neighbor_map.cpu().numpy()
            final_spatial_map = learned_gates * static_neighbor_map
            
        title_text = "Learned Spatial Edge Weights (Weather Dataset - Gated Model V2)"

    # 4. Plot the Spatial XAI Heatmap
    plt.figure(figsize=(12, 10))
    mask = final_spatial_map == 0
    # Using 'rocket' to visually distinguish Weather outputs from CSV ('mako')
    sns.heatmap(final_spatial_map, mask=mask, cmap="rocket", annot=False, 
                cbar_kws={'label': 'Connection Strength'})
    
    plt.title(title_text, pad=15)
    plt.xlabel("Target Node (Region)", labelpad=10)
    plt.ylabel("Source Node (Region)", labelpad=10)
    
    output_dir = BASE_DIR / "outputs" / "xai_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"xai_spatial_heatmap_weather_{MODEL_VERSION}_seed_{SEED_TO_VISUALIZE}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Spatial Heatmap saved to: {save_path}")

if __name__ == "__main__":
    generate_weather_spatial_xai()