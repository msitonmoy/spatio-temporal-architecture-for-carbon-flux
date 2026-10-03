import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import importlib.util
import sys
import json
from pathlib import Path
from PIL import Image

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

BASE_DIR = Path(__file__).resolve().parent.parent

labels_path = BASE_DIR / "data" / "raw" / "node_labels.json"
with open(labels_path, "r") as f:
    node_names = json.load(f)

# Pointing to scriptscsv for CSV architecture
data_loaders = load_module("data_loaders", BASE_DIR / "scriptscsv" / "03_data_loader_csv.py")
hybrid_arch = load_module("hybrid_arch", BASE_DIR / "scriptscsv" / "05_hybrid_architecture_csv.py")

def generate_dynamic_xai_csv():
    
    MODEL_VERSION = "v2" 
    SEED_TO_VISUALIZE = 42
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _ = data_loaders.prepare_dataloaders(window_size=14, batch_size=1)

    if MODEL_VERSION == "v1":
        model_desc = "Rigid Baseline V1"
        filename = f"best_csv_model_seed_{SEED_TO_VISUALIZE}.pth"
    elif MODEL_VERSION == "v2":
        model_desc = "Gated Model V2"
        filename = f"best_csv_model_seed_v2_{SEED_TO_VISUALIZE}.pth"
        
    model_path = BASE_DIR / "outputs" / "seed_csv" / filename
    
    print(f"\n[INFO] Generating dynamic XAI frames for CSV Dataset ({model_desc}) using {filename}...")

    # 1. Load adj_matrix onto CPU
    adj_matrix = hybrid_arch.load_adj_matrix() 
    
    # 2. Initialize the model entirely on CPU matching your exact __init__
    model = hybrid_arch.SpatioTemporalCarbonModel(
        adj_matrix=adj_matrix, 
        num_nodes=255, 
        window_size=14
    )
    
    # 3. Move the assembled model (and its internal matrices) to the GPU
    model = model.to(device)
    
    # 4. Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    frames_dir = BASE_DIR / "outputs" / "xai_visualizations" / f"frames_csv_{MODEL_VERSION}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_files = []

    for day_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        inputs.requires_grad_()
        
        model.zero_grad()
        predictions, attn_weights = model(inputs)
        
        actual_day_15 = targets.squeeze().cpu().numpy()
        target_idx = int(np.argmax(actual_day_15))
        target_name = node_names[target_idx]
        
        mean_temporal_attn = attn_weights[0].mean(dim=0).detach().cpu().numpy()
        anchor_day_idx = int(np.argmax(mean_temporal_attn)) 
        anchor_day_num = anchor_day_idx + 1
        
        spike_score = predictions[0, target_idx]
        spike_score.backward()
        
        saliency = inputs.grad.abs().squeeze().cpu().numpy()
        anchor_saliency = saliency[anchor_day_idx]
        
        if anchor_saliency.ndim > 1:
            node_importance = anchor_saliency.sum(axis=-1)
        else:
            node_importance = anchor_saliency
            
        node_importance[target_idx] = 0.0 
        
        top_15_indices = np.argsort(node_importance)[-15:]
        top_15_scores = node_importance[top_15_indices]
        driver_labels = [node_names[i] for i in top_15_indices]
        
        plt.figure(figsize=(11, 6))
        sns.barplot(x=top_15_scores, y=driver_labels, palette="magma")
        
        plt.title(
            f"Test Batch {day_idx + 1}: Drivers for Peak [{target_name}]\n"
            f"CSV Dataset ({model_desc}) - Attended Anchor: Day {anchor_day_num} of 14",
            fontsize=12, pad=12
        )
        plt.xlabel("Gradient Attribution Score (Influence on Spike)", fontsize=10)
        plt.ylabel(f"External Driver Node (at Day {anchor_day_num})", fontsize=10)
        plt.xlim(0, max(top_15_scores) * 1.2 if max(top_15_scores) > 0 else 1)
        plt.tight_layout()
        
        frame_path = frames_dir / f"frame_{day_idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        image_files.append(frame_path)


    if image_files:
        frames = [Image.open(img) for img in image_files]
        animation_path = BASE_DIR / "outputs" / "xai_visualizations" / f"dynamic_drivers_csv_{MODEL_VERSION}.gif"
        frames[0].save(
            animation_path,
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0
        )
        print(f"[SUCCESS] Animation saved to: {animation_path}")

if __name__ == "__main__":
    generate_dynamic_xai_csv()