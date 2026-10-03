import torch
import numpy as np
import importlib.util
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Helper function to dynamically import scripts starting with numbers
def load_script_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

data_module = load_script_module("data_loader", BASE_DIR / "scripts" / "03_data_loader.py")
arch_module = load_script_module("hybrid_arch", BASE_DIR / "scripts" / "05_hybrid_architecture.py")

def evaluate_ensemble():
    # 1. Load Data and Carbon Scaler (We only need to do this once)
    _, val_loader, scaler = data_module.prepare_dataloaders(window_size=14, batch_size=32)
    adj_matrix = arch_module.load_adj_matrix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ensemble_seeds = [42, 100, 2026, 333, 777]
    
    # Dictionaries to track the metrics across all 5 runs
    metrics = {'mae': [], 'rmse': [], 'r2': [], 'acc': []}
    global_mean = 0
    
    print(f"\n[INFO] Evaluating 5-Seed Ensemble on device: {device}")
    
    for seed in ensemble_seeds:
        # Initialize a fresh model for each seed
        model = arch_module.SpatioTemporalCarbonModel(adj_matrix=adj_matrix).to(device)
        
        # Load the specific seed's weights
        model_path = BASE_DIR / "outputs" / f"best_carbon_model_v2_weather_seed_{seed}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        all_preds = []
        all_actuals = []
        
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                preds, _ = model(val_x)
                
                all_preds.append(preds.cpu().numpy())
                all_actuals.append(val_y.cpu().numpy())
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_actuals = np.concatenate(all_actuals, axis=0)
        
        # Inverse Transform back to metric tonnes
        real_preds = scaler.inverse_transform(all_preds)
        real_actuals = scaler.inverse_transform(all_actuals)
        
        # Calculate metrics for this specific seed
        mae = mean_absolute_error(real_actuals, real_preds)
        rmse = np.sqrt(mean_squared_error(real_actuals, real_preds))
        r2 = r2_score(real_actuals.flatten(), real_preds.flatten())
        
        global_mean = np.mean(real_actuals) # This will be identical across runs
        acc = 100 - ((mae / global_mean) * 100)
        
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['r2'].append(r2)
        metrics['acc'].append(acc)
        
        print(f"  -> Seed {seed:4d} | R2: {r2:.4f} | Acc: {acc:.2f}% | MAE: {mae:.2f}")

    # Calculate final averages and standard deviations (The academic gold standard)
    print("\n" + "="*55)
    print("   MODEL V2 (WEATHER) ENSEMBLE METRICS (5 SEEDS)")
    print("="*55)
    print(f"Mean Absolute Error (MAE) : {np.mean(metrics['mae']):.2f} ± {np.std(metrics['mae']):.2f} Metric Tonnes")
    print(f"Root Mean Squared (RMSE)  : {np.mean(metrics['rmse']):.2f} ± {np.std(metrics['rmse']):.2f} Metric Tonnes")
    print(f"R-squared (R2) Score      : {np.mean(metrics['r2']):.4f} ± {np.std(metrics['r2']):.4f}")
    print(f"Overall Model Accuracy    : {np.mean(metrics['acc']):.2f}% ± {np.std(metrics['acc']):.2f}%")
    print(f"Global Average Carbon     : {global_mean:.2f} Metric Tonnes")
    print("="*55)

if __name__ == "__main__":
    evaluate_ensemble()