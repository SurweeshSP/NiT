import optuna
import json
import os
import sys

# Suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)

from main import run_pipeline

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    epochs = trial.suggest_int("epochs", 2, 5)
    budget = trial.suggest_int("aug_budget_max", 3, 7)
    semantic_threshold = trial.suggest_float("semantic_threshold", 0.75, 0.90)
    backbone = trial.suggest_categorical("backbone", ["distilbert-base-uncased", "roberta-base"])
    loss_type = trial.suggest_categorical("loss_type", ["CE", "Focal", "CBL"])
    
    hyperparams = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "aug_budget_max": budget,
        "semantic_threshold": semantic_threshold,
        "backbone": backbone,
        "loss_type": loss_type
    }
    
    print(f"Trial {trial.number}: Evaluating...")
    try:
        # Run pipeline with a very small configuration to save time (just the proposed hybrid)
        # We will use baselines param to restrict to Proposed Hybrid
        ablation_config = {
            "baselines": ["Proposed Hybrid GA-GWO"],
            "use_faci": True,
            "use_pm": True,
            "use_ga": True,
            "use_gwo": True,
            "use_sem": True,
            "use_rb": True,
            "use_mlm": False
        }
        
        metrics = run_pipeline(
            seed=42,
            run_id=trial.number,
            ablation_config=ablation_config,
            output_dir=f"results/tune_{trial.number}",
            debug=False,
            hyperparams=hyperparams
        )
        
        # We want to maximize validation Macro F1
        avg_f1 = metrics.get("macro_f1", 0.0)
        return avg_f1
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

def main():
    print("Starting Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5) # Kept small for demonstration
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Macro F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    best_params = trial.params
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    print("Saved best_hyperparameters.json")

if __name__ == "__main__":
    main()
