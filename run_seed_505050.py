import os
import json
import gc
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

from main import run_pipeline, set_seed
from run_experiments import load_hyperparams

SEED = 505050
OUTPUT_DIR = "results/multi_seed"

def main():
    print("=" * 70)
    print(f"  Running Independent Seed {SEED}")
    print("=" * 70)

    run_dir = os.path.join(OUTPUT_DIR, f"seed_{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Copy policy memory if it doesn't exist
    pm_path = os.path.join(run_dir, "policy_memory.csv")
    if not os.path.exists(pm_path):
        import shutil
        src_pm = os.path.join(OUTPUT_DIR, "policy_memory.csv")
        if os.path.exists(src_pm):
            shutil.copy(src_pm, pm_path)
            print(f"Copied base policy memory to {pm_path}")

    # Run pipeline
    metrics = run_pipeline(
        seed=SEED,
        run_id=SEED,
        ablation_config={"baselines": [
            "Proposed Hybrid GA-GWO", "No Augmentation", "EDA", "Synonym Replacement", 
            "Back Translation", "Contextual Augmentation (BERT)", "DistilBERT Only", "RoBERTa Only"
        ]},
        output_dir=run_dir,
        debug=False,
        hyperparams=load_hyperparams()
    )
    
    gc.collect()
    print(f"Completed Seed {SEED}")

if __name__ == "__main__":
    main()
