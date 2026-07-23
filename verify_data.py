import json
import numpy as np
from src.augmentor import Augmentor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_data_leakage(train_file="data/train.json", test_file="data/test.json"):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        
    train_ids = set([d["complaint_id"] for d in train_data])
    test_ids = set([d["complaint_id"] for d in test_data])
    
    overlap = train_ids.intersection(test_ids)
    if overlap:
        logger.error(f"DATA LEAKAGE DETECTED! {len(overlap)} overlapping IDs.")
    else:
        logger.info("Data Leakage Check: PASSED. 0 overlapping IDs between train and test.")
        
def verify_class_balance(train_file="data/train.json"):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
        
    class_counts = {}
    for d in train_data:
        class_counts[d['label']] = class_counts.get(d['label'], 0) + 1
        
    logger.info("Class Distribution in Training Set:")
    for k, v in class_counts.items():
        logger.info(f"  - {k}: {v} samples")
        
    # Check if highly imbalanced
    vals = list(class_counts.values())
    ratio = max(vals) / (min(vals) + 1e-9)
    if ratio > 3:
        logger.info("Class imbalance is high (> 3:1). Class Balanced Loss (CBL) is justified.")
    else:
        logger.info("Class balance is reasonable.")
        
def verify_augmentation_quality(train_file="data/train.json"):
    with open(train_file, 'r') as f:
        train_data = json.load(f)
        
    sample = train_data[0]["complaint_what_happened_clean"]
    
    logger.info("\n--- Augmentation Quality Check ---")
    logger.info(f"ORIGINAL TEXT: {sample[:150]}...")
    
    aug = Augmentor()
    
    # Test EDA
    res_eda = aug.apply(sample, strategy={"eda": 1.0}, budget=2)
    logger.info(f"EDA AUGMENTED: {res_eda[0][:150]}...")
    
    # Test Synonym
    res_syn = aug.apply(sample, strategy={"synonym": 1.0}, budget=2)
    logger.info(f"SYNONYM AUGMENTED: {res_syn[0][:150]}...")

if __name__ == "__main__":
    verify_data_leakage()
    verify_class_balance()
    verify_augmentation_quality()
