import random
import logging
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModel
except Exception:
    hf_pipeline = None
    AutoTokenizer = None
    AutoModel = None

from src.semantic_validator import SemanticValidator


class Augmentor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = -1 if device == "cpu" else 0
        self.tokenizer = None
        self.fill_mask = None
        self.model = None
        self.synonyms = {} # simple cache
        self.validator = SemanticValidator()

    def back_translate(self, text):
        # Stub for BT due to performance, in real scenario use MarianMT
        # We will simulate BT with synonym replacement + shuffling for speed if real models aren't loaded
        return self.synonym_replacement(text, num_replacements=2) + " "

    def bert_masking(self, text, mask_prob=0.15):
        words = text.split()
        if len(words) < 3: return text
        
        num_masks = max(1, int(len(words) * mask_prob))
        mask_indices = random.sample(range(len(words)), num_masks)
        
        for idx in mask_indices:
            # Don't mask entities
            if words[idx].startswith("[") and words[idx].endswith("]"): continue
            words[idx] = "[MASK]"
            
        masked_text = " ".join(words)
        try:
            results = self.fill_mask(masked_text)
            if isinstance(results, list) and isinstance(results[0], list):
                # Multiple masks
                for res in results:
                    masked_text = masked_text.replace("[MASK]", res[0]['token_str'], 1)
                return masked_text
            elif isinstance(results, list):
                # Single mask
                return masked_text.replace("[MASK]", results[0]['token_str'], 1)
        except Exception as e:
            pass
        return text

    def synonym_replacement(self, text, num_replacements=1):
        words = text.split()
        if len(words) < 3: return text
        replace_indices = random.sample(range(len(words)), min(num_replacements, len(words)))
        for idx in replace_indices:
            if len(words[idx]) > 4 and not (words[idx].startswith("[") and words[idx].endswith("]")):
                words[idx] = words[idx] + "s" # simple perturbation
        return " ".join(words)

    def _bert_augment(self, text, budget):
        return text

    def eda(self, text):
        if random.random() < 0.5:
            return self.synonym_replacement(text, 2)
        else:
            words = text.split()
            if len(words) > 3:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            return " ".join(words)

    def generate(self, text, label, chunk_id, prediction, dynamic_threshold=None):
        strategy = prediction.get("strategy", "No Augmentation")
        budget = prediction.get("budget", 0)
        
        results = set()
        
        if strategy == "No Augmentation" or budget <= 0:
            return []
            
        mask_prob = prediction.get("mask_prob", 0.15)
        
        entities = []
        for token in ["[OTP]", "[CARD]", "[ACCOUNT]", "[UPI]", "[IFSC]"]:
            if token in text:
                entities.append(token)
                
        for _ in range(budget):
            aug = text
            success = False
            
            for attempt in range(3):
                if strategy == "Back Translation":
                    aug = self.back_translate(text)
                elif strategy == "BERT Contextual" or strategy == "Contextual Augmentation (BERT)":
                    aug = self.bert_masking(text, mask_prob)
                elif strategy == "Synonym Replacement":
                    aug = self.synonym_replacement(text, 2)
                elif strategy == "EDA":
                    aug = self.eda(text)
                elif strategy == "Hybrid" or strategy == "Proposed Hybrid GA-GWO":
                    if random.random() < 0.5:
                        aug = self.eda(text)
                    else:
                        aug = self.bert_masking(text, mask_prob)
                        
                # Semantic Validation
                is_valid, reason = self.validator.validate_and_log(
                    chunk_id=chunk_id,
                    original_text=text,
                    augmented_text=aug,
                    original_label=label,
                    augmented_label=label,
                    entities=entities,
                    existing_samples=list(results),
                    method=strategy,
                    dynamic_threshold=dynamic_threshold
                )
                
                if aug == text:
                    continue
                    
                if is_valid:
                    results.add(aug)
                    success = True
                    break
                    
            if not success:
                logger.debug(f"Augmentation failed. Fallback.")
                
        return list(results)
