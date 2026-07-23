import random
from collections import defaultdict
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=1000, num_classes=5):
        self.capacity = capacity
        self.num_classes = num_classes
        self.buffer = []
        self.priorities = [] # parallel array for priorities
        
        # Keep track of indices per class
        self.class_indices = defaultdict(list)
        self.total_seen_per_class = defaultdict(int)
        self.total_seen = 0
        
    def add(self, samples, confidences=None):
        """
        samples: list of tuples (text, label)
        confidences: list of floats, priority is inversely proportional to confidence
        Uses Priority-aware Balanced Reservoir Sampling
        """
        if confidences is None:
            confidences = [1.0] * len(samples)
            
        for sample, conf in zip(samples, confidences):
            text, label = sample
            # Priority: lower confidence -> higher priority, cap at 2.0. Base 1.0.
            priority = 1.0 + (1.0 - conf)
            
            self.total_seen += 1
            self.total_seen_per_class[label] += 1
            
            # Per-class reservoir capacity (approx equal division)
            class_capacity = self.capacity // self.num_classes
            
            if len(self.class_indices[label]) < class_capacity:
                self.buffer.append(sample)
                self.priorities.append(priority)
                self.class_indices[label].append(len(self.buffer) - 1)
            else:
                # Weighted reservoir replacement
                # In standard reservoir, replace with prob (capacity / seen)
                # Here we introduce a simple priority chance
                j = random.randint(0, self.total_seen_per_class[label] - 1)
                if j < class_capacity:
                    buffer_idx = self.class_indices[label][j]
                    # Only replace if new sample has higher priority or by 30% chance
                    if priority > self.priorities[buffer_idx] or random.random() < 0.3:
                        self.buffer[buffer_idx] = sample
                        self.priorities[buffer_idx] = priority
                        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
            
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
            
        # Balanced batch sampling with max 35% per class rule
        max_samples_per_class = int(batch_size * 0.35)
        samples_per_class = max(1, batch_size // self.num_classes)
        
        batch_indices = []
        classes_available = list(self.class_indices.keys())
        
        for c in classes_available:
            indices_for_c = self.class_indices[c]
            if not indices_for_c:
                continue
                
            num_to_sample = min(samples_per_class, len(indices_for_c), max_samples_per_class)
            
            # Priority sampling
            weights = [self.priorities[idx] for idx in indices_for_c]
            try:
                sampled = random.choices(indices_for_c, weights=weights, k=num_to_sample)
            except ValueError:
                sampled = random.sample(indices_for_c, num_to_sample)
            batch_indices.extend(set(sampled)) # remove duplicates if choices picked same
            
        # If choices resulted in fewer unique indices, we pad
        # Fill remaining if needed
        remaining = batch_size - len(batch_indices)
        if remaining > 0:
            all_other_indices = list(set(range(len(self.buffer))) - set(batch_indices))
            if all_other_indices:
                weights = [self.priorities[idx] for idx in all_other_indices]
                try:
                    fill_sampled = random.choices(all_other_indices, weights=weights, k=remaining)
                except ValueError:
                    fill_sampled = random.sample(all_other_indices, min(remaining, len(all_other_indices)))
                batch_indices.extend(set(fill_sampled))
                
        # Shuffle batch
        random.shuffle(batch_indices)
        
        # Limit to batch size if set() expanded or contracted it
        batch_indices = batch_indices[:batch_size]
        
        return [self.buffer[i] for i in batch_indices]
        
    def get_statistics(self):
        return {
            "total_capacity": self.capacity,
            "current_size": len(self.buffer),
            "class_distribution": {k: len(v) for k, v in self.class_indices.items()},
            "total_seen": self.total_seen
        }
