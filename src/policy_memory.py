import csv
import os
import datetime
import numpy as np

class PolicyMemory:
    def __init__(self, capacity=1000, csv_path="results/policy_memory.csv"):
        self.capacity = capacity
        self.csv_path = csv_path
        self.memory = []
        
        # GWO specific tracking
        self.alpha_wolves = []
        self.beta_wolves = []
        self.delta_wolves = []
        self.elite_population = []
        self.seen_alphas = set() # For deduplication
        
        # Full-population candidate log (for surrogate training)
        self._candidates: list = []          # in-memory list of all candidates
        self._candidates_csv = csv_path.replace(".csv", "_candidates.csv")
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Chunk_ID", "Strategy", "Budget", "Fitness", "Macro_F1", 
                    "Utility", "Confidence", "Cost", "FACI_Vector", "Semantic_Threshold"
                ])
        else:
            self.load_from_csv()
            
        if not os.path.exists(self._candidates_csv):
            with open(self._candidates_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Chunk_ID", "Strategy", "Budget",
                    "Fitness", "Macro_F1", "Alpha", "FACI_Vector"
                ])
        else:
            self._load_candidates_from_csv()


    def load_from_csv(self):
        import ast
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        faci_vec = ast.literal_eval(row['FACI_Vector'])
                    except:
                        faci_vec = None
                    # We can't recover full elite_pop, but we can store strategy and faci vector
                    # to influence future decisions
                    entry = {
                        "chunk_id": row.get('Chunk_ID', 0),
                        "timestamp": row.get('Timestamp', ''),
                        "faci_vector": faci_vec,
                        "strategy": row.get('Strategy', ''),
                        "budget": float(row.get('Budget', 0)),
                        "fitness": float(row.get('Fitness', 0)),
                        "macro_f1": float(row.get('Macro_F1', 0)),
                        "utility": float(row.get('Utility', 0)),
                        "confidence": float(row.get('Confidence', 0)),
                        "cost": float(row.get('Cost', 0)),
                        "elite_pop": [] # Missing from CSV
                    }
                    self.memory.append(entry)
                    if len(self.memory) > self.capacity:
                        self.memory.pop(0)
        except Exception as e:
            print(f"Warning: Could not load policy memory from CSV: {e}")

    def _load_candidates_from_csv(self):
        """Reload candidate log from disk on startup."""
        import ast
        try:
            with open(self._candidates_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        alpha = ast.literal_eval(row.get('Alpha', '[]'))
                        faci_vec = ast.literal_eval(row.get('FACI_Vector', '[]'))
                    except:
                        alpha, faci_vec = [], []
                    self._candidates.append({
                        "chunk_id":  row.get('Chunk_ID'),
                        "strategy":  row.get('Strategy'),
                        "budget":    float(row.get('Budget', 0)),
                        "fitness":   float(row.get('Fitness', 0)),
                        "macro_f1":  float(row.get('Macro_F1', 0)),
                        "alpha":     alpha,
                        "faci_vector": faci_vec,
                    })
        except Exception as e:
            print(f"Warning: Could not load candidates CSV: {e}")

                
    def add_state(self, chunk_id, faci_vector, strategy, fitness, macro_f1, utility, confidence, cost, budget=0, alpha=None, beta=None, delta=None, elite_pop=None, semantic_threshold=None):
        timestamp = datetime.datetime.now().isoformat()
        
        # Deduplication check
        if alpha is not None:
            alpha_tuple = tuple(np.round(alpha, 4))
            if alpha_tuple in self.seen_alphas:
                pass 
            else:
                self.seen_alphas.add(alpha_tuple)
        
        entry = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "faci_vector": faci_vector,
            "strategy": strategy,
            "budget": budget,
            "fitness": fitness,
            "macro_f1": macro_f1,
            "utility": utility,
            "confidence": confidence,
            "cost": cost,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "elite_pop": elite_pop or [],
            "semantic_threshold": semantic_threshold
        }
        
        self.memory.append(entry)
        
        if alpha is not None: self.alpha_wolves.append(alpha)
        if beta is not None: self.beta_wolves.append(beta)
        if delta is not None: self.delta_wolves.append(delta)
        if elite_pop is not None and len(elite_pop) > 0: 
            # Deduplicate elites before assigning
            unique_elites = []
            seen_elites = set()
            for e in elite_pop:
                t = tuple(np.round(e, 4))
                if t not in seen_elites:
                    seen_elites.add(t)
                    unique_elites.append(e)
            self.elite_population = unique_elites

        if len(self.memory) > self.capacity:
            removed = self.memory.pop(0)
            if removed.get("alpha") is not None:
                a_t = tuple(np.round(removed["alpha"], 4))
                if a_t in self.seen_alphas:
                    self.seen_alphas.remove(a_t)
            if self.alpha_wolves: self.alpha_wolves.pop(0)
            if self.beta_wolves: self.beta_wolves.pop(0)
            if self.delta_wolves: self.delta_wolves.pop(0)
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, chunk_id, strategy, budget, fitness, macro_f1, 
                utility, confidence, cost, str(faci_vector), semantic_threshold
            ])

    def add_candidate(self, chunk_id: int, faci_vector, alpha, macro_f1: float,
                      strategy: str, budget: int, fitness: float) -> None:
        """
        Log a SINGLE evaluated candidate (not necessarily the winner) into
        the full-population candidate store used for surrogate training.
        This must be called for every candidate evaluated in GA/GWO, not
        just the alpha winner.
        """
        timestamp = datetime.datetime.now().isoformat()
        entry = {
            "chunk_id":   chunk_id,
            "strategy":   strategy,
            "budget":     budget,
            "fitness":    fitness,
            "macro_f1":   macro_f1,
            "alpha":      list(alpha) if alpha is not None else [],
            "faci_vector": list(faci_vector) if faci_vector is not None else [],
        }
        self._candidates.append(entry)
        with open(self._candidates_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, chunk_id, strategy, budget,
                fitness, macro_f1,
                str(list(alpha) if alpha is not None else []),
                str(list(faci_vector) if faci_vector is not None else [])
            ])

    def get_all_candidates(self):
        """
        Return all logged candidates as (X, y) for surrogate training.
        X = alpha vectors, y = macro_f1 values.
        Only returns entries with valid alpha vectors (len == 8).
        """
        X, y = [], []
        for c in self._candidates:
            alpha = c.get('alpha', [])
            f1    = c.get('macro_f1', None)
            if alpha and len(alpha) == 8 and f1 is not None:
                X.append(alpha)
                y.append(f1)
        return X, y

            
    def get_last_alpha_beta_delta(self):
        alpha = self.alpha_wolves[-1] if self.alpha_wolves else None
        beta = self.beta_wolves[-1] if self.beta_wolves else None
        delta = self.delta_wolves[-1] if self.delta_wolves else None
        return alpha, beta, delta
        
    def get_elite_population(self):
        return self.elite_population

    def get_elites_by_faci_similarity(self, current_faci_vector):
        """
        Retrieve previous elites based on Cosine Similarity of FACI vectors.
        """
        if not self.memory or current_faci_vector is None:
            return self.elite_population
            
        curr_vec = np.array(current_faci_vector)
        norm_curr = np.linalg.norm(curr_vec)
        if norm_curr == 0:
            norm_curr = 1e-9
            
        best_sim = -float('inf')
        best_elites = self.elite_population
        
        for entry in self.memory:
            if 'faci_vector' in entry and entry['faci_vector'] is not None:
                hist_vec = np.array(entry['faci_vector'])
                norm_hist = np.linalg.norm(hist_vec)
                if norm_hist == 0:
                    norm_hist = 1e-9
                    
                # Cosine Similarity
                sim = np.dot(curr_vec, hist_vec) / (norm_curr * norm_hist)
                
                if sim > best_sim:
                    best_sim = sim
                    if 'elite_pop' in entry and entry['elite_pop']:
                        best_elites = entry['elite_pop']
                        
        return best_elites
