import csv
import os
import logging
import numpy as np
from src.ga_optimizer import GeneticAlgorithm
from src.gwo_optimizer import GWOOptimizer

logger = logging.getLogger(__name__)

# ── Budget hard floor ──────────────────────────────────────────────────
# Derived from observed EDA / Synonym Replacement baseline behaviour:
# those baselines ran at budget=2, yielding ~0.40 accepted augs/item on
# average over all chunks.  The floor is therefore set to 2 budget units.
# FACI-based suppression is allowed to REDUCE budget above this floor
# (i.e. give extra budget in low-complexity chunks), but must NEVER push
# budget below it when augmentation is active.
BUDGET_HARD_FLOOR = 2


class HybridOptimizer:
    def __init__(self, ga_config, gwo_config, bounds, memory=None,
                 results_dir: str = "results"):
        self.ga_config    = ga_config
        self.gwo_config   = gwo_config
        self.bounds       = bounds
        self.memory       = memory
        self.results_dir  = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self._budget_log_path = os.path.join(self.results_dir, "budget_log.csv")
        self._init_budget_log()
        
    def _init_budget_log(self):
        if not os.path.exists(self._budget_log_path):
            with open(self._budget_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Chunk_ID", "Strategy", "Pre_Floor_Budget",
                    "Post_Floor_Budget", "Hard_Floor", "FACI_Scalar"
                ])

    def _log_budget(self, chunk_id: int, strategy: str,
                    pre: int, post: int, faci_scalar: float = 0.0):
        with open(self._budget_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([chunk_id, strategy, pre, post, BUDGET_HARD_FLOOR, round(faci_scalar, 4)])

    def _determine_strategy(self, p_bt, p_bert):
        if p_bt == 0 and p_bert == 0:
            return "No Augmentation"
        elif p_bt > 0.5 and p_bert > 0.5:
            return "Hybrid"
        elif p_bt > p_bert:
            return "Back Translation"
        else:
            return "BERT Contextual"

    def _apply_budget_floor(self, strategy: str, raw_budget: int,
                            chunk_id: int, faci_scalar: float = 0.0) -> int:
        """
        Enforce the hard budget floor.
        - If strategy is No Augmentation  → budget stays 0 (floor doesn't apply).
        - Otherwise                        → budget = max(BUDGET_HARD_FLOOR, raw_budget).
        Logs pre/post for audit.
        """
        if strategy == "No Augmentation":
            self._log_budget(chunk_id, strategy, 0, 0, faci_scalar)
            return 0
        post = max(BUDGET_HARD_FLOOR, raw_budget)
        self._log_budget(chunk_id, strategy, raw_budget, post, faci_scalar)
        return post

    def optimize(self, chunk_id: int, faci_features, faci_scalar: float = 0.0):
        logger.info(f"Starting Hybrid GA-GWO Optimization for Chunk {chunk_id}")

        # Pass chunk_id to GA so surrogate logger can annotate correctly
        ga = GeneticAlgorithm(self.ga_config, self.bounds, self.memory, faci_features)
        ga._current_chunk_id = chunk_id

        elite_policies, ga_metrics = ga.optimize()

        # ── Log ALL elite candidates into PolicyMemory ─────────────────
        if self.memory is not None and hasattr(self.memory, 'add_candidate'):
            # Use the heuristic fitness score for each elite as a proxy f1
            for i, ep in enumerate(elite_policies):
                strat = self._determine_strategy(ep[0], ep[1])
                bud   = int(round(ep[2]))
                fit   = ga.fitness_function(ep)
                # We don't have a real f1 for each candidate here — use fitness
                # as a proxy; real f1 from evaluator is added by main.py's
                # burn-in loop for the burn-in probes, and the alpha's f1 is
                # added via add_state().
                faci_vec = list(faci_features) if hasattr(faci_features, '__iter__') else []
                self.memory.add_candidate(
                    chunk_id=chunk_id,
                    faci_vector=faci_vec,
                    alpha=list(ep),
                    macro_f1=fit,          # proxy — overridden by real f1 in burn-in
                    strategy=strat,
                    budget=bud,
                    fitness=fit,
                )

        logger.info(f"GA complete. Best fitness: {ga_metrics['fitness_history'][-1]:.4f} | "
                    f"Surrogate fallbacks: {ga_metrics.get('surrogate_fallbacks', 0)}")
        
        # Phase 2: Local Exploitation via GWO
        gwo = GWOOptimizer(self.gwo_config, self.bounds, self.memory)
        alpha, beta, delta, alpha_score, gwo_metrics = gwo.optimize(elite_policies, ga.fitness_function)
        
        logger.info(f"GWO complete. Alpha score: {alpha_score:.4f}")
        
        # Phase 3: Policy decoding
        bt, bert, budget_raw, mask, sem, ent, pri, lr = alpha
        strategy = self._determine_strategy(bt, bert)

        # Apply hard budget floor
        pre_floor_budget = max(0, int(round(budget_raw)))
        budget = self._apply_budget_floor(strategy, pre_floor_budget, chunk_id, faci_scalar)
            
        expected_macro_f1 = min((bt * 0.4 + bert * 0.4) * (budget_raw / 5.0) + 0.5, 0.95)
        expected_cost     = (bt * 0.4 + bert * 0.2) * (budget_raw / 5.0)
        expected_utility  = alpha_score
        confidence        = min(alpha_score / 1.5, 1.0) if alpha_score > 0 else 0.0
        
        reason = (f"Based on FACI profile, GA-GWO selected {strategy} "
                  f"(Budget: {budget}, floor enforced from {pre_floor_budget}) "
                  f"aiming for Utility {expected_utility:.4f}.")
        
        prediction = {
            "strategy":          strategy,
            "budget":            budget,
            "expected_utility":  expected_utility,
            "expected_cost":     expected_cost,
            "expected_macro_f1": expected_macro_f1,
            "confidence":        confidence,
            "reason":            reason,
        }
        
        return {
            "policy":        alpha,
            "prediction":    prediction,
            "alpha":         alpha,
            "beta":          beta,
            "delta":         delta,
            "elite_policies": elite_policies.tolist(),
            "ga_metrics":    ga_metrics,
            "gwo_metrics":   gwo_metrics,
            "fitness":       alpha_score,
        }
