"""
src/surrogate.py — RandomForest surrogate model for GA-GWO policy fitness.

Features:
  - predict()         → (predicted_f1, uncertainty) via tree variance
  - quality_gate()    → True if R² ≥ threshold on 20% holdout
  - update()          → incremental refit with new (policy, f1) pairs
  - is_trusted        → property; True only after gate passes
  - Logs R², MAE, gate result and per-chunk fallback counts to CSV
"""

import os
import csv
import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False


class SurrogateModel:
    """
    Wraps RandomForestRegressor with uncertainty quantification and quality gating.

    Policy vector (8 dims):
        [bt_weight, bert_weight, budget, mask_prob, sem_thresh, entity_weight, chunk_priority, lr]
    """

    MIN_SAMPLES_FOR_GATE = 15   # Don't even try until we have this many
    R2_THRESHOLD        = 0.30  # Quality gate floor
    UNCERTAINTY_PERCENTILE = 75 # High-uncertainty fallback trigger

    def __init__(self, results_dir: str = "results", log_csv: str = "surrogate_log.csv"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self._log_path = os.path.join(self.results_dir, log_csv)
        self._init_log()

        self._X: List[List[float]] = []
        self._y: List[float]       = []

        self._rf: Optional["RandomForestRegressor"] = None
        self._is_trusted: bool  = False
        self._r2: float         = -999.0
        self._mae: float        = 999.0
        self._last_uncertainties: List[float] = []

        # 75th-percentile uncertainty threshold; updated each generation
        self._unc_threshold: float = 1.0

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def is_trusted(self) -> bool:
        return self._is_trusted

    @property
    def r2(self) -> float:
        return self._r2

    @property
    def sample_count(self) -> int:
        return len(self._y)

    def add_samples(self, X_new: List[List[float]], y_new: List[float]) -> None:
        """Add (policy_vector, f1) pairs to the training set."""
        for x, y in zip(X_new, y_new):
            self._X.append(list(x))
            self._y.append(float(y))

    def quality_gate(self, chunk_id: int = -1) -> bool:
        """
        Train on 80%, evaluate on 20%, log result.
        Returns True if R² ≥ R2_THRESHOLD and enough samples exist.
        """
        n = len(self._y)
        if n < self.MIN_SAMPLES_FOR_GATE or not _RF_AVAILABLE:
            self._log_gate(chunk_id, n, -999.0, 999.0, passed=False,
                           reason=f"insufficient samples ({n} < {self.MIN_SAMPLES_FOR_GATE})")
            return False

        X = np.array(self._X)
        y = np.array(self._y)

        if n < 10:
            # Too small for 80/20 split — skip gate
            self._log_gate(chunk_id, n, -999.0, 999.0, passed=False,
                           reason="too small for split")
            return False

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_val)
            r2  = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
        except Exception as e:
            logger.warning(f"[Surrogate] quality_gate fit failed: {e}")
            self._log_gate(chunk_id, n, -999.0, 999.0, passed=False, reason=str(e))
            return False

        self._r2  = r2
        self._mae = mae
        passed = r2 >= self.R2_THRESHOLD

        if passed:
            # Refit on full data for production use
            self._rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            self._rf.fit(X, y)
            self._is_trusted = True

        self._log_gate(chunk_id, n, r2, mae, passed=passed,
                       reason="OK" if passed else f"R²={r2:.3f} < {self.R2_THRESHOLD}")
        return passed

    def predict(self, policy: List[float]) -> Tuple[float, float]:
        """
        Returns (predicted_f1, uncertainty).
        uncertainty = std of predictions across individual trees.
        """
        if self._rf is None or not self._is_trusted:
            return 0.0, 1.0   # max uncertainty when not trusted

        arr = np.array(policy).reshape(1, -1)
        tree_preds = np.array([tree.predict(arr)[0] for tree in self._rf.estimators_])
        return float(np.mean(tree_preds)), float(np.std(tree_preds))

    def update_uncertainty_threshold(self, uncertainties: List[float]) -> None:
        """Recompute the 75th-percentile uncertainty cutoff from latest generation."""
        if uncertainties:
            self._last_uncertainties = uncertainties
            self._unc_threshold = float(np.percentile(uncertainties, self.UNCERTAINTY_PERCENTILE))

    def is_high_uncertainty(self, uncertainty: float) -> bool:
        return uncertainty > self._unc_threshold

    def update(self, X_new: List[List[float]], y_new: List[float]) -> None:
        """Add samples and refit the full RF (called at end of every chunk)."""
        self.add_samples(X_new, y_new)
        if self._is_trusted and len(self._y) >= self.MIN_SAMPLES_FOR_GATE and _RF_AVAILABLE:
            try:
                X = np.array(self._X)
                y = np.array(self._y)
                self._rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                self._rf.fit(X, y)
            except Exception as e:
                logger.warning(f"[Surrogate] update refit failed: {e}")

    def log_fallback(self, chunk_id: int, fallback_count: int,
                     total_candidates: int) -> None:
        """Log per-chunk fallback statistics."""
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                chunk_id, "FALLBACK", self.sample_count,
                self._r2, self._mae, self._is_trusted,
                self._unc_threshold,
                f"{fallback_count}/{total_candidates}", ""
            ])

    # ------------------------------------------------------------------ #
    # Private                                                               #
    # ------------------------------------------------------------------ #

    def _init_log(self) -> None:
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Chunk_ID", "Event", "Sample_Count",
                    "R2", "MAE", "Is_Trusted",
                    "Unc_Threshold", "Fallback_Ratio", "Reason"
                ])

    def _log_gate(self, chunk_id: int, n: int, r2: float, mae: float,
                  passed: bool, reason: str) -> None:
        logger.info(f"[Surrogate][Chunk {chunk_id}] Gate {'PASS' if passed else 'FAIL'} "
                    f"| n={n} | R²={r2:.3f} | MAE={mae:.4f} | {reason}")
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                chunk_id, "GATE_PASS" if passed else "GATE_FAIL", n,
                round(r2, 4), round(mae, 4), passed,
                self._unc_threshold, "", reason
            ])
