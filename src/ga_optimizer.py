import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self, config, bounds, memory=None, faci=None):
        self.pop_size = config['population_size']
        self.generations = config['generations']
        self.crossover_rate = config['crossover_rate']
        self.base_mutation_rate = config['mutation_rate']
        self.elite_size = config['elite_size']
        
        self.bounds = bounds
        self.memory = memory
        self.faci = faci
        self.num_genes = len(bounds)
        
        self.population = self._initialize_population()
        self.fitness_history = []
        self.diversity_history = []
        self.mutation_history = []
        self.selection_pressure_history = []
        
        self.fitness_cache = {}

        # ── Surrogate (RandomForest, trained from PolicyMemory candidates) ──
        self.surrogate = None
        self._surrogate_fallback_count = 0  # per-optimize() call
        self._train_surrogate()
        
    # ------------------------------------------------------------------ #
    # Surrogate management                                                  #
    # ------------------------------------------------------------------ #

    def _train_surrogate(self):
        """
        Build / refresh the SurrogateModel from the full candidate population
        stored in PolicyMemory (not just the alpha winner).
        Runs the quality gate; only marks surrogate trusted if R² ≥ 0.30.
        """
        from src.surrogate import SurrogateModel

        if not self.memory:
            return

        # Prefer full-population candidates; fall back to winner-only entries
        if hasattr(self.memory, 'get_all_candidates'):
            X, y = self.memory.get_all_candidates()
        else:
            X, y = [], []
            for entry in self.memory.memory:
                if 'alpha' in entry and entry['alpha'] is not None and 'macro_f1' in entry:
                    X.append(entry['alpha'])
                    y.append(entry['macro_f1'])

        # Create new surrogate and load data
        surrogate = SurrogateModel(results_dir=getattr(self.memory, 'results_dir', 'results'))
        surrogate.add_samples(X, y)

        # Gate check — only trust if quality threshold is met
        chunk_id = getattr(self, '_current_chunk_id', -1)
        gate_passed = surrogate.quality_gate(chunk_id=chunk_id)
        if gate_passed:
            self.surrogate = surrogate
            logger.info(f"[GA] Surrogate TRUSTED | n={surrogate.sample_count} | R²={surrogate.r2:.3f}")
        else:
            self.surrogate = surrogate  # keep for tracking, but is_trusted=False
            logger.info(f"[GA] Surrogate NOT trusted yet | n={surrogate.sample_count} | R²={surrogate.r2:.3f}")

    def _heuristic_fitness(self, chromosome):
        """Original hand-coded fitness heuristic (used as fallback)."""
        bt, bert, budget, mask, sem, ent, pri, lr = chromosome
        cif = self.faci.get('class_imbalance_factor', 1.0) if hasattr(self.faci, 'get') else 1.0
        comp_cost = (bt * 0.4 + bert * 0.2) * (budget / 5.0)
        macro_f1 = min((bt * 0.4 + bert * 0.4) * (budget / 5.0) + 0.5, 0.95)
        semantic_similarity = sem
        entity_preservation = ent
        diversity_metric = (bt * bert) + (mask * 0.5)
        minority_gain = (budget / 5.0) * (1.0 - 1.0 / max(cif, 1e-9)) if cif > 1.0 else 0.0
        augmentation_fairness = 1.0 - abs(bt - bert) * 0.5
        return (
            0.35 * macro_f1 +
            0.20 * minority_gain +
            0.15 * entity_preservation +
            0.10 * semantic_similarity +
            0.10 * diversity_metric +
            0.10 * augmentation_fairness -
            0.10 * comp_cost
        )

    # ------------------------------------------------------------------ #
    # Fitness function with surrogate + uncertainty fallback               #
    # ------------------------------------------------------------------ #

    def fitness_function(self, chromosome):
        # Fitness Caching
        chrom_tuple = tuple(np.round(chromosome, 4))
        if chrom_tuple in self.fitness_cache:
            return self.fitness_cache[chrom_tuple]

        bt, bert, budget, mask, sem, ent, pri, lr = chromosome
        cif = self.faci.get('class_imbalance_factor', 1.0) if hasattr(self.faci, 'get') else 1.0
        comp_cost = (bt * 0.4 + bert * 0.2) * (budget / 5.0)

        if self.surrogate is not None and self.surrogate.is_trusted:
            pred_f1, uncertainty = self.surrogate.predict(chromosome)

            if self.surrogate.is_high_uncertainty(uncertainty):
                # Active-learning fallback: evaluate with heuristic, feed back to surrogate
                heuristic_fit = self._heuristic_fitness(chromosome)
                # Blend: 70% heuristic, 30% surrogate (regularised tie-breaker)
                fitness = 0.70 * heuristic_fit + 0.30 * (pred_f1 - 0.05 * comp_cost)
                self._surrogate_fallback_count += 1
                # Feed this sample back immediately (active learning)
                self.surrogate.update([list(chromosome)], [heuristic_fit])
            else:
                fitness = pred_f1 - 0.05 * comp_cost
        else:
            fitness = self._heuristic_fitness(chromosome) - 0.05 * comp_cost

        self.fitness_cache[chrom_tuple] = fitness
        return fitness
        
    # ------------------------------------------------------------------ #
    # Population helpers                                                    #
    # ------------------------------------------------------------------ #

    def _initialize_population(self):
        pop = []
        if self.memory:
            # Reusing past elites
            elites = self.memory.get_elites_by_faci_similarity(self.faci)
            if elites:
                for p in elites:
                    pop.append(np.array(p))
                
        while len(pop) < self.pop_size:
            ind = [random.uniform(b[0], b[1]) for b in self.bounds]
            pop.append(np.array(ind))
        return np.array(pop[:self.pop_size])
        
    def _calculate_diversity(self):
        if len(self.population) < 2:
            return 0.0
        return np.mean(np.std(self.population, axis=0))
        
    def _tournament_selection(self, fitnesses, k=3):
        selected = random.sample(range(self.pop_size), k)
        best = max(selected, key=lambda i: fitnesses[i])
        return self.population[best]
        
    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            pt = random.randint(1, self.num_genes - 1)
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
            return c1, c2
        return p1.copy(), p2.copy()
        
    def _mutate(self, ind, current_mutation_rate):
        for i in range(self.num_genes):
            if random.random() < current_mutation_rate:
                ind[i] += random.gauss(0, 0.1 * (self.bounds[i][1] - self.bounds[i][0]))
                ind[i] = np.clip(ind[i], self.bounds[i][0], self.bounds[i][1])
        return ind
        
    def optimize(self):
        cif = self.faci.get('class_imbalance_factor', 1.0) if hasattr(self.faci, 'get') else 1.0
        
        best_overall_fitness = -float('inf')
        generations_without_improvement = 0
        self._surrogate_fallback_count = 0   # reset per optimize() call
        
        # Track per-generation uncertainties for threshold update
        all_uncertainties = []
        
        for gen in range(self.generations):
            fitnesses = [self.fitness_function(ind) for ind in self.population]

            # Collect uncertainties this generation for threshold calibration
            if self.surrogate is not None and self.surrogate.is_trusted:
                gen_unc = [self.surrogate.predict(ind)[1] for ind in self.population]
                all_uncertainties.extend(gen_unc)
                self.surrogate.update_uncertainty_threshold(gen_unc)

            max_fit = np.max(fitnesses)
            self.fitness_history.append(max_fit)
            
            # Early Stopping Check
            if max_fit > best_overall_fitness:
                best_overall_fitness = max_fit
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 5:
                break
            
            avg_fit = np.mean(fitnesses)
            selection_pressure = max_fit / (avg_fit + 1e-9)
            self.selection_pressure_history.append(selection_pressure)
            
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            target_diversity = 0.2
            base_mut = self.base_mutation_rate
            if cif > 2.0:
                base_mut *= 1.5
                
            if diversity < target_diversity:
                current_mut_rate = min(base_mut * 2.0, 0.5)
            else:
                current_mut_rate = base_mut
                
            self.mutation_history.append(current_mut_rate)
            
            # Elite Preservation
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_pop = [self.population[i] for i in elite_indices]
            
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(fitnesses)
                p2 = self._tournament_selection(fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1, current_mut_rate))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2, current_mut_rate))
                    
            self.population = np.array(new_pop)
            
        final_fitness = [self.fitness_function(ind) for ind in self.population]
        best_indices = np.argsort(final_fitness)[::-1]
        elite_policies = self.population[best_indices]

        # Log fallback count to surrogate if available
        if self.surrogate is not None:
            self.surrogate.log_fallback(
                chunk_id=getattr(self, '_current_chunk_id', -1),
                fallback_count=self._surrogate_fallback_count,
                total_candidates=self.pop_size * max(1, len(self.fitness_history))
            )
        
        metrics = {
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
            "mutation_history": self.mutation_history,
            "selection_pressure": self.selection_pressure_history,
            "surrogate_fallbacks": self._surrogate_fallback_count,
        }
        
        return elite_policies, metrics
