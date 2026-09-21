from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class GAResult:
    selected_features: list[str]
    best_score: float
    best_generation: int
    history: pd.DataFrame
    chromosome: np.ndarray


class GeneticRFSelector:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        initial_feature_probability: float = 0.7,
        elite_size: int = 2,
        n_splits: int = 5,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.initial_feature_probability = (
            initial_feature_probability
        )
        self.elite_size = elite_size
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._fitness_cache: dict[bytes, float] = {}

    def _initialize_population(
        self,
        n_features: int,
    ) -> list[np.ndarray]:
        population = []

        for _ in range(self.population_size):
            chromosome = (
                np.random.random(n_features)
                < self.initial_feature_probability
            )

            # Chromosome with zero features is invalid.
            if not chromosome.any():
                chromosome[
                    np.random.randint(0, n_features)
                ] = True

            population.append(chromosome)

        return population

    def _fitness(
        self,
        chromosome: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        key = chromosome.tobytes()

        if key in self._fitness_cache:
            return self._fitness_cache[key]

        if not chromosome.any():
            return 0.0

        X_selected = X.iloc[:, chromosome]

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        score = cross_val_score(
            model,
            X_selected,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()

        score = float(score)

        self._fitness_cache[key] = score

        return score

    def _select_parent(
        self,
        population: list[np.ndarray],
        scores: np.ndarray,
    ) -> np.ndarray:
        total = scores.sum()

        if total == 0:
            index = random.randrange(
                len(population)
            )
            return population[index].copy()

        probabilities = scores / total

        index = np.random.choice(
            len(population),
            p=probabilities,
        )

        return population[index].copy()

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        child1 = parent1.copy()
        child2 = parent2.copy()

        if (
            random.random() < self.crossover_rate
            and len(parent1) > 2
        ):
            point = random.randint(
                1,
                len(parent1) - 2,
            )

            child1 = np.concatenate(
                (
                    parent1[:point],
                    parent2[point:],
                )
            )

            child2 = np.concatenate(
                (
                    parent2[:point],
                    parent1[point:],
                )
            )

        return child1, child2

    def _mutate(
        self,
        chromosome: np.ndarray,
    ) -> np.ndarray:
        chromosome = chromosome.copy()

        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = not chromosome[i]

        # Keep at least one selected feature.
        if not chromosome.any():
            chromosome[
                random.randrange(len(chromosome))
            ] = True

        return chromosome

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> GAResult:
        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        if X.isna().any().any():
            raise ValueError(
                "X contains missing values."
            )

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        self._fitness_cache = {}

        population = self._initialize_population(
            X.shape[1]
        )

        best_score = -1.0
        best_chromosome: np.ndarray | None = None
        best_generation = -1

        history = []

        for generation in range(
            self.generations
        ):
            scores = np.array(
                [
                    self._fitness(
                        chromosome,
                        X,
                        y,
                    )
                    for chromosome in population
                ],
                dtype=float,
            )

            # Sort primarily by accuracy (descending),
            # then by feature count (ascending).
            order = sorted(
                range(len(population)),
                key=lambda i: (
                    -scores[i],
                    int(population[i].sum()),
                ),
            )

            population = [
                population[i]
                for i in order
            ]

            scores = scores[order]

            generation_best = float(
                scores[0]
            )

            # Several chromosomes can have exactly the
            # same best accuracy. Choose the smallest one.
            best_indices = np.flatnonzero(
                np.isclose(
                    scores,
                    generation_best,
                )
            )

            generation_best_index = min(
                best_indices,
                key=lambda i: int(
                    population[i].sum()
                ),
            )

            generation_best_chromosome = (
                population[
                    generation_best_index
                ].copy()
            )

            feature_count = int(
                generation_best_chromosome.sum()
            )

            is_better_score = (
                generation_best > best_score
                and not np.isclose(
                    generation_best,
                    best_score,
                )
            )

            is_equal_but_smaller = (
                np.isclose(
                    generation_best,
                    best_score,
                )
                and best_chromosome is not None
                and feature_count
                < int(best_chromosome.sum())
            )

            if (
                best_chromosome is None
                or is_better_score
                or is_equal_but_smaller
            ):
                best_score = generation_best

                best_chromosome = (
                    generation_best_chromosome.copy()
                )

                best_generation = generation

            global_feature_count = int(
                best_chromosome.sum()
            )

            history.append(
                {
                    "generation": generation,
                    "best_accuracy": generation_best,
                    "best_features": feature_count,
                    "global_best_accuracy": best_score,
                    "global_best_features":
                        global_feature_count,
                    "unique_evaluations": len(
                        self._fitness_cache
                    ),
                }
            )

            print(
                f"gen={generation:4d} "
                f"best={generation_best:.4f} "
                f"features={feature_count:2d} "
                f"global={best_score:.4f} "
                f"global_features="
                f"{global_feature_count:2d}"
            )

            # Elitism:
            # keep the best chromosomes unchanged.
            next_population = [
                chromosome.copy()
                for chromosome
                in population[
                    :self.elite_size
                ]
            ]

            while (
                len(next_population)
                < self.population_size
            ):
                parent1 = self._select_parent(
                    population,
                    scores,
                )

                parent2 = self._select_parent(
                    population,
                    scores,
                )

                child1, child2 = (
                    self._crossover(
                        parent1,
                        parent2,
                    )
                )

                child1 = self._mutate(
                    child1
                )

                child2 = self._mutate(
                    child2
                )

                next_population.append(
                    child1
                )

                if (
                    len(next_population)
                    < self.population_size
                ):
                    next_population.append(
                        child2
                    )

            population = next_population

        if best_chromosome is None:
            raise RuntimeError(
                "GA did not produce a valid chromosome."
            )

        selected_features = (
            X.columns[
                best_chromosome
            ]
            .tolist()
        )

        return GAResult(
            selected_features=selected_features,
            best_score=best_score,
            best_generation=best_generation,
            history=pd.DataFrame(history),
            chromosome=best_chromosome,
        )