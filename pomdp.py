import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from scipy.stats import binom


class SyntheticPOMDP:
    def __init__(self, n_states, n_obs, n_actions, seed=42):
        self.S = n_states
        self.O = n_obs
        self.A = n_actions
        self.seed = seed
        np.random.seed(self.seed)
        
        self.T = np.zeros((n_actions, self.S, self.S))
        self.E = np.zeros((self.A, self.S, self.O))
        
        # Initial belief: Uniform distribution
        self.initial_belief = np.full(self.S, 1.0 / self.S)

    def generate_irreducible_transitions(self, p_move=0.7, p_random=0.5):
        """Action-based dynamics ensuring a strongly connected state space."""
        while True:
            T = np.zeros((self.A, self.S, self.S))
            for s in range(self.S):
                # Action 0/1: Noisy Cycling
                T[0, s, (s + 1) % self.S] = p_move
                T[0, s, :] += (1 - p_move) / self.S
                
                T[1, s, (s - 1) % self.S] = p_move
                T[1, s, :] += (1 - p_move) / self.S
                
                # Action 2: Random Jump + Self-Loop
                T[2, s, s] = p_random
                T[2, s, :] += (1 - p_random) * np.random.dirichlet([1.0] * self.S)

            # Combined adjacency check for irreducibility
            combined_adj = np.sum(T, axis=0) > 0
            G = nx.from_numpy_array(combined_adj, create_using=nx.DiGraph)
            if nx.is_strongly_connected(G):
                self.T = T
                break

    def _create_candidate_emissions(self):
        """Moderate spread: alpha [1.5, 3.5], cap 0.6, floor 0.01."""
        emissions = np.zeros((self.A, self.S, self.O))
        for a in range(self.A):
            for s in range(self.S):
                alpha = np.random.uniform(1.5, 3.5, size=self.O)
                dist = np.random.dirichlet(alpha)
                if dist.max() > 0.6:
                    dist = dist * (0.6 / dist.max())
                    dist = dist / dist.sum()
                dist = (dist + 0.01)
                emissions[a, s, :] = dist / dist.sum()
        return emissions

    def optimize_emissions(self, ent_range=(0.9, 0.95), max_iter=2000):
        """Finds E with max confusability and returns (max_pairs, corresponding_entropy)."""
        max_pairs = -1
        best_E = None
        best_ent = 0.0
        
        for _ in range(max_iter):
            candidate_E = self._create_candidate_emissions()
            pairs, ent = self._calculate_metrics(candidate_E)
            
            if ent_range[0] <= ent <= ent_range[1]:
                if pairs > max_pairs:
                    max_pairs = pairs
                    best_E = candidate_E
                    best_ent = ent
        
        if best_E is None:
            raise RuntimeError("Entropy constraints not met within max_iter.")
            
        self.E = best_E
        return max_pairs, best_ent

    def _calculate_metrics(self, emission_tensor):
        """Calculates Action-Conditioned TV pairs < 0.4 and Normalized Entropy."""
        confusable = 0
        for s1 in range(self.S):
            for s2 in range(s1 + 1, self.S):
                for a in range(self.A):
                    tv = 0.5 * np.sum(np.abs(emission_tensor[a, s1, :] - emission_tensor[a, s2, :]))
                    if tv < 0.4:
                        confusable += 1
        
        entropies = []
        for a in range(self.A):
            for s in range(self.S):
                p = emission_tensor[a, s, :]
                ent = -np.sum(p * np.log(p + 1e-12)) / np.log(self.O)
                entropies.append(ent)
                
        return confusable, np.mean(entropies)

    def step(self, state: int, action: int) -> Tuple[int, int]:
        """Pure function sampling (s, a) -> (o, s')."""
        p_next = self.T[action, state, :]
        next_state = np.random.choice(self.S, p=p_next)
        p_obs = self.E[action, next_state, :]
        observation = np.random.choice(self.O, p=p_obs)
        return observation, next_state

    def compute_true_kernel(self, history: List[Tuple[int, int]], action: int) -> np.ndarray:
        """Vectorized Bayesian filtering to find P(o | h, a)."""
        belief = self.initial_belief.copy()
        for obs, act in history:
            b_pred = self.T[act, :, :].T @ belief
            b_new = self.E[act, :, obs] * b_pred
            norm = b_new.sum()
            belief = b_new / norm if norm > 0 else self.initial_belief.copy()

        # P(o | h, a) = (belief * T_a) * E_a
        # Note (numerical stability): in exact arithmetic this is a probability distribution over observations
        # (i.e., it sums to 1). In practice, tiny floating-point drift in T/E normalization can make the sum
        # deviate slightly from 1; renormalize/clip downstream if you rely on strict normalization.
        kernel = (belief @ self.T[action]) @ self.E[action]
        kernel = np.clip(kernel, 0.0, None)
        s = kernel.sum()
        if s > 0:
            kernel = kernel / s
        else:
            kernel = np.full(self.O, 1.0 / self.O)
        return kernel


    def compute_belief(self, history: List[Tuple[int, int]]) -> np.ndarray:
        belief = self.initial_belief.copy()
        for obs, act in history:
            b_pred = self.T[act, :, :].T @ belief
            b_new = self.E[act, :, obs] * b_pred
            norm = b_new.sum()
            belief = b_new / norm if norm > 0 else self.initial_belief.copy()

        return belief


class PretrainedPOMDPAgent:
    """
    Agent that decides between two labels (action_a/action_b) based on 
    whether its learned kernel's CDF at a threshold is >= 0.5.
    """
    def __init__(self, env):
        self.env = env
        # Key: (tuple(history), action), Value: np.ndarray (learned kernel)
        self.learned_kernel_cache = {}

    def set_knowledge(self, history: List[Tuple[int, int]], action: int, kernel: np.ndarray):
        """Inject the learned kernel into the agent's memory."""
        self.learned_kernel_cache[(tuple(history), action)] = kernel

    def get_kernel_estimate(self, history: List[Tuple[int, int]], action: int) -> np.ndarray:
        """Retrieve cached kernel or raise error."""
        key = (tuple(history), action)
        if key not in self.learned_kernel_cache:
            raise ValueError(f"No knowledge for history {history} and action {action}")
        return self.learned_kernel_cache[key]

    def optimal_action_for_goal(self, history: List[Tuple[int, int]],
                               action_a: int, action_b: int,
                               target_obs: int, n_trials: int,
                               threshold: int) -> int:
        """
        Decision logic: Compares the probability of the outcome being <= threshold 
        against the probability of it being > threshold.
        """
        # Note: We query the kernel for action_a specifically to extract its belief
        kernel = self.get_kernel_estimate(history, action_a)
        p = kernel[target_obs]

        prob_a = binom.cdf(threshold, n_trials, p)
        prob_b = 1 - prob_a

        return action_a if prob_a >= prob_b else action_b
