import numpy as np
from collections import defaultdict
from utils import forward_algorithm_scaled, backward_algorithm_scaled, compute_xi_gamma, m_step

"""
Markov Models
"""
# Order 0 Markov Model
def order_0_mm(seq):
    counts = defaultdict(int)
    total = len(seq)
    
    # Count nucleotide frequencies
    for nucleotide in seq:
        counts[nucleotide] += 1
    
    # Calculate probabilities
    probabilities = {nucleotide: count/total for nucleotide, count in counts.items()}
    
    return probabilities


# Order 1 Markov Model Implementation
def order_1_mm(seq):
    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    # Count nucleotide transition frequencies
    for i in range(len(seq) - 1):
        prefix, next_nucleotide = seq[i], seq[i+1]
        counts[prefix][next_nucleotide] += 1
        total_counts[prefix] += 1
    
    # Calculate conditional probabilities
    probabilities = {
        prefix: {next_nucleotide: count/total_counts[prefix] for next_nucleotide, count in next_nucleotide_counts.items()}
        for prefix, next_nucleotide_counts in counts.items()
    }
    
    return probabilities


# Order 2 Markov Model Implementation
def order_2_mm(seq):
    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    # Count nucleotide transition frequencies for order 2
    for i in range(len(seq) - 2):
        prefix, next_nucleotide = seq[i:i+2], seq[i+2]
        counts[prefix][next_nucleotide] += 1
        total_counts[prefix] += 1
    
    # Calculate conditional probabilities
    probabilities = {
        prefix: {next_nucleotide: count/total_counts[prefix] for next_nucleotide, count in next_nucleotide_counts.items()}
        for prefix, next_nucleotide_counts in counts.items()
    }
    
    return probabilities


"""
Hidden Markov Models
"""
# HMM Class
class HMM:
    def __init__(self, states, symbols, transition_probs=None, emission_probs=None):
        self.states = states  # Hidden states
        self.symbols = symbols  # Observable symbols
        
        # If transition and emission probabilities are not provided, 
        # initialize with uniform distributions
        if transition_probs is None:
            self.transition_probs = np.ones((len(states), len(states))) / len(states)
        else:
            self.transition_probs = transition_probs  # Transition probabilities
            
        if emission_probs is None:
            self.emission_probs = np.ones((len(states), len(symbols))) / len(symbols)
        else:
            self.emission_probs = emission_probs  # Emission probabilities


# Randomly initialized HMM
def random_initialize_hmm(num_states, num_symbols):
    """
    Randomly initialize transition and emission probabilities for the HMM.
    """
    states_candidates = ["st1", "st2", "st3", "st4"]

    assert num_states <= len(states_candidates), "num_states > 4"
    assert num_symbols == 4, "num_symbols must be 4"
    # Randomly initialize transition probabilities
    transition_probs = np.random.rand(num_states, num_states)
    transition_probs /= transition_probs.sum(axis=1, keepdims=True)
    
    # Randomly initialize emission probabilities
    emission_probs = np.random.rand(num_states, num_symbols)
    emission_probs /= emission_probs.sum(axis=1, keepdims=True)
    
    return HMM(states=states_candidates[:num_states], symbols=['A', 'C', 'G', 'T'], 
               transition_probs=transition_probs, emission_probs=emission_probs)


# Baum-Welch with random initializations
def baum_welch_with_random_initializations(sequence, num_states, num_symbols, num_initializations, num_iterations):
    """
    Run Baum-Welch algorithm with multiple random initializations.
    """
    best_hmm = None
    best_log_prob = float('-inf')
    
    for num_init in range(num_initializations):
        print(f"run {num_init+1}")
        # Randomly initialize the HMM
        hmm = random_initialize_hmm(num_states, num_symbols)
        
        # Run Baum-Welch for a fixed number of iterations
        for num_iter in range(num_iterations):
            # E-step
            alpha, scaling_factors, _ = forward_algorithm_scaled(hmm, sequence)
            beta = backward_algorithm_scaled(hmm, sequence, scaling_factors)
            xi, gamma = compute_xi_gamma(hmm, sequence, alpha, beta)
            
            # M-step
            hmm = m_step(hmm, sequence, xi, gamma)
        
        # Compute log probability of the sequence for the final model
        _, _, log_prob = forward_algorithm_scaled(hmm, sequence)
        
        # Update the best model if current one is better
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_hmm = hmm
            print("Improved!")
    
    return best_hmm, best_log_prob