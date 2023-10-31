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


# Baum-Welch algorithm
def baum_welch_with_log_likelihood(hmm, sequence, max_iterations=100, convergence_threshold=1e-6):
    """
    Run the Baum-Welch algorithm with convergence based on the change in log likelihood.
    
    :param hmm: Initial HMM model
    :param sequence: Observed sequence
    :param max_iterations: Maximum number of iterations
    :param convergence_threshold: Threshold for log likelihood change to consider as convergence
    :return: Trained HMM model, log probability of the sequence, number of iterations
    """
    prev_log_prob = None
    
    for num_iter in range(max_iterations):
        # E-step: Using the user-provided forward algorithm and the modified backward algorithm
        alpha, _, _ = forward_algorithm_scaled(hmm, sequence)
        beta, _, _ = backward_algorithm_scaled(hmm, sequence)
        xi, gamma = compute_xi_gamma(hmm, sequence, alpha, beta)
        
        # M-step
        hmm = m_step(hmm, sequence, xi, gamma)
        
        # Compute log probability of the sequence for the current model
        _, _, log_prob = forward_algorithm_scaled(hmm, sequence)
        
        # Check for convergence if prev_log_prob is not None
        if prev_log_prob is not None:
            log_prob_change = abs(log_prob - prev_log_prob)
            print(f"Iter {num_iter+1}:{log_prob_change}")
            if log_prob_change < convergence_threshold:
                break
        
        # Update previous log probability for the next iteration
        prev_log_prob = log_prob
    
    return hmm, log_prob, num_iter + 1

