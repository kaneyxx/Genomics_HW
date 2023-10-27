import argparse
import pandas as pd
from models import order_0_mm, order_1_mm, order_2_mm, \
    HMM, random_initialize_hmm, baum_welch_with_random_initializations, forward_algorithm_scaled
from utils import extract_and_process_sequence, log2_prob_markov_chain

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="GRCh38_latest_genomic.fna")
parser.add_argument('--target', type=str, default="NC_000006.12")
parser.add_argument('--start', type=int, default=100000)
parser.add_argument('--end', type=int, default=1200000)
parser.add_argument('--model', type=str, default="mm", help="mm or hmm")
parser.add_argument('--num_state', type=int, default=2, help="#states for hmm")
parser.add_argument('--num_symbol', type=int, default=4, help="#symbols for hmm")
parser.add_argument('--num_init', type=int, default=2, help="#initializations for hmm")
parser.add_argument('--num_iter', type=int, default=10, help="#iterations for hmm")
parser.add_argument('--test', action="store_true", help="for testing")
parser.add_argument('--test_target', type=str, default="NC_000007.14")

if __name__=="__main__":
    args = parser.parse_args()

    # Extract and process the sequence for Homo sapiens chromosome 6
    sequences = extract_and_process_sequence(args.file, 
                                             args.target, 
                                             args.start, 
                                             args.end)
    
    print(f"Extracted sequence (first 20 nucleotides):{sequences[0][:20]}")
    print(f"Extracted sequence (last 20 nucleotides):{sequences[0][-20:]}")

    S = sequences[0]

    if args.model=="mm":
        print("\nApplying order 0 Markov model to extracted sequence")
        order_0_prob = order_0_mm(S)
        print("Probabilities of order 0 Markov model:")
        print(pd.DataFrame([order_0_prob]))

        print("\nApplying order 1 Markov model to extracted sequence")
        order_1_prob = order_1_mm(S)
        print("Probabilities of order 1 Markov model:")
        print(pd.DataFrame(order_1_prob))
        
        print("\nApplying order 2 Markov model to extracted sequence")
        order_2_prob = order_2_mm(S)
        print("Probabilities of order 2 Markov model:")
        print(pd.DataFrame(order_2_prob))


        log_prob_0, log_prob_1, log_prob_2 = log2_prob_markov_chain(S, 
                                                                    order=-1, 
                                                                    prob_0=order_0_prob,
                                                                    prob_1=order_1_prob,
                                                                    prob_2=order_2_prob)
        print("Log2 probability of each Markov model:")
        print(f"Order 0:{log_prob_0}")
        print(f"Order 1:{log_prob_1}")
        print(f"Order 2:{log_prob_2}")
    
    elif args.model=="hmm":
        print("\nApplying random initialized hidden Markov model to extracted sequence")
        # Start with an initial random HMM model
        initial_hmm = random_initialize_hmm(num_states=args.num_state, 
                                            num_symbols=args.num_symbol)
        _, _, initial_log2_prob_S = forward_algorithm_scaled(initial_hmm, S)
        print(f"Log2 probability of HMM:{initial_log2_prob_S}")


        # We will use the baum_welch_with_random_initializations function to run the 
        # Baum-Welch algorithm multiple times with random initializations to get the 
        # best HMM model for S.
        print("\nApplying Baum-Welch algorithm to extracted sequence")
        optimized_hmm, optimized_log2_prob_S = baum_welch_with_random_initializations(S, 
                                                                    num_states=args.num_state, 
                                                                    num_symbols=args.num_symbol, 
                                                                    num_initializations=args.num_init, 
                                                                    num_iterations=args.num_iter)
        
        print(f"Optimized log2 probability:{optimized_log2_prob_S}")
        print(f"Optimized transition probabilities:\n{pd.DataFrame(optimized_hmm.transition_probs)}")
        print(f"Optimized emission probabilities:\n{pd.DataFrame(optimized_hmm.emission_probs)}")
        
        # Apply optimized parameters to test sequence segment
        if args.test:
            print("\n Test section")
            test_sequences = extract_and_process_sequence(args.file, 
                                                          args.test_target, 
                                                          args.start, 
                                                          args.end)
            test_S = test_sequences[0]

            _, _, log2_prob_chromosome_7_segment = forward_algorithm_scaled(optimized_hmm, test_S)
            print(f"Log2 probability of test sequence:{log2_prob_chromosome_7_segment}")

