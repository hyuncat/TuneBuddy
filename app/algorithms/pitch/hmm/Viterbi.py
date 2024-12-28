import numpy as np
from typing import Optional, Tuple

def viterbi(observations, states, initial_matrix, transition_matrix, emission_matrix):
    """
    Viterbi algorithm for finding the most likely sequence of hidden states
    given a sequence of observations and the HMM model parameters.

    Args:
        observations (np.ndarray): 2D matrix of observations (rows = time steps, columns = features)
        states (np.ndarray): 1D array of all possible hidden states
        initial_matrix (np.ndarray): Initial state probabilities
        transition_matrix (np.ndarray): Transition probabilities between states
        emission_matrix (np.ndarray): Emission probabilities of observations from states

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the most likely sequence of hidden states
        and the probability of the most likely sequence
    """
    # Initialize variables
    num_observations = len(observations)
    num_states = len(states)
    viterbi_matrix = np.zeros((num_states, num_observations))
    backpointer_matrix = np.zeros((num_states, num_observations))

    # Initialize the first column of the Viterbi matrix
    viterbi_matrix[:, 0] = initial_matrix * emission_matrix[:, observations[0]]

    # Iterate over the rest of the observations
    for t in range(1, num_observations):
        for s in range(num_states):
            # Calculate the probabilities of transitioning to the current state
            # from all other states at the previous time step
            transition_probabilities = viterbi_matrix[:, t - 1] * transition_matrix[:, s]

            # Find the maximum probability and corresponding backpointer
            viterbi_matrix[s, t] = np.max(transition_probabilities) * emission_matrix[s, observations[t]]
            backpointer_matrix[s, t] = np.argmax(transition_probabilities)

    # Find the most likely final state
    final_state = np.argmax(viterbi_matrix[:, num_observations - 1])

    # Backtrack to find the most likely sequence of states
    state_sequence = np.zeros(num_observations, dtype=int)
    state_sequence[-1] = final_state
    # Iterate backwards from the 2nd-to-last observation --> 1st observation
    for t in range(num_observations-2, -1, -1): 
        state_sequence[t] = backpointer_matrix[state_sequence[t + 1], t + 1]

    # Calculate the probability of the most likely sequence
    max_probability = np.max(viterbi_matrix[:, num_observations - 1])

    return state_sequence, max_probability, viterbi_matrix, backpointer_matrix


def viterbi2(transition_matrix, emission_matrix):
    """
    Viterbi algorithm for finding the most likely sequence of hidden states
    given a sequence of observations and the HMM model parameters.

    Args:
        observations (np.ndarray): 2D matrix of observations (rows = time steps, columns = features)
        states (np.ndarray): 1D array of all possible hidden states
        initial_matrix (np.ndarray): Initial state probabilities
        transition_matrix (np.ndarray): Transition probabilities between states
        emission_matrix (np.ndarray): Emission probabilities of observations from states

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the most likely sequence of hidden states
        and the probability of the most likely sequence
    """
    # Initialize variables
    n_frames = np.shape(emission_matrix)[1]
    n_bins = np.shape(emission_matrix)[0]

    viterbi_matrix = np.zeros((n_bins, n_frames))
    backpointer_matrix = np.zeros((n_bins, n_frames))

    # Initialize the first column of the Viterbi matrix
    viterbi_matrix[:, 0] = emission_matrix[:, 0]

    # Iterate over the rest of the observations
    for t in range(1, n_frames):
        for s in range(n_bins):
            # Calculate the probabilities of transitioning to the current state
            # from all other states at the previous time step
            transition_probabilities = viterbi_matrix[:, t - 1] * transition_matrix[:, s]

            # Find the maximum probability and corresponding backpointer
            viterbi_matrix[s, t] = np.max(transition_probabilities) * emission_matrix[s, t]
            backpointer_matrix[s, t] = np.argmax(transition_probabilities)

    # Find the most likely final state
    final_state = np.argmax(viterbi_matrix[:, n_frames - 1])

    # Backtrack to find the most likely sequence of states
    state_sequence = np.zeros(n_frames, dtype=int)
    state_sequence[-1] = final_state
    # Iterate backwards from the 2nd-to-last observation --> 1st observation
    for t in range(n_frames-2, -1, -1): 
        state_sequence[t] = backpointer_matrix[state_sequence[t + 1], t + 1]

    # Calculate the probability of the most likely sequence
    max_probability = np.max(viterbi_matrix[:, n_frames - 1])

    return state_sequence, max_probability
    # return state_sequence, max_probability, viterbi_matrix, backpointer_matrix