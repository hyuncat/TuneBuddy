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


def parse_viterbi_index(v_idx, n_pitch_bins):
    """Parses the row index of a 2d viterbi dp matrix and returns
    the pitch bin and the voicing decision for the pitch.

    Returns:
        pitch_bin (int): index of the raw pitch bin
        is_voiced (bool): if the index is > (unv) or < (v) n_pitch_bins
    """
    if v_idx < n_pitch_bins:
        return v_idx, True

    else:
        return int(v_idx - n_pitch_bins), False


def trans_prob(p_i, v_i, p_j, v_j):
    """ 
    Return transition probability of going from note_i --to--> note_j,
    where each note contains a pitch bin value and a voicing decision.

    Transition is computed to minimize pitch jumps and voicing changes,
    and independence is assumed between pitch jump and voicing change for final product.
    """
    is_pitch_jump = abs(p_i - p_j) > 25
    is_voice_jump = v_i != v_j

    PITCH_JUMP_PROB = .2
    VOICE_JUMP_PROB = .01

    p = PITCH_JUMP_PROB if is_pitch_jump else 1-PITCH_JUMP_PROB
    v = VOICE_JUMP_PROB if is_voice_jump else 1-VOICE_JUMP_PROB

    return p*v


@staticmethod
def viterbi3(obs_mat: np.ndarray):
    """
    Decoding a path of pitches through the original PYIN frequency estimates.
    Uses a modified version of the viterbi algorithm.

    Returns:
        best_path (list): best pitch voicing sequence
        best_path_prob (float): final probability
    """
    N, T = obs_mat.shape
    viterbi_mat = np.full((N, T), -np.inf) # stores probabilities in log space
    log_obs_mat = np.log(obs_mat + 1e-10) # add small offset to avoid log(0)
    backpointer_mat = np.zeros((N, T), dtype=int)

    # init first column of viterbi matrix
    # init prob = uniform across nonzero pitches
    init_n_states = len(np.nonzero(obs_mat[:, 0])[0])
    viterbi_mat[:, 0] = log_obs_mat[:, 0] + np.log(1/init_n_states)

    # loop through all time steps in observation matrix
    for t in range(1, T-1):

        # iterate through all nonzero pitch candidates at time=t
        curr_states = np.nonzero(obs_mat[:, t])[0]
        for i in curr_states:
            obs_prob = log_obs_mat[i, t] # get log obs_prob instead, @ same index

            # get the previous state values (log-ed) and their indices
            prev_states = np.where(viterbi_mat[:, t-1] > -np.inf)[0]
            v_prevs = np.array([viterbi_mat[v_prev, t-1] for v_prev in prev_states])

            best_path_prob, best_v_prev_idx = v_prevs[0], prev_states[0]
            for v_idx, v_prev in zip(prev_states, v_prevs):
                p_i, v_i = parse_viterbi_index(v_idx, int(N/2))
                p_j, v_j = parse_viterbi_index(i, int(N/2))
                
                # based on pitch jump / voicing / octave?
                tr_prob = trans_prob(p_i, v_i, p_j, v_j)
                log_tr_prob = np.log(tr_prob) + 1e-10
                path_prob = v_prev + log_tr_prob + obs_prob

                # update best path found if better than prev
                if path_prob > best_path_prob:
                    best_path_prob = path_prob
                    best_v_prev_idx = v_idx

            # update viterbi / backpointer matrices
            viterbi_mat[i, t] = best_path_prob
            backpointer_mat[i, t] = best_v_prev_idx

    # termination step
    best_path_pointer = np.argmax(viterbi_mat[:, T-1])
    best_path_prob = viterbi_mat[best_path_pointer, T-1]

    # decode the best path
    best_path = [best_path_pointer]
    for t in range(T-1, 0, -1): # backwards loop
        best_path_pointer = backpointer_mat[best_path_pointer, t]
        best_path.append(best_path_pointer)
    best_path.reverse() # reverse the path

    return best_path, np.exp(best_path_prob) # return to non-log space for final path prob