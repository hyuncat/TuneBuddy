import numpy as np

class PYinHMM:
    def __init__(self):
        pass

    def transition_matrix(n_pitch_bins, transition_width, switch_prob, window="triangle"):
        """
        Construct the transition matrix using the Kronecker product approach.
        This method creates a local pitch transition matrix and combines it with the voicing state transitions.
        """

        # Create a local pitch transition matrix (for within voiced/unvoiced states)
        local_transition = np.zeros((n_pitch_bins, n_pitch_bins))

        for i in range(n_pitch_bins):
            start = max(0, i - transition_width // 2)
            end = min(n_pitch_bins, i + transition_width // 2 + 1)
            local_transition[i, start:end] = np.linspace(1, 0, end - start)

        # Normalize the rows of the local transition matrix
        local_transition /= local_transition.sum(axis=1, keepdims=True)

        # Create the voicing transition matrix (2x2 matrix for voiced/unvoiced)
        t_switch = np.array([[1 - switch_prob, switch_prob], [switch_prob, 1 - switch_prob]])

        # Use Kronecker product to combine the voicing and local pitch transitions
        full_transition_matrix = np.kron(t_switch, local_transition)

        return full_transition_matrix