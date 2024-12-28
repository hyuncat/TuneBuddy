
class PYIN:
    def __init__(self):
        pass

    @staticmethod
    def pyin(audio_data: np.ndarray, mean_threshold: float=0.3, 
             sr: int=44100, fmin: int=196, fmax: int=3000):
        
        audio_data = Filter.high_pass_iir_filter(audio_data, cutoff_freq=fmin)
        
        # Config variables
        FRAME_SIZE = 2048
        HOP_SIZE = 128

        N_THRESHOLDS = 100
        thresholds = np.linspace(0, 1, N_THRESHOLDS) 
        cdf_thresholds = np.linspace(0, 1, N_THRESHOLDS + 1) # add one because we create using np.diff of a cdf

        # Create beta distribution centered around the desired mean threshold
        alpha, beta = PYin.calculate_alpha_beta(mean_threshold, total=20)
        beta_cdf = scipy.stats.beta.cdf(x=cdf_thresholds, a=alpha, b=beta) # How are alpha and beta calculated?
        beta_pdf = np.diff(beta_cdf) # where we know the total mass = 1

        # Prepare variables for frame iteration
        pitches = []
        most_likely_pitches = []
        volumes = []
        
        pitch_config = PitchConfig( # Defines resolution of pitch bins
            bins_per_semitone=10, tuning=440.0, fmin=fmin, fmax=fmax
        )
        num_frames = (len(audio_data) - FRAME_SIZE) // HOP_SIZE

        for frame_idx in range(num_frames):
            # Print the frame count in place
            print(f"\rProcessing frame {frame_idx + 1}/{num_frames}", end='')

            i = frame_idx*HOP_SIZE
            time = i/sr

            # Compute the CMNDF function for each frame
            audio_frame = audio_data[i:i+FRAME_SIZE]

            volume = np.sqrt(np.mean(audio_frame ** 2)) # Computed with

            cmndf_frame, power_spec, amplitudes = Yin.cmndf(audio_frame, FRAME_SIZE//2)
            
            # Compute all possible pitch candidates for the frame
            frequencies, probabilities = PYin.probabilistic_thresholding(cmndf_frame, thresholds, beta_pdf)

            for freq, prob in zip(frequencies, probabilities):
                pitch = Pitch(time=time, frequency=freq, probability=prob, volume=volume, config=pitch_config)
                pitches.append(pitch)

            # Append the most likely pitch candidate for the frame to a separate list
            i = np.argmax(probabilities)
            best_prob = probabilities[i]
            best_freq = frequencies[i]
            most_likely_pitch = Pitch(time=time, frequency=best_freq, probability=best_prob, volume=volume, config=pitch_config)

            most_likely_pitches.append(most_likely_pitch)
            
        print("\nDone!")
        return pitches, most_likely_pitches

    # @staticmethod
    # def pyin2(audio_data: AudioData, mean_threshold: float=0.3, 
    #          fmin: int=196, fmax: int=3000):
    #     """
    #     The Probabilistic YIN algorithm for pitch estimation.

    #     Args:
    #         audio_data (AudioData): The audio data containing user pitches to process.
    #         mean_threshold (float): The desired mean threshold for the beta distribution. What we expect the threshold to be.
    #         fmin (int): The minimum frequency to consider in pitch estimation. (for high-pass filtering + pitch bins)
    #         fmax (int): The maximum frequency to consider in pitch estimation. (for pitch bins)

    #     Returns:
    #         Tuple[list[Pitch], list[Pitch]]: Lists of all pitch candidates and the most likely pitch candidates.
    #     """

    #     print("Starting pYIN algorithm...")

    #     # Config variables
    #     sr = 44100
    #     FRAME_SIZE = 2048
    #     HOP_SIZE = 128
        
    #     audio_data = audio_data.data
    #     # High-pass filter the audio data
    #     audio_data = Filter.high_pass_iir_filter(audio_data, cutoff_freq=fmin)

    #     N_THRESHOLDS = 100
    #     thresholds = np.linspace(0, 1, N_THRESHOLDS) 
    #     cdf_thresholds = np.linspace(0, 1, N_THRESHOLDS + 1) # add one because we create using np.diff of a cdf

    #     # Create beta distribution centered around the desired mean threshold
    #     alpha, beta = PYin.calculate_alpha_beta(mean_threshold, total=20)
    #     beta_cdf = scipy.stats.beta.cdf(x=cdf_thresholds, a=alpha, b=beta) # How are alpha and beta calculated?
    #     beta_pdf = np.diff(beta_cdf) # where we know the total mass = 1

    #     # Prepare variables for frame iteration
    #     pitches = []
    #     most_likely_pitches = []
    #     voiced_probs = []

    #     pitch_config = PitchConfig( # Defines resolution of pitch bins
    #         bins_per_semitone=10, tuning=440.0, fmin=fmin, fmax=fmax
    #     )
    #     num_frames = (len(audio_data) - FRAME_SIZE) // HOP_SIZE

    #     # 2M pitch_bins for both voiced and unvoiced states
    #     n_pitch_bins = int(pitch_config.n_pitch_bins)
    #     emission_matrix = np.zeros((n_pitch_bins*2, len(audio_data) // HOP_SIZE))
    #     init_voicing_probs = {
    #         "voiced": 0.5,
    #         "unvoiced": 0.5
    #     }
    #     max_transition_rate = 135.92  # Octaves per second
    #     transition_width = round(max_transition_rate * 12 * HOP_SIZE / sr) * pitch_config.bins_per_semitone + 1
    #     switch_prob = 0.01

    #     # Construct the transition matrix using the Kronecker product approach
    #     transition_matrix = PYin.transition_matrix(n_pitch_bins, transition_width, switch_prob)

    #     for frame_idx in range(num_frames):
    #         # Print the frame count in place
    #         print(f"\rProcessing frame {frame_idx + 1}/{num_frames}", end='')

    #         i = frame_idx*HOP_SIZE
    #         time = i/sr

    #         # Compute the CMNDF function for each frame
    #         audio_frame = audio_data[i:i+FRAME_SIZE]
    #         cmndf_frame, power_spec = Yin.cmndf(audio_frame, FRAME_SIZE//2)
            
    #         # Compute all possible pitch candidates for the frame
    #         frequencies, probabilities = PYin.probabilistic_thresholding(cmndf_frame, thresholds, beta_pdf)

    #         # max_prob = 0
    #         most_likely_pitch = None

    #         for freq, prob in zip(frequencies, probabilities):
    #             pitch = Pitch(time=time, frequency=freq, probability=prob, config=pitch_config)
    #             pitches.append(pitch)

    #         # Get the voiced probability for each frame as the sum of all pitch probabilities
    #         voiced_prob = np.clip(np.sum(probabilities), 0, 1)
    #         assert 0 <= voiced_prob <= 1
    #         voiced_probs.append(voiced_prob)

    #         # Create the sparse vector for the frame
    #         for j, (freq, prob) in enumerate(zip(frequencies, probabilities)):
    #             pitch = Pitch(time=time, frequency=freq, probability=prob, config=pitch_config)
    #             bin_idx = pitch.bin_index
    #             emission_matrix[bin_idx, frame_idx] = prob
            
    #         emission_matrix[n_pitch_bins:, frame_idx] = (1-voiced_prob) / n_pitch_bins

    #         # Append the most likely pitch candidate for the frame to a separate list
    #         i = np.argmax(probabilities)
    #         best_prob = probabilities[i]
    #         best_freq = frequencies[i]
    #         most_likely_pitch = Pitch(time=time, frequency=best_freq, probability=best_prob, config=pitch_config)

    #         most_likely_pitches.append(most_likely_pitch)
            
    #     print("\nDone!")
    #     return pitches, most_likely_pitches
    #     # return pitches, most_likely_pitches, voiced_probs, emission_matrix, transition_matrix

    # def transition_matrix(n_pitch_bins: int, transition_width, switch_prob, window="triangle"):
    #     """
    #     Construct the transition matrix using the Kronecker product approach.
    #     This method creates a local pitch transition matrix and combines it with the voicing state transitions.
    #     """

    #     # Create a local pitch transition matrix (for within voiced/unvoiced states)
    #     local_transition = np.zeros((n_pitch_bins, n_pitch_bins))

    #     for i in range(n_pitch_bins):
    #         start = max(0, i - transition_width // 2)
    #         end = min(n_pitch_bins, i + transition_width // 2 + 1)
    #         local_transition[i, start:end] = np.linspace(1, 0, end - start)

    #     # Normalize the rows of the local transition matrix
    #     local_transition /= local_transition.sum(axis=1, keepdims=True)

    #     # Create the voicing transition matrix (2x2 matrix for voiced/unvoiced)
    #     t_switch = np.array([[1 - switch_prob, switch_prob], [switch_prob, 1 - switch_prob]])

    #     # Use Kronecker product to combine the voicing and local pitch transitions
    #     full_transition_matrix = np.kron(t_switch, local_transition)

    #     return full_transition_matrix