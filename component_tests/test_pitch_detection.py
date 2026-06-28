# test_pitch_detection.py
import pytest
import pandas as pd
import librosa
import numpy as np
from algorithms.PitchDetector import PitchDetector
from algorithms.PitchSmoother import PitchSmoother
from algorithms.Config import Config

@pytest.fixture
def config():
    return Config(fmin=96, fmax=270)

def test_bassoon_pitch_accuracy(config):
    audio, sr = librosa.load("component_tests/fixtures/Bach10-mf0-synth/audio_stems/01_AchGottundHerr_bassoon.RESYN.wav", sr=config.sr)
    ground_truth = pd.read_csv("component_tests/fixtures/Bach10-mf0-synth/annotation_stems/01_AchGottundHerr_bassoon.RESYN.csv",
                                 header=None, names=["time", "f0"])
    
    detector = PitchDetector(config=config)
    smoother = PitchSmoother(config=config)
    raw = detector.detect_pitches(audio)
    smoothed = smoother.smooth(raw)

    non_zero_gt = ground_truth[ground_truth["f0"] > 0]
    gt_min= non_zero_gt["f0"].min()
    gt_max = non_zero_gt["f0"].max()
    print(f"Ground truth min: {gt_min}, max: {gt_max}")

    #dict from time to array of (detected pitch, ground truth pitch, error in cents)
    errors = {}

    for pitch in smoothed:
        if pitch is None or pitch.unvoiced_prob >= config.pitch_thresh or not pitch.candidates:
            continue
        # find the closest ground truth row by time
        closest = non_zero_gt.iloc[(non_zero_gt["time"] - pitch.time).abs().argmin()]
        gt_f0 = closest["f0"]
        detected_f0 = config.midi_to_freq(pitch.candidates[0][0])
        error_cents = abs(1200 * np.log2(detected_f0 / gt_f0))
        if error_cents <50:
            continue
        errors[pitch.time] = (detected_f0, gt_f0, error_cents)

    print(f"Total frames with detected pitch: {len(errors)}")
    print("At time (seconds): Detected f0 (Hz), Ground truth f0 (Hz), Error (cents)")
    for time, (detected, gt, error) in errors.items():
        print(f"  {time:.2f}: {detected:.1f}, {gt:.1f}, {error:.1f}")
