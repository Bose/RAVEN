import numpy as np
import librosa


def crop_pad_audio(fp, sr, crop_length):

    audio, _ = librosa.load(fp, sr=sr, duration=5)

    if len(audio) < sr * crop_length:
        audio = np.pad(audio, (0, sr * crop_length - len(audio)))

    return audio

def apply_gain(audio, seed, gain_lower_dB=-10, gain_upper_dB=10):
    if seed != None:
        np.random.seed(seed)
    dB_gain = np.random.uniform(gain_lower_dB, gain_upper_dB)
    gain_factor = 10**(dB_gain/20)
    return audio * gain_factor, gain_factor
