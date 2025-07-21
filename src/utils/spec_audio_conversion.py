import librosa
import numpy as np
from src.utils.utils import crop_pad_audio
import torch




def convert_to_complex_spectrogram_with_compression_torch(audio, sr=16000, window_size=400, hop_size=160, n_fft=512, p=0.3, compressed=True):
    """
    Convert to complex spectrogram after power compression only on magnitude
    Phase is in radians
    """

    window = torch.hann_window(window_size, device=audio.device)
    spec = torch.stft(audio, n_fft = n_fft, win_length=window_size, hop_length=hop_size, window=window, center=True, return_complex=True)
    mag_spec = torch.abs(spec)
    phase = torch.angle(spec)
    
    if compressed:
        mag_spec = torch.clip(mag_spec, 1e-8, None)
        mag_spec = torch.pow(mag_spec, p)
    
    return (mag_spec, phase)




def convert_to_audio_from_complex_spectrogram_after_compression_torch(complex_spec: tuple, sr=16000, window_size=400, hop_size=160, n_fft=512, p=0.3, compressed=True):
    """
    Take the complex spectrogram and convert to audio.
    Assume complex spectrogram is compressed.
    complex_spec has dexpected dimension (frames, bins)
    Note - mag_spec has been clipped to minimum of 1e-8 prior to power compression
    """
    mag_spec, phase = complex_spec
    mag_spec = mag_spec.T
    phase = phase.T
    if compressed:
        mag_spec = torch.pow(mag_spec, 1/p)
    spec = mag_spec * torch.exp(1j*phase)
    spec = spec.to(torch.device("cuda"))
    spec = spec.permute(2, 0, 1)
    window = torch.hann_window(window_size, device=spec.device)
    audio = torch.istft(spec, n_fft=n_fft, hop_length=hop_size, win_length=window_size, window=window, center=True, length=sr*5)
    return audio
    


def get_padded_phase(audio, sr=16000, window_size=400, hop_size=160, n_fft=512, p=0.3):
    """
    Get the phase of the audio after normalizing and padding
    """

    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=window_size, window="hann", center=False)
    phase = np.angle(spec)
    # pad phase from 497 frames to 500 frames
    phase = np.pad(phase, ((0, 0), (0, 3)))
    return phase

