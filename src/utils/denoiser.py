from df.enhance import enhance, init_df
import torch
import soundfile as sf


from torchaudio.transforms import Resample


def denoise_speech(noisy_audio, orig_sr=16000, crop_length=5):  
    # noisy_audio = librosa.resample(noisy_audio, orig_sr = orig_sr, target_sr=48000)
    resampler  = Resample(orig_sr, 48000, dtype = noisy_audio.dtype).to(noisy_audio.device)
    # breakpoint()
    noisy_audio = resampler(noisy_audio)
    
    # noisy_audio = torch.tensor(noisy_audio).unsqueeze(0)
    # initialize denoiser model
    model, df_state, _ = init_df(log_level='ERROR')
    # print(f"Checking model's device: {model.device}")
    enhanced_audio = enhance(model, df_state, noisy_audio.cpu())
    enhanced_audio = enhanced_audio.to(noisy_audio.device)
    downsampler = Resample(48000, orig_sr, dtype = enhanced_audio.dtype).to(enhanced_audio.device)
    enhanced_audio = downsampler(enhanced_audio)

    return enhanced_audio


if __name__ == "__main__":

    
    import os
    orig_sr = 16000
    

    
