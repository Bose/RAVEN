import pandas as pd
import numpy as np
from src.utils.utils import crop_pad_audio
import librosa
import hashlib
import os
import soundfile as sf
from tqdm import tqdm

SAVE_TEST_FP = False


def random_select_fps(df, n=1000):
    
    df = df[df["split"] == "test"].sample(n, random_state=42, replace=False)["audio_fp"]
    if SAVE_TEST_FP:
        df.to_csv("./data/VoxCeleb2_test_1000_fps.txt", index=False)
    
    return df




def generate_noise_condition(target_fp, noise_df, snr, orig_sr=16000, crop_length=5):
    
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    noise_fp = noise_df.sample(1, random_state=seed)["filepath"].values[0]
    noise_fp = os.path.join("/data/MUSAN", noise_fp)
    noise_audio = crop_pad_audio(noise_fp, orig_sr, crop_length)
    noise_audio = librosa.util.normalize(noise_audio)
    
    mixed_audio = mix_per_snr(target_audio, noise_audio, snr)
    
    return mixed_audio


def generate_one_interfering_speaker_condition(target_fp, subset_test_df, snr, orig_sr=16000, crop_length=5):
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    # generate seed based on target_fp
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    interfering_speaker_fp = subset_test_df.sample(1, random_state=seed)["audio_fp"].values[0]
    interfering_speaker_audio = crop_pad_audio(interfering_speaker_fp, orig_sr, crop_length)
    interfering_speaker_audio = librosa.util.normalize(interfering_speaker_audio)
    
    mixed_audio = mix_per_snr(target_audio, interfering_speaker_audio, snr)
    
    return mixed_audio

def generate_three_interfering_speakers_condition(target_fp, subset_test_df, snr, orig_sr=16000, crop_length=5):
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    # generate seed based on target_fp
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    interfering_speaker_fp = subset_test_df.sample(3, random_state=seed)["audio_fp"].values
    interfering_speaker_audio = [crop_pad_audio(fp, orig_sr, crop_length) for fp in interfering_speaker_fp]
    interfering_speaker_audio = [librosa.util.normalize(audio) for audio in interfering_speaker_audio]
    interfering_speaker_audio = np.sum(interfering_speaker_audio, axis=0)
    assert len(interfering_speaker_audio) == len(target_audio)
    
    mixed_audio = mix_per_snr(target_audio, interfering_speaker_audio, snr)
    
    return mixed_audio
    




def main(condition, snr):
    musan_fps = pd.read_csv("./data/musan_split.csv")
    main_df = pd.read_csv("./data/split.csv")
    subset_test_df = main_df[main_df["split"] == "test"]
    target_test_df = random_select_fps(main_df)
    musan_noise_test_df = musan_fps[(musan_fps["split"] == "test") & (musan_fps["type"] == "noise")]
    
    for i, target_fp in tqdm(enumerate(target_test_df)):
        if condition == "noise_only":
            mixed_audio = generate_noise_condition(target_fp, musan_noise_test_df, snr)
        elif condition == "one_interfering_speaker":
            mixed_audio = generate_one_interfering_speaker_condition(target_fp, subset_test_df, snr)
        elif condition == "three_interfering_speakers":
            mixed_audio = generate_three_interfering_speakers_condition(target_fp, subset_test_df, snr)
        
        mixed_audio_fp = target_fp.replace("/aac/", f"/mixed_wav/{condition}/{snr}/").replace(".m4a", ".wav")
        # check if directory exists
        os.makedirs(os.path.dirname(mixed_audio_fp), exist_ok=True)
        
        sf.write(mixed_audio_fp, mixed_audio, 16000)
        

        
            
        
    
    
        




def mix_per_snr(target_speaker, all_interference, dB_snr):
    """ Mix target speaker with interfering speaker and noise on GPU. """

    if dB_snr == "mixed":
        dB_snr = np.random.uniform(-10, 10)
    else:
        dB_snr = int(dB_snr)
    snr_factor = 10**(dB_snr / 10)
    
    target_speaker_power = np.mean(target_speaker**2)
    all_interference_power = np.mean(all_interference**2)

    scale_factor = np.sqrt(target_speaker_power / (snr_factor * all_interference_power))
    mixed_audio = target_speaker + all_interference * scale_factor
    mixed_audio = librosa.util.normalize(mixed_audio)
    
    return mixed_audio
    
    
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data under various conditions.")
    parser.add_argument("--condition", type=str, required=True,
                        help='Condition(s), comma-separated if multiple: noise_only,one_interfering_speaker')
    parser.add_argument("--snr", type=str, required=True,
                        help='SNR value(s), comma-separated if multiple: mixed,-10,-5, 0')

    args = parser.parse_args()

    # Split by comma, strip whitespace
    conditions = [c.strip() for c in args.condition.split(",")]
    snrs = [s.strip() for s in args.snr.split(",")]

    for condition in conditions:
        for snr in snrs:
            print(f"Running condition: {condition}, SNR: {snr}")
            main(condition, snr)
