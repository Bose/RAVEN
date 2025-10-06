import os
import numpy as np
import torch
from torch.utils import data
import pandas as pd
import librosa
import hashlib
from src.utils.utils import crop_pad_audio
from src.utils.augment_visual import augment_visual
import src.config as config


SPLIT_FILE_PATH = "./data/split.parquet"
SAMPLING_RATE = config.SAMPLING_RATE


class VoxCeleb2(data.Dataset):
    
    def __init__(self, split, data_path, visual_encoder, embedding_size, all_data=None, condition=None, snr=None):
        self.split = split
        self.data_path = data_path
        if all_data is None:
            all_data = pd.read_parquet(SPLIT_FILE_PATH)
        musan_fps = pd.read_csv("./data/musan_split.csv")
        self.musan_fps = musan_fps[musan_fps["split"] == self.split]
        if self.split == "train":
            self.data = all_data[all_data["split"] == self.split]["audio_fp"]
            if "AVHuBERT" in visual_encoder:
                failed = pd.read_csv("./data/failed_avhubert_frontE.txt", header=None)[0]
                failed = failed.str.replace("/mp4/", "/aac/").str.replace(".mp4", ".m4a")
                self.data = self.data[~self.data.isin(failed)]
        elif self.split == "val":
            self.data = pd.read_csv("./data/VoxCeleb2_val_1000_fps.txt", header=None)[0]
        
        elif self.split == "test":
            self.data = pd.read_csv("./data/VoxCeleb2_test_1000_fps.txt", header=None)[0]
            self.condition = condition
            self.snr = snr
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        
        self.embedding_path_dict = {
            "VSRiW": "/vsriw/",
            "TalkNet": "/TalkNet_feats/",
            "Loconet": "/Loconet_feats/",
            "AVHuBERT": "/AVHuBERT_feats/",
            "VSRIW_LRS3": "/vsriw_lrs3/",
            
        }
        self.device = "cuda"
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_fp = os.path.join(self.data_path, self.data.iloc[idx])
        mixed_audio_fp = audio_fp.replace("/aac/", "/mixed_wav/").replace(".m4a", ".wav")
        # get raw target audio
        audio = crop_pad_audio(audio_fp, SAMPLING_RATE, 5)
        input_audio = librosa.util.normalize(audio)
        
        if self.split == "test":
            mixed_audio_fp = mixed_audio_fp.replace("/mixed_wav/", f"/mixed_wav/{self.condition}/{self.snr}/")
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)

        else:
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)
        input_audio, mixed_audio = torch.tensor(input_audio), torch.tensor(mixed_audio)
        if_speaker_fp = "None"
        
        
        
        
        face_embed = torch.zeros((125, self.embedding_size))
        # get face embeddings
        if "_" in self.visual_encoder: 
            
            if "addition" in self.visual_encoder:
                face_embed = self._add_embeddings(audio_fp, self.visual_encoder)
            elif "concatenate" in self.visual_encoder:
                face_embed = self._concatenate_embeddings(audio_fp, self.visual_encoder)
            
        else:
            fe_fp = audio_fp.replace("/aac/", self.embedding_path_dict[self.visual_encoder]).replace(".m4a", ".npy")
            face_embed = np.load(fe_fp, mmap_mode="r")
            # crop and pad face embeddings to 5 seconds
            face_embed = self._crop_pad_face_embeddings(face_embed)
            face_embed = torch.tensor(face_embed)
            
            if self.split != "test":
                face_embed = augment_visual(face_embed, self.visual_encoder)[0]

        
        return {
            "face_embed": face_embed,
            "input_audio": input_audio,
            "mixed_audio": mixed_audio,
            "audio_fp": audio_fp,
            "interfering_speaker_fp": if_speaker_fp
        }
    
    def _concatenate_embeddings(self, audio_fp, combined_features):
        
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        fe_fp1 = audio_fp.replace("/aac/", self.embedding_path_dict[fe1]).replace(".m4a", ".npy")
        fe_fp2 = audio_fp.replace("/aac/", self.embedding_path_dict[fe2]).replace(".m4a", ".npy")
        face_embed1 = np.load(fe_fp1, mmap_mode="r")
        face_embed2 = np.load(fe_fp2, mmap_mode="r")
        
        face_embed1, face_embed2 = self._crop_pad_face_embeddings(face_embed1), self._crop_pad_face_embeddings(face_embed2)
        
        face_embed1, face_embed2 = torch.tensor(face_embed1), torch.tensor(face_embed2)
        
        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        
        return torch.cat((face_embed1, face_embed2), dim=1)
    
    def _add_embeddings(self, audio_fp, combined_features):
        
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        fe_fp1 = audio_fp.replace("/aac/", self.embedding_path_dict[fe1]).replace(".m4a", ".npy")
        fe_fp2 = audio_fp.replace("/aac/", self.embedding_path_dict[fe2]).replace(".m4a", ".npy")
        face_embed1 = np.load(fe_fp1, mmap_mode="r")
        face_embed2 = np.load(fe_fp2, mmap_mode="r")
        
        face_embed1, face_embed2 = self._crop_pad_face_embeddings(face_embed1), self._crop_pad_face_embeddings(face_embed2)
        
        face_embed1, face_embed2 = torch.tensor(face_embed1), torch.tensor(face_embed2)
        
        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        
        return face_embed1 + face_embed2
    
    
    def normalize(self, x, norm='l2'):
        if norm == 'l2':
            return self.l2_normalize(x)
        elif norm == 'z_score':
            return self.z_score_normalization(x)
        else:
            return
    
    def l2_normalize(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)
    
    def z_score_normalization(self, x):
        mean = x.mean(dim=1, keepdim=True)  
        std = x.std(dim=1, keepdim=True) + 1e-6  
        return (x - mean) / std
    


    
    def _crop_pad_face_embeddings(self, fe):
        
        if len(fe) < 25*5:
            fe = np.pad(fe, ((0, 25*5 - len(fe)), (0,0)))
        fe = fe[:25*5]
        
        return fe
        
        
    

    
    
    