
import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import src.config as config
sys.path.append(config.TALKNET_PATH)
from talkNet import talkNet
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from torchvision.transforms.functional import center_crop
from pathlib import Path

from torchvision.transforms import ToPILImage

DATA_FOLDER_PATH = config.DATA_FOLDER_PATH



class VideoDataset(Dataset):
    def __init__(
            self, 
            data_split_path ,
            data_path,

            split
            ) :
        super().__init__()
        self.df = pd.read_csv(data_split_path)
        self.data_path = data_path
        self.split= split
        self.file_list = pd.read_csv("./data/split.csv", header=None)[0].str.replace("/aac/", "/mp4/").str.replace(".m4a", ".mp4").tolist()


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        fp = os.path.join(self.data_path, self.file_list[index])
        video_tensor_orig = torchvision.io.read_video(fp, pts_unit='sec')[0]
        video_tensor = random_5s_clip(video_tensor_orig)
        assert video_tensor.shape[0] == 125, f"Video {fp} has {video_tensor.shape} frames"
        
        rgb_to_grayscale_weight = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
        rgb_to_grayscale_weight = rgb_to_grayscale_weight.view(1, 1, 1, 3)
        gray_video = torch.sum(video_tensor * rgb_to_grayscale_weight, dim=-1, keepdim=False)
        gray_video = center_crop(gray_video, [112, 112])
        
        return gray_video, fp


class TalkNetBatchedPreprocessing:
    def __init__(
            self,
            model_path,
            data_split_path,
            data_path,
            split,
            batch_size: int = None,
            num_workers: int = 1,

            device = "cuda:1"
        ) -> None:
        super().__init__()
        self.split = split
        self.ds = VideoDataset(data_split_path, data_path, split = self.split)
        self.dataloader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
        self.model = talkNet().to(device)
        self.model.loadParameters(model_path)
        self.device = device

    def extract_features(self):
        
        
        failed_text_path = os.path.join(config.DATA_FOLDER_PATH, f"failed_Loconet_frontE_feat_{self.split}.txt")
        with open(failed_text_path, 'w') as failed_txt:
            i = 0
            for video_tensor, fp in tqdm(self.dataloader):
                if video_tensor is not None:
                    video_tensor = video_tensor.unsqueeze(0)
                    B,T,H,W = video_tensor.shape
                    video_tensor = video_tensor.to(self.device)
                    video_tensor = video_tensor.view(B*T, 1, 1, W, H)
                    video_tensor = (video_tensor / 255 - 0.4161) / 0.1688
                    
                    video_embed = self.model.model.visualFrontend(video_tensor)
                    video_embed = video_embed.view(B, T, -1)
                    video_embed = video_embed.squeeze(0)

                    output_path = fp.replace("/mp4/", "/TalkNet_feats/")
                    output_ft_path = output_path.replace(".mp4", ".npy")
                    
                    if not os.path.exists(os.path.dirname(output_ft_path)):
                        os.makedirs(os.path.dirname(output_ft_path))

                    video_embed = video_embed.detach().cpu().numpy()
                    
                    assert video_embed.shape == (125, 512), f"Video {fp} has shape {video_embed.shape}"
                    
                    np.save(output_ft_path, video_embed)
                    print(f"NUMPY FILE SAVED TO {output_ft_path}")

                else:
                    print(f"Video {fp} preprocessing failed.")
                    failed_txt.write(fp)
    
    def test_zero(self):
        
        with torch.no_grad():

            video_tensor = torch.zeros(1, 112, 112)
            video_tensor = video_tensor.unsqueeze(0)
            B,T,H,W = video_tensor.shape
            video_tensor = video_tensor.to(self.device)
            video_tensor = video_tensor.view(B*T, 1, 1, W, H)
            
            video_embed = self.model.model.visualFrontend(video_tensor)
            video_embed = video_embed.view(B, T, -1)
            video_embed = video_embed.squeeze(0)
            
            np.save(f"./pretrained_zeros/TalkNet_zero_oneframe.npy", video_embed.detach().cpu().numpy())
        
            print(video_embed.shape)
            print(video_embed)
        


def random_5s_clip(video, frame_rate=25):

    T, H, W, C = video.shape
    target_frames = 5 * frame_rate 

    if T >= target_frames:
        
        return video[:target_frames, :, :, :]
    else:
        
        pad_length = target_frames - T
        padding = (0, 0,  
                   0, 0,  
                   0, 0,  
                   0, pad_length)  
        
        return F.pad(video, padding, mode='constant', value=0)

def main():
    model_path = os.path.join(config.TALKNET_PATH, "pretrain_TalkSet.model")
    data_split = "./data/split.csv"
    process = TalkNetBatchedPreprocessing(
        model_path,
        data_split,
        DATA_FOLDER_PATH,
        split = "test",
        num_workers=64
    )
    process.extract_features()

if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"
    main()

    