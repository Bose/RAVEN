import torch.nn as nn
import torch.nn.functional as F

class Video_Encoder(nn.Module):
    
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        
        
    def forward(self, x):
        
        # upsample frame rate to 100 Hz using simple nearest neighbor interpolation from 25 Hz
        # note that this makes the video embeddings have the exact same number of frames as the stft spectrogram for 5 second audio
        num_frames = x.size(1) * 4
        padded_length = (num_frames - num_frames % 4)

        x = F.interpolate(x[:, None, :], size=(padded_length, self.embedding_size), mode='nearest') # None for channel
        x = x.squeeze(1)
        
        return x
    