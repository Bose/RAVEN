import torch.nn as nn
import torch.nn.functional as F

class Audio_Encoder(nn.Module):
    
    def __init__(self, sr=16000, window_size=400, hop_size=160, n_fft=512, p=0.3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 7), padding=self.get_padding((5, 7), (1, 1)), dilation=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Conv2d(96, 96, kernel_size=(1, 1), padding=self.get_padding((1, 1), (1, 1)), dilation=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Conv2d(96, 96, kernel_size=(1, 5), padding=self.get_padding((1, 5), (1, 1)), dilation=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Conv2d(96, 96, kernel_size=(1, 5), padding=self.get_padding((1, 5), (1, 1)), dilation=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Conv2d(96, 8, kernel_size=(1, 1), padding=self.get_padding((1, 1), (1, 1)), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
    
    def get_padding(self, kernel_size, dilation):
        return ((kernel_size[0] - 1) * dilation[0] // 2, (kernel_size[1] - 1) * dilation[1] // 2)
    
    def forward(self, spec):
        spec = spec.unsqueeze(1)  # add channel dimension
        output = self.conv_layers(spec)
        return output
    

    
        
        