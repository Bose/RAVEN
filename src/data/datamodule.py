import pytorch_lightning as pl
from torch.utils import data
from src.data.dataset import VoxCeleb2
import torch
import pandas as pd

SPLIT_FILE_PATH = "./data/split.parquet"

class VoxCeleb2DataModule(pl.LightningDataModule):

    def __init__(self, data_path, visual_encoder, embedding_size, batch_size=4, num_workers=2):
        super(VoxCeleb2DataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        self.data_path = data_path

    def setup(self, test_condition, test_snr):
        all_data = pd.read_parquet(SPLIT_FILE_PATH)
        self.train_dataset = VoxCeleb2("train", self.data_path, self.visual_encoder, self.embedding_size, all_data)
        self.val_dataset = VoxCeleb2("val", self.data_path, self.visual_encoder, self.embedding_size, all_data)
        self.test_dataset = VoxCeleb2("test", self.data_path, self.visual_encoder, self.embedding_size, all_data, test_condition, test_snr)


    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self, small=True):
        if small:
            subset_sampler = data.SubsetRandomSampler(range(1000))
            return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, sampler=subset_sampler, persistent_workers=True, prefetch_factor=2)
        else:
            return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=2)
    


    def collate_fn_custom(self, batch):
        return {
            'face_embed': torch.stack([x['face_embed'] for x in batch]).to("cuda", non_blocking=True), 
            'input_audio': torch.stack([x['input_audio'] for x in batch]).to("cuda", non_blocking=True),
            'mixed_audio': torch.stack([x['mixed_audio'] for x in batch]).to("cuda", non_blocking=True),
            'audio_fp': [x['audio_fp'] for x in batch],
            'interfering_speaker_fp': [x['interfering_speaker_fp'] for x in batch],
        }


if __name__ == "__main__":
    
    import time
    import cProfile
    datamodule = VoxCeleb2DataModule(visual_encoder="TalkNet" , embedding_size=1024 ,batch_size=4, num_workers=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i == 10:
            break
    print(f"Time to load 1 batches: {time.time() - start_time} seconds")