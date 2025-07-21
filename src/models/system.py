import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import random
from src.models.fusion import Fusion
from src.utils.spec_audio_conversion import convert_to_audio_from_complex_spectrogram_after_compression_torch, convert_to_complex_spectrogram_with_compression_torch
from src.utils.denoiser import denoise_speech
from pytorch_lightning.utilities import grad_norm




class System(pl.LightningModule):
    def __init__(self, model, loss, metrics):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = nn.ModuleDict(metrics)
        self.base_seed = 42
        self.error_files = []
        self.error_log_path = 'error_files.txt'
        self.test_dict = {
            'test_loss': [],
            'test_pesq': [],
            'test_sisdr': [],  
            'test_sisdr_gain': [],
            'test_snr': [],
            'test_snr_gain': [],
            'test_estoi': []
        }


        
    def forward(self, spec, face_embed):
        return self.model(spec, face_embed)

    
    def common_step(self, batch, batch_idx):

        input_audio = batch['input_audio']
        mixed_audio = batch['mixed_audio']
        face_embed = batch['face_embed']
        
        with torch.no_grad():
            clean_audio_target = denoise_speech(input_audio)
        
        # dims for below 4: (B, N, T) batch, bins, frames
        mixed_mag_spec, mixed_phase = convert_to_complex_spectrogram_with_compression_torch(mixed_audio)
        clean_mag_spec, clean_phase = convert_to_complex_spectrogram_with_compression_torch(clean_audio_target)
        
        # model requires (B, T, N) dims
        mixed_mag_spec, mixed_phase, clean_mag_spec, clean_phase = mixed_mag_spec.permute(0, 2, 1), mixed_phase.permute(0, 2, 1), clean_mag_spec.permute(0, 2, 1), clean_phase.permute(0, 2, 1)
        padded_length = mixed_mag_spec.size(1) - mixed_mag_spec.size(1) % 4
        mixed_mag_spec, mixed_phase, clean_mag_spec, clean_phase = mixed_mag_spec[:, :padded_length, :], mixed_phase[:, :padded_length, :], clean_mag_spec[:, :padded_length, :], clean_phase[:, :padded_length, :]

        clean_complex_spec_target = (clean_mag_spec, clean_phase)

        mag_spec_est = self.model(mixed_mag_spec, face_embed)
        clean_complex_spec_est = (mag_spec_est, mixed_phase)
        loss = self.loss(clean_complex_spec_est, clean_complex_spec_target)
        clean_audio_pred = convert_to_audio_from_complex_spectrogram_after_compression_torch(clean_complex_spec_est).to("cuda")

        clean_audio_target = clean_audio_target.detach().to(clean_audio_pred.device)
        assert type(clean_audio_pred) == type(clean_audio_target), f"Expected clean_audio_pred and clean_audio_target to have the same type but got pred {type(clean_audio_pred)} and target {type(clean_audio_target)}"
        with torch.no_grad():
            sisdr = self.metrics['sisdr'](clean_audio_pred, clean_audio_target)
            sisdr_gain = sisdr - self.metrics['sisdr'](mixed_audio, clean_audio_target)
            snr = self.metrics['snr'](clean_audio_pred, clean_audio_target)
            snr_gain = snr - self.metrics['snr'](mixed_audio, clean_audio_target)
            
        return loss, sisdr, sisdr_gain,snr,snr_gain ,clean_audio_pred, clean_audio_target
    
    def training_step(self, batch, batch_idx):
    
        loss, sisdr, sisdr_gain, snr, snr_gain,  clean_audio_pred , clean_audio_target = self.common_step(batch, batch_idx)
        self.log_dict({"train/loss": loss, "train/sisdr": sisdr, "train/sisdr_gain": sisdr_gain, "train/snr": snr, "train/snr_gain": snr_gain})
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, sisdr, sisdr_gain, snr, snr_gain, clean_audio_pred, clean_audio_target = self.common_step(batch, batch_idx)
        pesq = self.get_pesq(clean_audio_pred, clean_audio_target, batch)
        estoi = self.get_estoi(clean_audio_pred, clean_audio_target, batch)
        self.log_dict({"val/loss": loss, "val/sisdr": sisdr, "val/sisdr_gain": sisdr_gain, "val/snr": snr, "val/snr_gain": snr_gain, "val/pesq": pesq,  "val/estoi": estoi})
        
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        loss, sisdr, sisdr_gain, snr, snr_gain, clean_audio_pred, clean_audio_target = self.common_step(batch, batch_idx)
        pesq = self.get_pesq(clean_audio_pred, clean_audio_target, batch)
        estoi = self.get_estoi(clean_audio_pred, clean_audio_target, batch)
        self.test_dict["test_loss"].append(loss)
        self.test_dict["test_snr"].append(snr)
        self.test_dict["test_snr_gain"].append(snr_gain)
        self.test_dict["test_pesq"].append(pesq)
        self.test_dict["test_estoi"].append(estoi)
        self.test_dict["test_sisdr"].append(sisdr)
        self.test_dict["test_sisdr_gain"].append(sisdr_gain)
        
        
        
    def on_test_epoch_end(self):
        avg_loss = sum(self.test_dict["test_loss"]) / len(self.test_dict["test_loss"])
        avg_sisdr = sum(self.test_dict["test_sisdr"]) / len(self.test_dict["test_sisdr"])
        avg_sisdr_gain = sum(self.test_dict["test_sisdr_gain"]) / len(self.test_dict["test_sisdr_gain"])
        avg_snr = sum(self.test_dict["test_snr"]) / len(self.test_dict["test_snr"])
        avg_snr_gain = sum(self.test_dict["test_snr_gain"]) / len(self.test_dict["test_snr_gain"])
        avg_pesq = sum(self.test_dict["test_pesq"]) / len(self.test_dict["test_pesq"])
        avg_estoi = sum(self.test_dict["test_estoi"]) / len(self.test_dict["test_estoi"])
        

        print(f"AVG Test Loss: {avg_loss}, Test SISDR: {avg_sisdr}, Test SISDR Gain: {avg_sisdr_gain}, Test SNR: {avg_snr}, Test SNR Gain: {avg_snr_gain}, Test PESQ: {avg_pesq}, Test estoi: {avg_estoi}")

        
        return avg_loss, avg_pesq, avg_sisdr, avg_sisdr_gain, avg_snr, avg_snr_gain, avg_estoi
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        total_grad_norm = norms['grad_2.0_norm_total'] 
        self.log('train/grad_norm_total', total_grad_norm)
        
    def on_train_epoch_start(self):
        # Update seed for each epoch
        epoch_seed = self.base_seed + self.current_epoch  
        torch.manual_seed(epoch_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)

    
    
    
        
    def log_error(self, file_path, error_message):
        with open(self.error_log_path, "a") as f:
            f.write(f"{file_path}: {error_message}\n")
            
    @torch.no_grad()
    def get_pesq(self, audio_pred, audio_target, batch):
        
            
        try:
            pesq = self.metrics['pesq'](audio_pred, audio_target)
        except Exception as e:
            pesq = 0.0
            self.error_files.append(batch['audio_fp'])
            self.log_error(batch['audio_fp'], str(e))
            
        return pesq

    
    @torch.no_grad()
    def get_estoi(self, audio_pred, audio_target, batch):
        
        try:
            estoi = self.metrics['estoi'](audio_pred, audio_target)
        except Exception as e:
            estoi = 0.0
            self.error_files.append(batch['audio_fp'])
            self.log_error(batch['audio_fp'], str(e))
            
        return estoi
    
            