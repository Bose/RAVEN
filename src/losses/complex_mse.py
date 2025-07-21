import torch
import torch.nn as nn

class PSA_MSE(nn.Module):
    def __init__(self):
        super(PSA_MSE, self).__init__()
    
    def forward(self, complex_spec_est: tuple, complex_spec_target: tuple, complex_loss_weight: float = 1.0, magnitude_loss_weight: float = 1.0):
        
        """
        complex_spec_est: tuple of magnitude spectrogram and phase, phase is in radians
        
        """
        mag_spec_est, phase_est = complex_spec_est
        complex_spec_est = mag_spec_est * torch.exp(1j*phase_est)
        
        mag_spec_target, phase_target = complex_spec_target
        complex_spec_target = mag_spec_target * torch.exp(1j*phase_target)
        
        
        complex_loss = self._complex_mse_loss(complex_spec_est, complex_spec_target)
        magnitude_loss = self._magnitude_mse_loss(mag_spec_est, mag_spec_target)
        
        total_psa_loss = complex_loss_weight * complex_loss + magnitude_loss_weight * magnitude_loss
        
        return total_psa_loss
    
    def _complex_mse_loss(self, complex_spec_est: torch.Tensor, complex_spec_target: torch.Tensor):
        """
        Given complex spectrogram estimates and target, calculate the complex MSE loss by
        taking the MSE of the real and imaginary parts of the complex spectrograms
        """
        
        complex_spec_real_est, complex_spec_imag_est = torch.real(complex_spec_est), torch.imag(complex_spec_est)
        complex_spec_real_target, complex_spec_imag_target = torch.real(complex_spec_target), torch.imag(complex_spec_target)
        mse = nn.MSELoss()
        real_loss = mse(complex_spec_real_est, complex_spec_real_target)
        imag_loss = mse(complex_spec_imag_est, complex_spec_imag_target)
        return real_loss + imag_loss
    
    def _magnitude_mse_loss(self, mag_spec_est: torch.Tensor, mag_spec_target: torch.Tensor):
        """
        Given magnitude spectrogram estimates and target, calculate the magnitude MSE loss
        """
        mse = nn.MSELoss()
        return mse(mag_spec_est, mag_spec_target)
        

