import torch
import torch.nn as nn
import torch.nn.functional as F
# defing the SNR and PSNR fucntion for mel spectrogram
def PSNR(mel_ref_dB, mel_recon_dB, mode='amplitude'):
    '''
    compute PSNR between reference mel spectrogram and a reconstructed one
    '''
    
    if mode == 'amplitude':
        '''
        compute PSNR with amplitude spectrogram
        '''
        
        # convert the power spectrogram to the ones in amplitude
        mel_ref_a = torch.pow(10, 0.05 * mel_ref_dB)
        mel_recon_a = torch.pow(10, 0.05 * mel_recon_dB)        
        
        # signal: maximum power in the reference spectrogram
        # noise: MSE between the referene and reconstructed one 
        snr = 10*torch.log(torch.max(mel_ref_a**2)/F.mse_loss(mel_ref_a,mel_recon_a))
        
    elif mode == 'scale':
        '''
        compute PSNR with scaled spectrogram
        '''
        
        # scale the range of decible spectrogram to [0, 1]
        mmax = torch.max(mel_ref_dB)
        mmin = torch.min(mel_ref_dB)
        
        mel_ref_scale = (mel_ref_dB-mmin)/(mmax-mmin)
        mel_recon_scale = (mel_recon_dB-mmin)/(mmax-mmin)
        
        
        # signal: 1, the maximum scaled value in the reference spectrogram
        # noise: MSE between the scaled eferene and reconstructed one 
        snr = 10*torch.log(1/F.mse_loss(mel_ref_scale,mel_recon_scale))
        
    return snr
        
def SNR(mel_ref_dB, mel_recon_dB, mode='amplitude'):
    '''
    compute SNR between reference mel spectrogram and a reconstructed one, this method is finally used to evaluate the proposed approach
    '''
    
    if mode == 'amplitude':
        '''
        compute SNR with amplitude spectrogram
        '''
        
        # convert the power spectrogram to the ones in amplitude
        mel_ref_a = torch.pow(10, 0.05 * mel_ref_dB)
        mel_recon_a = torch.pow(10, 0.05 * mel_recon_dB)
        
        # signal: sum power in the reference spectrogram
        # noise: sum of MSE between the referene and reconstructed one 
        snr = 10*torch.log(torch.sum(mel_ref_a**2)/F.mse_loss(mel_ref_a,mel_recon_a,reduction='sum'))
        
    elif mode == 'scale':
        '''
        compute SNR with scaled spectrogram
        '''
        
        # scale the range of decible spectrogram to [0, 1]
        mmax = torch.max(mel_ref_dB)
        mmin = torch.min(mel_ref_dB)
        
        mel_ref_scale = (mel_ref_dB-mmin)/(mmax-mmin)
        mel_recon_scale = (mel_recon_dB-mmin)/(mmax-mmin)
        
        
        # signal: sum of scaled scaled value in the reference spectrogram
        # noise: sum of MSE between the scaled eferene and reconstructed one 
        snr = 10*torch.log(torch.sum(mel_ref_scale**2)/F.mse_loss(mel_ref_scale,mel_recon_scale,reduction='sum'))
        
    return snr
        
def MSE(mel_ref_dB, mel_recon_dB, mode='amplitude'):
    '''
    simply compute MSE as the criterion
    '''
    if mode == 'amplitude':
        mel_ref_a = torch.pow(10, 0.05 * mel_ref_dB)
        mel_recon_a = torch.pow(10, 0.05 * mel_recon_dB)
        
        return F.mse_loss(mel_ref_a, mel_recon_a)
    
    if mode == 'scale':
        mmax = torch.max(mel_ref_dB)
        mmin = torch.min(mel_ref_dB)
        
        mel_ref_scale = (mel_ref_dB-mmin)/(mmax-mmin)
        mel_recon_scale = (mel_recon_dB-mmin)/(mmax-mmin)
        
        return F.mse_loss(mel_ref_scale, mel_recon_scale)
    if mode == 'dB':
        
        return F.mse_loss(mel_ref_dB, mel_recon_dB)
    
    
        
def MAE(mel_ref_dB, mel_recon_dB, mode='amplitude'):
    
    '''
    simply compute MSE as the criterion
    '''
    
    if mode == 'amplitude':
        mel_ref_a = torch.pow(10, 0.05 * mel_ref_dB)
        mel_recon_a = torch.pow(10, 0.05 * mel_recon_dB)
        
        return F.l1_loss(mel_ref_a, mel_recon_a)
    
    if mode == 'scale':
        mmax = torch.max(mel_ref_dB)
        mmin = torch.min(mel_ref_dB)
        
        mel_ref_scale = (mel_ref_dB-mmin)/(mmax-mmin)
        mel_recon_scale = (mel_recon_dB-mmin)/(mmax-mmin)
        
        return F.l1_loss(mel_ref_scale, mel_recon_scale)
    if mode == 'dB':
        
        return F.l1_loss(mel_ref_dB, mel_recon_dB)