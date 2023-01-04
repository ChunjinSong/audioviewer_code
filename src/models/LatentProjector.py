import torch
import torch.nn as nn
import math
def identity_init(linear_layer):
    '''
    The method is to initialize the a latent projector weights with indentity-like matrices
    '''
    out_features, in_features  = linear_layer.weight.shape
    device = linear_layer.weight.device
    
    if in_features<=out_features:
        linear_layer.weight = nn.Parameter(torch.eye(in_features).repeat(
            math.ceil(out_features/in_features),1)[:out_features].to(device))
    else:
        linear_layer.weight = nn.Parameter(torch.eye(out_features).repeat(1,
            math.ceil(in_features/out_features))[:,:in_features].to(device))
        
    linear_layer.bias=nn.Parameter(torch.zeros(out_features).to(device))
    return

class LatentProjector(nn.Module):
    '''
    LatentProjector has 2 single-layer linear networks to project latent codes
    '''
    def __init__(self, audio_latent, visual_latent):
        super().__init__()
        self.audio2visual = nn.Linear(audio_latent,visual_latent)
        self.visual2audio = nn.Linear(visual_latent,audio_latent)
    def A2V(self,x):
        return self.audio2visual(x)
    def V2A(self,x):
        return self.visual2audio(x)