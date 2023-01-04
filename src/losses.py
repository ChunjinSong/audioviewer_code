
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision.models import vgg19_bn

# losses
def kld(mu, logvar, q_mu=None, q_logvar=None):
    """compute dimension-wise KL-die=1e-8vergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    if q_mu is None:
        q_mu = torch.zeros(mu.size(),device=mu.get_device())
    else:
        print("using non-default q_mu %s" % q_mu)

    if q_logvar is None:
        q_logvar = torch.zeros(logvar.size(),device=mu.get_device())
    else:
        print("using non-default q_logvar %s" % q_logvar)

    mu_shape = list(mu.size())
    q_mu_shape = list(q_mu.size())
    logvar_shape = list(logvar.size())
    q_logvar_shape = list(q_logvar.size())

    if not np.all(mu_shape == logvar_shape):
        raise ValueError("mu_shape (%s) and logvar_shape (%s) does not match" % (
          mu_shape, logvar_shape))
    if not np.all(mu_shape == q_mu_shape):
        raise ValueError("mu_shape (%s) and q_mu_shape (%s) does not match" % (
          mu_shape, q_mu_shape))
    if not np.all(mu_shape == q_logvar_shape):
        raise ValueError("mu_shape (%s) and q_logvar_shape (%s) does not match" % (
          mu_shape, q_logvar_shape))

    # print(mu.get_device(),q_mu.get_device())


    return -0.5 * (1 + logvar - q_logvar - \
          (torch.pow(mu - q_mu, 2) + torch.exp(logvar)) / torch.exp(q_logvar))

def log_gauss(mu, logvar, x):
    """compute point-wise log prob of Gaussian"""
    x_shape = list(x.size())
    mu_shape = list(mu.size())
    logvar_shape = list(logvar.size())
  
    if not np.all(x_shape == mu_shape):
        raise ValueError("x_shape (%s) and mu_shape (%s) does not match" % (
          x_shape, mu_shape))
    if not np.all(x_shape == logvar_shape):
        raise ValueError("x_shape (%s) and logvar_shape (%s) does not match" % (
          x_shape, logvar_shape))

    return 0.5 * (np.log(2 * np.pi) + logvar + torch.pow((x - mu), 2) / torch.exp(logvar))

def log_normal(x):
    """compute point-wise log prob of Gaussian"""
    # seems unused
    return -0.5 * (np.log(2 * np.pi) + torch.pow(x, 2))

def loss_fn(outputs, targets):

    kld_loss = torch.mean(torch.sum(kld(*outputs['qz_x']), dim=1))

    x_mu, x_logvar = outputs['px_z']
    logpx_z = torch.mean(torch.sum(
            log_gauss(x_mu, x_logvar, targets), 
            dim=(1, 2, 3)))

    loss = kld_loss + logpx_z
    return loss, {"kld_loss" : kld_loss.item(),
                  "logpx_z" : logpx_z.item()}

def pair_loss_mse(batch_out, batch_in, e=1e-8):
    return F.mse_loss(batch_out["pt"], torch.abs(batch_in['dt']))

def pair_loss_mse_log(batch_out, batch_in, e=1e-8):
    return F.mse_loss(torch.log(torch.abs(batch_out["pt"]+e)), 
        torch.log(torch.abs(batch_in['dt'])))

def pair_loss_q(batch_out, batch_in, e=1e-8):
    return F.mse_loss((batch_out["pt"]+e)/(torch.abs(batch_in['dt'])+e),
                      torch.ones_like(batch_out["pt"]))

def pair_loss_hinge(batch_out, batch_in, threshold_t=0.2, margin=1.0):
    y = (torch.abs(batch_in['dt'])<threshold_t)*2-1
    if isinstance(batch_in, dict):

        N = batch_in['s1'].shape[0]
        z_diff = batch_out['z'][:N]-batch_out['z'][N:]
    else:
        N = batch_in.shape[0]
        z_diff = z[:-1]-z[1:]
    x = torch.sqrt(torch.sum((z_diff)**2, -1, keepdim=True))
    return F.hinge_embedding_loss(x,y,margin)

def pair_loss_hinge_pt(batch_out, batch_in, threshold_t=0.2, margin=1.0):
    y = (torch.abs(batch_in['dt'])<threshold_t)*2-1

    return F.hinge_embedding_loss(torch.abs(batch_out['pt']),y,margin)

# feature loss computing module
class featureLoss():
    def __init__(self,feature_network = None,
                 feature_layers = None,
                 device='cuda',
                 loss_fn='L2'):
        self.device = device
        if feature_network==None:
            self.feature_network = vgg19_bn(pretrained=True).to(device)
            if feature_layers==None:
                self.feature_layers = ['14', '24', '34', '43']
            else:
                self.feature_layers = feature_layers
        else:
            self.feature_network = feature_network.to(device)
            self.feature_layers = feature_layers
            
        if loss_fn == 'L2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_fn == 'L1':
            self.loss_fn = nn.L1loss(reduction='none')
        
            
        self.feature_network.eval()
        return
    
    def extract_features(self,
                         input):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :return: List of the extracted features
        """
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result).detach()
            if(key in self.feature_layers):
                features.append(result)
#         print([f.shape for f in features])
        return features
    
    def loss(self, source, target):
#         print(type(source))
        if type(source) is np.ndarray:
            source = torch.from_numpy(source.astype(np.float32)).to(self.device)
        elif source.device is not self.device:
            source = source.to(self.device)
            
        if type(target) is np.ndarray:
            target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        elif target.device is not self.device:
            target = target.to(self.device)
            
        source_features = self.extract_features(source)
        target_features = self.extract_features(target)
        
        feature_loss = torch.zeros(source.shape[0])
        for (s, t) in zip(source_features, target_features):
            
            feature_loss += self.loss_fn(s, t).mean((1,2,3)).detach().cpu()   
#         print(feature_loss.shape)
        return feature_loss
    def pairwise(self,source, target=None):
        
        # print(self.device)
        if type(source) is np.ndarray:
            source = torch.from_numpy(source.astype(np.float32)).to(self.device)
        elif source.device is not self.device:
            source = source.to(self.device)
        
        source_features = self.extract_features(source)

        if target is None:
            feature_loss =torch.zeros(source.shape[0],
                                      source.shape[0])
            
            for s in source_features:
                for i in range(s.shape[0]-1):
                    feature_loss[i:i+1,i+1:] += self.loss_fn(s[i:i+1].expand_as(s[i+1:]),s[i+1:]).mean((1,2,3)).detach().cpu()
            feature_loss += feature_loss.transpose(0,1).clone()

        else:
            if type(target) is np.ndarray:
                target = torch.from_numpy(target.astype(np.float32)).to(self.device)
            elif target.device is not self.device:
                target = target.to(self.device)
            target_features = self.extract_features(target)
            
            feature_loss = torch.zeros(source.shape[0],
                                  target_features[0].shape[0])
            
            for (s, t) in zip(source_features, target_features):
                for i in range(s.shape[0]):
                    feature_loss[i:i+1] += self.loss_fn(s[i:i+1].expand_as(t), t).mean((1,2,3)).detach().cpu()   
        return feature_loss
        

# SWD computation. revised based on SWAE implementation 
def get_random_projections(latent_dim, num_samples, proj_dist='normal'):
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """
    if proj_dist == 'normal':
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == 'cauchy':
        rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                   torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
    else:
        raise ValueError('Unknown projection distribution.')

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
    return rand_proj # [S x D]


def compute_swd(z1, z2, p=2):
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors and computing
    their Wasserstein distance along those projections.

    :param z1: Latent samples 1 # [N  x D]
    :param z2: Latent samples 2 # [N  x D]
    :param p: Value for the p^th Wasserstein distance
    :return:
    """
    if z1.shape != z2.shape:
        raise RuntimeError("Tensor size mismatch")
    
    device = z1.device
    
    num_projections, latent_dim = z1.shape

    proj_matrix = get_random_projections(latent_dim,
                                              num_samples=num_projections).transpose(0,1).to(device)

    z1_projections = z1.matmul(proj_matrix) # [N x S]
    z2_projections = z2.matmul(proj_matrix) # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = torch.sort(z1_projections.t(), dim=1)[0] - \
             torch.sort(z2_projections.t(), dim=1)[0]
    w_dist = w_dist.pow(p)
    return w_dist.mean()
    
def compute_swd_LCM(z1, z2, p=2):
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors of different sample
    size and computing their Wasserstein distance along those projections.

    :param z1: Latent samples 1 # [N1  x D]
    :param z2: Latent samples 2 # [N2  x D]
    :param p: Value for the p^th Wasserstein distance
    :return:
    """
    if z1.shape == z2.shape:
        return compute_swd(z1,z2,p)
    
    elif z1.dim() == z2.dim() and z1.shape[1] == z2.shape[1]:
        # compute the least common multiple of the two batch size and resize the inputs
        n1 = z1.shape[0]
        n2 = z2.shape[0]
        nm = np.lcm(n1,n2)
        
        z1_exp = z1.repeat(int(nm/n1),1)
        z2_exp = z2.repeat(int(nm/n2),1)
        
        return compute_swd(z1_exp, z2_exp, p)
    else:
        raise RuntimeError(f"Tensor size mismatch: {z1.shape}, {z2.shape}")




''' ===================== add by scj ======================='''

def simple_recomb_loss(model, outputs):
    '''
    recombined reconstruction loss, reconstruct only the first samples in the triplets
    '''
    N = outputs["z"].shape[0] // 3
    targets = outputs['targets'][:N]
    spk_idx = list(range(2 * N, 3 * N))
    ph_idx = list(range(N, 2 * N))

    z_recomb = torch.cat([outputs['z'][spk_idx, :model.s_dim], outputs['z'][ph_idx, model.s_dim:]], 1)

    px_z, _ = model.decode(z_recomb)
    x_mu, x_logvar = px_z

    logpx_z = torch.mean(torch.sum(
        log_gauss(x_mu, x_logvar, targets),
        dim=(1, 2, 3)).to(torch.device('cuda:0')))

    return logpx_z, px_z
