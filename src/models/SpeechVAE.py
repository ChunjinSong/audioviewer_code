import torch
import torch.nn as nn

class SpeechVAE(nn.Module):
    def __init__(self, input_dim, latent_dim,act_layer=nn.Tanh()):
        super().__init__();
        self.F = input_dim
        self.z_dim = latent_dim
        # encoder
        self.conv_layers_enc = nn.Sequential(
                      nn.Conv2d(1,64,(1,self.F),(1,1)), # VALID
                      nn.BatchNorm2d(64),
                      act_layer, 
                      nn.Conv2d(64,128,(3,1),(2,1),(1,0)),  # SAME
                      nn.BatchNorm2d(128),
                      act_layer,
                      nn.Conv2d(128,256,(3,1),(2,1),(1,0)), # SAME
                      nn.BatchNorm2d(256),
                      act_layer
                    )
        self.fc_layer_enc = nn.Sequential(
                    nn.Linear(1280,512), 
                    nn.BatchNorm1d(512),
                    act_layer
                  )
        self.mu_enc = nn.Linear(512,self.z_dim )
        self.var_enc = nn.Linear(512,self.z_dim )

        # decoder
        self.fc_layer_dec = nn.Sequential(
                    nn.Linear(self.z_dim ,512), 
                    nn.BatchNorm1d(512),
                    act_layer,
                    nn.Linear(512,1280), 
                    nn.BatchNorm1d(1280),
                    act_layer
                  )
        self.deconv_layers_dec = nn.Sequential(
                      nn.ConvTranspose2d(256,128,(3,1),(2,1),(1,0),(1,0)), # SAME
                      nn.BatchNorm2d(128),
                      act_layer,
                      nn.ConvTranspose2d(128,64,(3,1),(2,1),(1,0),(1,0)), # SAME
                      nn.BatchNorm2d(64),
                      act_layer,
                    )

        self.mu_dec = nn.ConvTranspose2d(64,1,(1,self.F),(1,1)) # VALID
        self.var_dec = nn.ConvTranspose2d(64,1,(1,self.F),(1,1))  # VALID
    def encode(self, x):
        x = torch.flatten(self.conv_layers_enc(x),1);
        x = self.fc_layer_enc(x)
        z_mu = self.mu_enc(x)
        z_var = self.var_enc(x)
        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(z_mu)
        return [z_mu, z_var], z_sample
    def decode(self, z):
        z = self.fc_layer_dec(z)
        B = z.shape[0]
        z = z.reshape(B,256,-1,1)
        z = self.deconv_layers_dec(z)
        x_mu = self.mu_dec(z)
        x_var = self.var_dec(z)

        # reparameterize
        std = torch.exp(x_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(x_mu)

        return [x_mu, x_var], x_sample
    def forward(self, x):
        # encode
        qz_x, z = self.encode(x)


        # decode
        px_z, x_sample = self.decode(z)

        outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x_sample, "targets": x}

        return outputs
        
class SpeechVAE_Pair(SpeechVAE):
    def __init__(self, *args, **kwargs):
        super(SpeechVAE_Pair,self).__init__(*args, **kwargs)
        self.scaler=nn.Linear(1,1,bias=False)
    def forward(self, batch):
       
        
        if isinstance(batch, dict):

            N = batch['s1'].shape[0]

            targets = torch.cat([batch['s1'],batch['s2']],0)
            # encode
            qz_x, z = self.encode(targets)

            # decode
            px_z, x = self.decode(z)

            # scaler output
            z_diff = z[:N]-z[N:]

        else:
            targets = batch
            N = batch.shape[0]
            # encode
            qz_x, z = self.encode(targets)

            # decode
            px_z, x = self.decode(z)

            # scaler output
            z_diff = z[:-1]-z[1:]
            
        z_dist = torch.sqrt(torch.sum((z_diff)**2, -1, keepdim=True))
    #         print(z_diff.shape,z_dist.shape)
        pt = self.scaler(z_dist)

        outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x,
                   "pt":pt, "targets":targets}


        return outputs
        
        
class SpeechVAE_Triplet(SpeechVAE):
    def __init__(self,style_dim=None, *args, **kwargs):
        super(SpeechVAE_Triplet,self).__init__(*args, **kwargs)
        if style_dim is None:
            self.s_dim = self.z_dim//2
        else:
            self.s_dim = style_dim
    def forward(self, batch):
       
        
        if isinstance(batch, dict):

            N = batch['s_sp'].shape[0]

            targets = torch.cat([batch['s_sp'],batch['s_xp'],batch['s_sx']],0)
            # encode
            qz_x, z = self.encode(targets)

            # decode
            px_z, x = self.decode(z)

        else:
            targets = batch
            N = batch.shape[0]
            # encode
            qz_x, z = self.encode(targets)

            # decode
            px_z, x = self.decode(z)


        outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x, 
        "targets":targets}


        return outputs
        
class SpeechVAE_Triplet_Pair(SpeechVAE_Triplet):
    def __init__(self, *args, **kwargs):
        super(SpeechVAE_Triplet_Pair,self).__init__(*args, **kwargs)
        self.scaler=nn.Linear(1,1,bias=False)
    def forward(self, batch):
       
        
        if isinstance(batch, dict):
            if 's_sp' in batch.keys():

                N = batch['s_sp'].shape[0]

                targets = torch.cat([batch['s_sp'],batch['s_xp'],batch['s_sx']],0)
                # encode
                qz_x, z = self.encode(targets)

                # decode
                px_z, x = self.decode(z)
                
                # scaler output
                z_diff = z[:-1,self.s_dim:]-z[1:,self.s_dim:]

            elif 's1' in batch.keys():

                N = batch['s1'].shape[0]

                targets = torch.cat([batch['s1'],batch['s2']],0)
                # encode
                qz_x, z = self.encode(targets)

                # decode
                px_z, x = self.decode(z)

                # scaler output
                z_diff = z[:N,self.s_dim:]-z[N:,self.s_dim:]
        else:
            targets = batch
            N = batch.shape[0]
            # encode
            qz_x, z = self.encode(targets)

            # decode
            px_z, x = self.decode(z)
            # scaler output
            z_diff = z[:-1,self.s_dim:]-z[1:,self.s_dim:]


        z_dist = torch.sqrt(torch.sum((z_diff)**2, -1, keepdim=True))
    #         print(z_diff.shape,z_dist.shape)
        pt = self.scaler(z_dist)
        
        outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x, 
                   "pt":pt,  "targets":targets}


        return outputs
        