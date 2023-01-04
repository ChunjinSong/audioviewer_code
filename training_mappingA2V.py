import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import warnings,sys
from visualization import vis_train_mappingA2V
from utils_training import *
warnings.filterwarnings('ignore',category=FutureWarning)
sys.path.insert(0, './src/')
sys.path.insert(0, './style_soft_intro_vae/')
from torch.utils.tensorboard import SummaryWriter
from dataset_read import *
from style_soft_intro_vae import *
from models.mappingA2V import MappingA2V
from models.mappingV2A import MappingV2A
# sys.path.append(0,'../src/')
from models.dfcvae import *
from models.SpeechVAE import SpeechVAE, SpeechVAE_Pair, SpeechVAE_Triplet,SpeechVAE_Triplet_Pair
from losses import *




class Train(object):
    def __init__(self, args):
        print('====================================================')
        print('================ Train mapping =====================')
        print('====================================================')

        '''params'''
        self.z_dim_audio = args['trainer']['z_dim_audio']
        self.s_dim_audio = args['trainer']['s_dim_audio']
        self.z_dim_visual = args['trainer']['z_dim_visual']
        self.n_FBank = args['trainer']['n_FBank']

        self.epochs = args['trainer']['epochs']
        self.epoch = 0
        self.itern = 0
        self.itern_global = 0

        self.gpu_id = args['trainer']['GPU_ID']
        self.lr_map = args['trainer']['lr_map']
        self.betas_map = args['trainer']['betas_map']
        self.wd_map = args['trainer']['wd_map']
        self.eps_map = args['trainer']['eps_map']
        self.lr_decay_rate_map = args['trainer']['lr_decay_rate_map']

        self.scale1 = args['trainer']['scale1']
        self.scale2 = args['trainer']['scale2']
        self.scale3 = args['trainer']['scale3']

        self.weight_kl = args['trainer']['weight_kl']
        self.weight_cycle1 = args['trainer']['weight_cycle1']
        self.weight_cycle2 = args['trainer']['weight_cycle2']
        self.weight_iso1 = args['trainer']['weight_iso1']
        self.weight_iso2 = args['trainer']['weight_iso2']



        self.path_model_audio = os.path.join(args['trainer']['dir_model'], args['trainer']['path_model_audio'])
        self.config_video = args['trainer']['config_video']
        self.type = args['trainer']['type']

        self.plot_every = args['trainer']['plot_every']

        self.batch_size_single = args['data']['batch_size_single']
        self.batch_size_pair = args['data']['batch_size_pair']
        self.batch_size_triplet = args['data']['batch_size_triplet']

        # self.args_video = get_config(self.config_video)
        self.dir_log, self.dir_model, self.dir_img_result = mkdir_output_train(args)


        '''load dataset'''
        print('====================load dataset begin=========================')
        self.audio_loader_dict = get_audio_loader_dict(args)
        # self.visual_loader_dict, self.celeba_train, self.celeba_val = get_video_loader_dict(args)
        self.gmin, self.gmax = get_max_min_single(args)

        self.audio_dl_iterater = {}
        self.batch_audio = {}
        self.audio_dl_iterater['single'] = iter(self.audio_loader_dict['single']['train'])
        self.audio_dl_iterater['pair'] = iter(self.audio_loader_dict['pair']['train'])
        self.audio_dl_iterater['triplet'] = iter(self.audio_loader_dict['triplet']['train'])
        print('====================load dataset end===========================')


        '''load model'''
        print('====================load model begin=========================')
        self.model_audio = SpeechVAE_Triplet_Pair(self.s_dim_audio, self.n_FBank, self.z_dim_audio, nn.ReLU(True)).to(torch.device('cuda:0'))
        self.model_audio.load_state_dict(torch.load(self.path_model_audio))

        print('load model_audio successfully:' + self.path_model_audio)
        # self.params_audio = self.model_audio.parameters()

        self.model_visual = soft_intro_vae_init(self.config_video).to(torch.device('cuda:0'))
        # self.params_visual = self.model_visual.parameters()
        print('load model_visual successfully:' + self.path_model_audio)

        if self.type == 'content':
            dim_in = self.s_dim_audio
        elif self.type == 'style':
            dim_in = self.s_dim_audio
        else:
            dim_in = self.z_dim_audio

        self.model_mappingA2V = MappingA2V(dim_in, self.z_dim_visual).to(torch.device('cuda:0'))
        self.model_mappingV2A = MappingV2A(self.z_dim_visual, dim_in).to(torch.device('cuda:0'))
        # self.model_mappingA2V.load_state_dict(torch.load('/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step/2021_10_11/0_223333/model/MappingA2V_1281300.pth'))

        print('mapping_A2V model initialized.')
        print('====================load model end===========================')

        '''optimizer'''
        self.optimizer_mapA2V = optim.Adam(self.model_mappingA2V.parameters(), lr=self.lr_map, betas=self.betas_map,
                                           eps=self.eps_map, weight_decay=self.wd_map)
        self.scheduler_mapA2V = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_mapA2V,
                                                                     factor=self.lr_decay_rate_map)

        self.optimizer_mapV2A = optim.Adam(self.model_mappingV2A.parameters(), lr=self.lr_map, betas=self.betas_map,
                                           eps=self.eps_map, weight_decay=self.wd_map)
        self.scheduler_mapV2A = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_mapV2A,
                                                                     factor=self.lr_decay_rate_map)

        '''SummaryWriter'''
        self.writer = SummaryWriter(log_dir=self.dir_log)
        temp_input_map = torch.rand(8,dim_in)
        temp_input_map2A = torch.rand(8, self.z_dim_visual)
        temp_input_audio = {'s1': torch.rand(4, 1, 20, 80),
                            's2': torch.rand(4, 1, 20, 80),
                            'dt': torch.rand(4, 1)
                            }
        # temp_input_visual = torch.rand(8, 256, 256, 3)
        self.writer.add_graph(self.model_mappingA2V, temp_input_map)
        self.writer.add_graph(self.model_mappingV2A, temp_input_map2A)

        '''losses'''
        self.losses_dict = {'loss_cycle':[],
                       'loss_smooth':[]}

    def train(self):

        while self.epoch < self.epochs:
            '''load data'''
            try:
                self.batch_audio['single'] = next(self.audio_dl_iterater['single'])
                self.batch_audio['pair'] = next(self.audio_dl_iterater['pair'])
                self.batch_audio['triplet'] = next(self.audio_dl_iterater['triplet'])
            except:
                self.audio_dl_iterater['single'] = iter(self.audio_loader_dict['single']['train'])
                self.batch_audio['single'] = next(self.audio_dl_iterater['single'])

                self.audio_dl_iterater['pair'] = iter(self.audio_loader_dict['pair']['train'])
                self.batch_audio['pair'] = next(self.audio_dl_iterater['pair'])

                self.audio_dl_iterater['triplet'] = iter(self.audio_loader_dict['triplet']['train'])
                self.batch_audio['triplet'] = next(self.audio_dl_iterater['triplet'])

                self.epoch += 1
                self.itern = 0
                self.scheduler_mapA2V.step(losses)
                self.scheduler_mapV2A.step(losses)

            self.model_audio.eval()
            self.model_visual.eval()
            self.model_mappingA2V.train()
            self.model_mappingV2A.train()

            self.optimizer_mapA2V.zero_grad()
            self.optimizer_mapV2A.zero_grad()

            '''forward'''
            latents_audio_origin = {}
            latents_audio = {}
            latents_mapping = {}
            latents_mapping2A = {}
            outputs_visual = {}
            latents_visual = {}

            # latents_audio_origin['single'] = self.model_audio(self.batch_audio['single'])
            latents_audio_origin['pair'] = self.model_audio(self.batch_audio['pair'])
            latents_audio_origin['triplet'] = self.model_audio(self.batch_audio['triplet'])

            if self.type == 'content':
                # latents_audio['single'] = latents_audio_origin['single']['z'][:, self.s_dim_audio:]
                latents_audio['pair'] = latents_audio_origin['pair']['z'][:, self.s_dim_audio:]
                latents_audio['triplet'] = latents_audio_origin['triplet']['z'][:, self.s_dim_audio:]
            elif self.type == 'style':
                # latents_audio['single'] = latents_audio_origin['single']['z'][:, :self.s_dim_audio]
                latents_audio['pair'] = latents_audio_origin['pair']['z'][:, :self.s_dim_audio]
                latents_audio['triplet'] = latents_audio_origin['triplet']['z'][:, :self.s_dim_audio]
            else:
                # latents_audio['single'] = latents_audio_origin['single']['z']
                latents_audio['pair'] = latents_audio_origin['pair']['z']
                latents_audio['triplet'] = latents_audio_origin['triplet']['z']

            # latents_mapping['single'] = self.model_mappingA2V(latents_audio['single'])
            latents_mapping['pair'] = self.model_mappingA2V(latents_audio['pair'])
            latents_mapping['triplet'] = self.model_mappingA2V(latents_audio['triplet'])

            # s, outputs_visual['single'] = self.model_visual.generate(6, 1, z=latents_mapping['single'], mixing=False, noise=True, return_styles=True, no_truncation=True)
            s, outputs_visual['pair'] = self.model_visual.generate(6, 1, z=latents_mapping['pair'], mixing=False, noise=True, return_styles=True, no_truncation=True)
            s, outputs_visual['triplet'] = self.model_visual.generate(6, 1, z=latents_mapping['triplet'], mixing=False, noise=True, return_styles=True, no_truncation=True)

            # latents_visual['single'], mu_rec, logvar_rec = self.model_visual.encode(outputs_visual['single'].detach(), 6, 1)
            latents_visual['pair'], mu_rec, logvar_rec = self.model_visual.encode(outputs_visual['pair'].detach(), 6, 1)
            latents_visual['triplet'], mu_rec, logvar_rec = self.model_visual.encode(outputs_visual['triplet'].detach(), 6, 1)

            # latents_mapping2A['pair'] = self.model_mappingV2A(latents_visual['pair'])
            # latents_mapping2A['triplet'] = self.model_mappingV2A(latents_visual['triplet'])

            latents_mapping2A['pair'] = self.model_mappingV2A(latents_mapping['pair'])
            latents_mapping2A['triplet'] = self.model_mappingV2A(latents_mapping['triplet'])

            '''loss'''
            latent_loss_fn = nn.L1Loss()
            # loss_cycle = latent_loss_fn(latents_visual['single'], latents_mapping['single'])
            # loss_cycle += latent_loss_fn(latents_visual['pair'], latents_mapping['pair'])

            loss_cycle1 = latent_loss_fn(latents_visual['pair'], latents_mapping['pair'])
            loss_cycle1 += latent_loss_fn(latents_visual['triplet'], latents_mapping['triplet'])

            loss_cycle01 = latent_loss_fn(latents_mapping2A['pair'], latents_audio['pair'])
            loss_cycle01 += latent_loss_fn(latents_mapping2A['triplet'], latents_audio['triplet'])

            loss_kl = latent_loss_fn(latents_visual['pair'], torch.zeros(latents_visual['pair'].shape))
            loss_kl += latent_loss_fn(latents_visual['triplet'], torch.zeros(latents_visual['triplet'].shape))

            # loss_smooth1 = self.get_loss_smooth(latents_mapping['single'], latents_visual['single'], type = 'single', weight = 1)
            loss_cycle2 = self.get_loss_smooth(latents_mapping['pair'], latents_visual['pair'], type = 'pair', weight = self.scale1)
            loss_cycle2 += self.get_loss_smooth(latents_mapping['triplet'], latents_visual['triplet'], type = 'triplet', weight = self.scale1)
            #
            # loss_smooth2 = self.get_loss_smooth(latents_audio['single'], latents_mapping['single'], type='single', weight = 2)
            loss_iso1 = self.get_loss_smooth(latents_audio['pair'], latents_mapping['pair'], type='pair', weight = self.scale2)
            loss_iso1 += self.get_loss_smooth(latents_audio['triplet'], latents_mapping['triplet'], type='triplet', weight = self.scale2)

            # loss_smooth = loss_smooth1 + loss_smooth2

            loss_iso2 = self.get_loss_smooth_originSpace(self.batch_audio['pair'], outputs_visual['pair'], weight = self.scale3)

            loss_iso1 = self.weight_iso1 * loss_iso1 * 0.5
            loss_iso2 = self.weight_iso2 * loss_iso2
            loss_iso = loss_iso1 + loss_iso2

            loss_cycle1 = self.weight_cycle1 * loss_cycle1
            loss_cycle2 = self.weight_cycle2 * loss_cycle2 * 0.5
            loss_cycle = loss_cycle1 + loss_cycle01*0.0 + loss_cycle2

            loss_kl = self.weight_kl * loss_kl

            losses = loss_cycle + loss_iso + loss_kl

            losses.backward()

            nn.utils.clip_grad_norm_(self.model_mappingA2V.parameters(), max_norm=1.0, norm_type=2)
            nn.utils.clip_grad_norm_(self.model_mappingV2A.parameters(), max_norm=1.0, norm_type=2)

            self.optimizer_mapA2V.step()
            self.optimizer_mapV2A.step()

            '''summary'''
            self.writer.add_scalar(f'CycleLoss/cycle1', loss_cycle1, self.itern_global)
            self.writer.add_scalar(f'CycleLoss/cycle01', loss_cycle01, self.itern_global)
            # self.writer.add_scalar(f'CycleLoss/cycle2', loss_cycle2, self.itern_global)
            self.writer.add_scalar(f'CycleLoss/iso1', loss_iso1, self.itern_global)
            self.writer.add_scalar(f'CycleLoss/iso2', loss_iso2, self.itern_global)
            self.writer.add_scalar(f'Loss/loss_kl', loss_kl, self.itern_global)

            # for tag, param in self.model_mappingA2V.named_parameters():
            #     self.writer.add_histogram(tag+'_data', param.data, self.itern_global)
            #     self.writer.add_histogram(tag+'_grad', param.grad, self.itern_global)
            self.write_grad_summary(self.model_mappingA2V, self.writer, self.itern_global, 'mappingA2V')
            self.write_grad_summary(self.model_mappingV2A, self.writer, self.itern_global, 'mappingV2A')

            print("GPU:%s, Epoch:[%2d] [%6d], kl: %.8f, ll_cyc1: %.8f, ll_cyc01: %.8f, l_iso1: %.8f, l_iso2: %.8f" \
                  % (self.gpu_id, self.epoch, self.itern, loss_kl, loss_cycle1, loss_cycle01, loss_iso1, loss_iso2))

            self.itern += 1
            self.itern_global += 1

            '''visualization data & loss && save model'''
            if self.itern % self.plot_every == 0:
                '''visualization data & loss'''
                vis_path = os.path.join(self.dir_img_result, str(self.epoch)+ '_' + str(self.itern) + '.jpg')
                vis_data = {'audio':latents_audio_origin['pair'],
                            'visual':outputs_visual['pair'],
                            'losses':self.losses_dict,
                            'scale_audio':[self.gmin, self.gmax]}
                vis_train_mappingA2V(vis_data, vis_path, vis_sz=self.batch_size_pair*2)

            if self.itern % (20*self.plot_every) == 0:
                '''save model'''
                name_model_mapping = 'MappingA2V_' + str(self.epoch) + '_' + str(self.itern) +'.pth'
                path_model_mapping = os.path.join(self.dir_model, name_model_mapping)
                torch.save(self.model_mappingA2V.state_dict(), path_model_mapping)

                name_model_mapping = 'MappingV2A_' + str(self.epoch) + '_' + str(self.itern) + '.pth'
                path_model_mapping = os.path.join(self.dir_model, name_model_mapping)
                torch.save(self.model_mappingV2A.state_dict(), path_model_mapping)

                # print(f'Visual model saved to {path_model_mapping}')
                # self.writer.close()

        '''save model'''
        model_mapping = f'MappingA2V_' + str(self.epoch) + '_' + str(self.itern) + '.pth'
        model_path_mapping = os.path.join(self.dir_model, model_mapping)

        torch.save(self.model_mappingA2V.state_dict(), model_path_mapping)
        print(f'Visual model saved to {model_path_mapping}')
        print(f'finish training train_mappingA2V ')

        model_mapping = f'MappingV2A_' + str(self.epoch) + '_' + str(self.itern) + '.pth'
        model_path_mapping = os.path.join(self.dir_model, model_mapping)

        torch.save(self.model_mappingV2A.state_dict(), model_path_mapping)
        print(f'Visual model saved to {model_path_mapping}')
        print(f'finish training train_mappingV2A ')

        self.writer.close()


    def get_loss_smooth_originSpace(self, input_audio, output_visual, e = 1e-8, weight = 0.001):
        dt_audio = input_audio['dt']
        N = input_audio['dt'].shape[0]
        d_visual = output_visual[:N,:]- output_visual[N:,:]
        d_visual = d_visual.reshape(N, -1)
        d_visual = torch.sum(d_visual ** 2, -1, keepdim=True)

        d_visual = torch.sqrt(d_visual) * weight
        loss_smoothness = F.mse_loss(torch.log(torch.abs(d_visual + e)),
                   torch.log(torch.abs(dt_audio))) * 1 / N

        return loss_smoothness


    def get_loss_smooth(self, latent_audio, latents_visual, type = 'single', e = 1e-8, weight = 1):
        if type == 'triplet':
            N = int(self.batch_size_triplet)

            z1_audio = latent_audio[:N, :]
            z2_audio = latent_audio[N:2*N, :]
            z3_audio = latent_audio[2*N:3*N, :]

            z1_visual = latents_visual[:N, :]
            z2_visual = latents_visual[N:2*N, :]
            z3_visual = latents_visual[2*N:3*N, :]

            d1_audio = torch.sqrt(torch.sum((z1_audio - z2_audio) ** 2, -1, keepdim=True)) * weight
            d2_audio = torch.sqrt(torch.sum((z1_audio - z3_audio) ** 2, -1, keepdim=True)) * weight
            d3_audio = torch.sqrt(torch.sum((z2_audio - z3_audio) ** 2, -1, keepdim=True)) * weight

            d1_visual = torch.sqrt(torch.sum((z1_visual - z2_visual) ** 2, -1, keepdim=True))
            d2_visual = torch.sqrt(torch.sum((z1_visual - z3_visual) ** 2, -1, keepdim=True))
            d3_visual = torch.sqrt(torch.sum((z2_visual - z3_visual) ** 2, -1, keepdim=True))

            loss = F.mse_loss(torch.log(torch.abs(d1_visual + e)),
                                     torch.log(torch.abs(d1_audio)))

            loss += F.mse_loss(torch.log(torch.abs(d2_visual + e)),
                              torch.log(torch.abs(d2_audio)))

            loss += F.mse_loss(torch.log(torch.abs(d3_visual + e)),
                              torch.log(torch.abs(d3_audio)))
            loss = loss * 1/(3*N)

        elif type == 'pair':
            N = int(self.batch_size_pair)

            z1_audio = latent_audio[:N, :]
            z2_audio = latent_audio[N:2 * N, :]

            z1_visual = latents_visual[:N, :]
            z2_visual = latents_visual[N:2 * N, :]

            d1_audio = torch.sqrt(torch.sum((z1_audio - z2_audio) ** 2, -1, keepdim=True)) * weight

            d1_visual = torch.sqrt(torch.sum((z1_visual - z2_visual) ** 2, -1, keepdim=True))

            # d1_audio = self.model_audio.scaler(d1_audio)

            # d1_visual = self.model_audio.scaler(d1_visual)

            loss = F.mse_loss(torch.log(torch.abs(d1_visual + e)),
                              torch.log(torch.abs(d1_audio)))
            loss = loss * 1 / N

        else:
            N = int(self.batch_size_single / 2)

            z1_audio = latent_audio[:N, :]
            z2_audio = latent_audio[N:2 * N, :]

            z1_visual = latents_visual[:N, :]
            z2_visual = latents_visual[N:2 * N, :]

            d1_audio = torch.sqrt(torch.sum((z1_audio - z2_audio) ** 2, -1, keepdim=True)) * weight

            d1_visual = torch.sqrt(torch.sum((z1_visual - z2_visual) ** 2, -1, keepdim=True))

            # d1_audio = self.model_audio.scaler(d1_audio)

            # d1_visual = self.model_audio.scaler(d1_visual)

            loss = F.mse_loss(torch.log(torch.abs(d1_visual + e)),
                              torch.log(torch.abs(d1_audio)))

            loss = loss * 1 / N

        return loss

    # Add gradient summary into tensorboard
    def write_grad_summary(self, network, writer, epoch_num, name_net):
        for name, cur_para in network.named_parameters():
            # print("Grad name:%s", name)
            # tt = torch.norm(cur_para.grad.detach(), 1)
            writer.add_scalar(name_net+'/'+'Grad/%s_norm' % name, torch.norm(cur_para.grad.detach(), 2), epoch_num)
            writer.add_histogram(name_net+'/'+name, cur_para, epoch_num)
            writer.add_histogram(name_net+ '/' + name + '_grad', cur_para.grad, epoch_num)
        # print("Write gradients to tensorboard")



































