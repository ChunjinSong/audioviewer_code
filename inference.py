import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import subprocess
import gc
import yaml

sys.path.insert(0, './src/')
sys.path.insert(0, './style_soft_intro_vae/')
from melDatasets import *
from models.SpeechVAE import *
from visualization import *
from utils import *
from models.dfcvae import *
from models.SpeechVAE import SpeechVAE, SpeechVAE_Pair, SpeechVAE_Triplet, SpeechVAE_Triplet_Pair
from losses import *
from models.mappingA2V import MappingA2V
from models.mappingV2A import MappingV2A
from style_soft_intro_vae import * 
from eval_crit import *
from dataset_read import *
import time
from torchsummary import summary


class Inference(object):
    def __init_mel__(self, args):

        '''params'''
        self.path_model_audio = args['model']['path_audio']
        self.path_model_visual = args['model']['path_visual']
        self.config_visual = args['model']['config_visual']
        self.path_model_mappingA2V = args['model']['path_mappingA2V']

        self.z_dim_audio = args['model']['z_dim_audio']
        self.s_dim_audio = args['model']['s_dim_audio']
        self.z_dim_visual = args['model']['z_dim_visual']
        self.n_FBank = args['model']['n_FBank']

        self.gpu_id = args['GPU_ID']
        self.type = args['model']['type']

        self.batch_size_single = args['data']['batch_size_single']
        self.batch_size_pair = args['data']['batch_size_pair']
        self.batch_size_triplet = args['data']['batch_size_triplet']

        self.idx_img_save = 0

        '''load dataset'''
        self.tms_ds = TIMIT_RawMel_Dataset_Triplet('/local/mparmis/datasets/TIMIT_spec/specs/data.pkl',
                                                   '/local/mparmis/datasets/TIMIT_spec/specs/utter_info_timit.pkl')

        self.gmin, self.gmax = get_max_min_single(args)

    def __init__(self, args):

        '''params'''
        self.path_model_audio = args['model']['path_audio']
        self.path_model_visual = args['model']['path_visual']
        self.config_visual = args['model']['config_visual']
        self.path_model_mappingA2V = args['model']['path_mappingA2V']

        self.z_dim_audio = args['model']['z_dim_audio']
        self.s_dim_audio = args['model']['s_dim_audio']
        self.z_dim_visual = args['model']['z_dim_visual']
        self.n_FBank = args['model']['n_FBank']

        self.gpu_id = args['GPU_ID']
        self.type = args['model']['type']

        self.batch_size_single = args['data']['batch_size_single']
        self.batch_size_pair = args['data']['batch_size_pair']
        self.batch_size_triplet = args['data']['batch_size_triplet']

        self.idx_img_save = 0

        '''load dataset'''
        self.tms_ds = TIMIT_RawMel_Dataset_Triplet('/local/mparmis/datasets/TIMIT_spec/specs/data.pkl',
                                                   '/local/mparmis/datasets/TIMIT_spec/specs/utter_info_timit.pkl')

        # self.sample_triplet = self.tms_ds[0]

        '''load model'''
        print('====================load model begin=========================')
        self.model_audio = SpeechVAE_Triplet_Pair(self.s_dim_audio, self.n_FBank, self.z_dim_audio, nn.ReLU(True)).to(
            torch.device('cuda:0'))
        self.model_audio.load_state_dict(torch.load(self.path_model_audio))
        print('load model_audio successfully:' + self.path_model_audio)

        self.model_visual = soft_intro_vae_init(self.config_visual).to(torch.device('cuda:0'))
        print('load model_visual successfully:' + self.path_model_audio)

        if self.type == 'content':
            dim_in = self.s_dim_audio
        elif self.type == 'style':
            dim_in = self.s_dim_audio
        else:
            dim_in = self.z_dim_audio

        self.model_mappingA2V = MappingA2V(dim_in, self.z_dim_visual).to(torch.device('cuda:0'))
        # self.path_model_mappingA2V = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_06/0_140859/model/MappingA2V_10_1.pth'
        # self.model_mappingA2V.load_state_dict(torch.load(self.path_model_mappingA2V))

        # path_a2v_all = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_10_24/0_194248/model/MappingA2V_5_3200.pth'
        # self.model_mappingA2V.load_state_dict(torch.load(path_a2v_all))

        print('mapping_A2V model initialized.')
        print('====================load model end===========================')

        self.model_audio.eval()
        self.model_visual.eval()
        self.model_mappingA2V.eval()

        self.model_mappingV2A = MappingV2A(self.z_dim_visual, dim_in).to(torch.device('cuda:0'))
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_09/0_230753/model/MappingV2A_10_1.pth'
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_10/0_230955/model/MappingV2A_10_1.pth'
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_12/0_131604/model/MappingV2A_10_1.pth'
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_06/0_140859/model/MappingV2A_10_1.pth'
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_13/0_005311/model/MappingV2A_1000_1.pth'
        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_13/0_132838/model/MappingV2A_100_1.pth'

        # self.path_model_mappingV2A = '/local/chunjins/project/AudioViewer/output/train/recomb_pair_256_3step_content/2021_11_15/0_114441/model/MappingV2A_575_181.pth'
        # self.model_mappingV2A.load_state_dict(torch.load(self.path_model_mappingV2A))
        # self.model_mappingV2A.eval()

        config_dfc = '/local/chunjins/project/AudioViewer/src/configs/dfc_vae.yaml'
        with open(config_dfc, 'r') as stream:
            config_dfc_args = yaml.full_load(stream)

        # config_dfc_args = yaml.load(config_dfc, Loader=yaml.FullLoader)
        print(int(config_dfc_args['model_params']['in_channels']))
        # self.model_visual_DFC = DFCVAE(in_channels=int(config_dfc_args['model_params']['in_channels']),
        #                       latent_dim=int(config_dfc_args['model_params']['latent_dim'])).to(torch.device('cuda:0'))
        # visual_model_path = '/local/yuchi/Speech2Face/model/DVAE_face_38_0.27617739126207164.pth'
        # self.model_visual_DFC.load_state_dict(torch.load(visual_model_path))

        # print('model_audio======================================================')
        # print(self.model_audio)
        # print('model_visual_HR======================================================')
        # print(self.model_visual)
        # print('model_visual_LR======================================================')
        # print(self.model_visual_DFC)
        #
        # print('model_A2V======================================================')
        # print(self.model_mappingA2V)
        # print('model_V2A======================================================')
        # print(self.model_mappingV2A)
        # print('end')

        # print('vector_audio============================================')
        # summary(self.model_audio, (20,80), batch_size=1, device='cuda:0')
        # summary(self.model_visual, (3, 256, 256))
        # print('vector_visual============================================')

    def predict_test(self, dir_result):
        self.dir_result = dir_result
        self.idx_img_save = 0
        index_time = 0
        time_all = 0
        with torch.no_grad():
            for i in range(0, 20):
                self.sample_triplet = self.tms_ds[i]
                self.idx_img_save = 0
                for sample in self.sample_triplet:
                    ori_mel = sample['spectrogram'].squeeze()
                    demo_name = f'{sample["ID"]}_{sample["sentence_ID"]}_{self.type}'
                    path_out = os.path.join(self.dir_result[1], demo_name)
                    dir_out_img = os.path.join(self.dir_result[0], demo_name)
                    os.makedirs(dir_out_img, exist_ok=True)
                    self.idx_img_save = 0
                    ms = torch.from_numpy(ori_mel).clone().unsqueeze(0).unsqueeze(0)

                    fps = 25
                    sr = 16000
                    FC, L = ori_mel.shape
                    step = 100 // fps
                    ffps = float(fps)

                    sl = []
                    el = []
                    pl = []

                    with open(sample['path'].replace('.WAV', '.TXT'), 'r') as f:
                        _, _, sentence = f.readlines()[0].split(' ', 2)
                    #         print(f'Sentence: {x}')
                    with open(sample['path'].replace('.WAV', '.PHN'), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            s, e, p = line[:-1].split(' ', 2)
                            #         print(s,e,p)
                            sl.append(float(s))
                            el.append(float(e))
                            pl.append(p)

                    sl = np.asarray(sl) / sr
                    el = np.asarray(el) / sr

                    mss = seg_data(ms, 20, step).permute(0, 1, 3, 2).cuda()

                    mss = mss[0, :, :, :]
                    shape = mss.size()
                    mss = mss.reshape([1, shape[0], shape[1], shape[2]])

                    N = mss.size()[0]

                    # mss_np = mss.squeeze().detach().cpu().numpy().reshape(N, -1)
                    start = time.time()
                    latent_mel = self.model_audio.encode(mss)[0][0]

                    if self.type == 'content':
                        latent_mel = latent_mel[:, self.s_dim_audio:]
                    elif self.type == 'style':
                        latent_mel = latent_mel[:, self.s_dim_audio]

                    latents_mapping = self.model_mappingA2V(latent_mel)

                    # face_list = []
                    # for i in range(latent_mel.shape[0]):
                    #     face_list.append(self.model_visual.generate(6, 1, latent_mel[i:i + 1].detach().cuda(), 1, mixing=True))
                    # face_mel = torch.cat(face_list)
                    # print(face_mel.shape)

                    # z = torch.randn(latents_mapping.shape[0], latents_mapping.shape[1])

                    # latents_mapping = 0.5* latents_mapping + z* 0.5

                    # outputs_visual = self.model_visual_DFC.decode(latent_mel)

                    s, outputs_visual = self.model_visual.generate(6, 1, z=latents_mapping, mixing=False,
                                                                   noise=True, return_styles=True,
                                                                   no_truncation=True)

                    end = time.time()
                    index_time += 1
                    time_all += end - start
                    print('time:', index_time, time_all / index_time)
                    fig = plt.figure(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')

                    axes = []
                    grid = plt.GridSpec(3, 3, hspace=0.05, wspace=0.05)

                    axes.append(fig.add_subplot(grid[:, 0]))
                    axes.append(fig.add_subplot(grid[0:2, 1:3]))
                    axes.append(fig.add_subplot(grid[2, 1:3]))

                    mel_seg = axes[0].imshow(np.zeros((80, 20)) - 80,
                                             cmap='viridis',
                                             vmin=np.min(ori_mel), vmax=np.max(ori_mel),
                                             origin='lower', )
                    face = axes[1].imshow(np.zeros((64, 64, 3)))
                    phone_list = []
                    #     phone = axes[2].text(0,0,[],animated=True)

                    fig.suptitle(sentence)

                    def animate(i, mel_dur=0.2, shift_hw=0.01):

                        for ph in phone_list:
                            ph.remove()
                        phone_list[:] = []

                        st = i / ffps
                        et = i / ffps + 0.2

                        ph_i = np.where((sl >= st) & (sl < et))[0]
                        #         print(ph_i)

                        mel_seg.set_data(mss[i].squeeze().detach().cpu().numpy().T)
                        #         mel_seg.set_extent([i/ffps,i/ffps+0.2,0,80])
                        #         print(mel_seg.get_extent())

                        for idx, pi in enumerate(ph_i):
                            x = (sl[pi] - st) / shift_hw
                            #             print(idx,pi,sl[pi],x,pl[pi])
                            ph = axes[0].text(x=x, y=65 + (pi % 3) * 5,
                                              s=pl[pi], color='red', fontsize=20)
                            phone_list.append(ph)

                        axes[2].clear()
                        axes[2].axis('off')

                        if len(ph_i) > 0:
                            axes[2].text(0.5, 0.5, ', '.join(pl[ph_i[0]:ph_i[-1] + 1]), fontsize=40,
                                         ha='center', va='center')

                        face_data = outputs_visual[i].permute(1, 2, 0).squeeze().detach().cpu().numpy() / 2 + 0.5
                        face_data[face_data < 0] = 0
                        face_data[face_data > 1] = 1
                        face.set_data(face_data)

                        axes[1].set_xticks([])
                        axes[1].set_yticks([])
                        path_save = os.path.join(dir_out_img, str(self.idx_img_save) + '.jpg')
                        fig.savefig(path_save)
                        self.idx_img_save += 1
                        return (mel_seg, face,)

                    # Compile the animation. Setting blit=True will only re-draw the parts that have changed.

                    anim = animation.FuncAnimation(fig, animate, init_func=None,
                                                   frames=N, interval=1000.0 / ffps,
                                                   blit=True)

                    # write the generated video to file
                    anim.save(f'{path_out}_muted.mp4', writer='ffmpeg', bitrate=1.5e5, codec="libx264",
                              extra_args=['-pix_fmt', 'yuv420p'])
                    # #                         break
                    plt.close()

                    # add  original utterance as soundtrack
                    cmd = f'ffmpeg -y -i {path_out}_muted.mp4 -i {sample["path"]}  -vcodec copy -acodec  aac -ab 320k \
                                           -map 0:0 -map 1:0 {path_out}.mp4'

                    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                    result.stdout.decode('utf-8')

                    # recycle memory
                    gc.collect()













