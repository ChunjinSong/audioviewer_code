import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os, sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
from utils_training import *
sys.path.insert(0,'./src/')
from models.SpeechVAE import SpeechVAE_Triplet_Pair
from losses import *
from runBuilder import *
from dataset_read import *
from itertools import chain


class Train(object):
    def __init__(self, args):
        print('==============================================')
        print('================ Train Audio =================')
        print('==============================================')

        self.n_FBank = args['trainer']['n_FBank']
        self.z_dim = args['trainer']['z_dim']

        self.lr = args['trainer']['lr']
        self.betas = args['trainer']['betas']
        self.eps = args['trainer']['eps']
        self.wd = args['trainer']['wd']
        self.lr_decay_rate = args['trainer']['lr_decay_rate']
        self.epochs = args['trainer']['epochs']
        self.plot_every = args['trainer']['plot_every']

        self.dir_log, self.dir_model, self.dir_img_result = mkdir_output_train(args)

        '''load data'''
        self.audio_loader_pair = get_audio_loader_pair(args)
        self.audio_loader_triplet = get_audio_loader_triplet(args)
        self.gmin, self.gmax = get_max_min_triplet(args)

        '''RunBuilder'''
        self.params = get_runs_params_audio(args)
        self.runs = RunBuilder.get_runs(self.params)

    def model_init(self, run):
        self.svae = None
        torch.cuda.empty_cache()
        self.svae = SpeechVAE_Triplet_Pair(run[2], self.n_FBank, self.z_dim, nn.ReLU(True)).to(torch.device('cuda:0'))
        self.itern = 0

        self.writer = SummaryWriter(log_dir=self.dir_log + '/' + f'{run[2]}_{run[0].__name__}_{run[1]}_{run[3]}',
                               comment=f'-{run}')

        '''loss'''
        self.train_losses = {'kld_loss': [],
                        'logpx_z': [],
                        'recomb_loss': [],
                        'pair_loss': []}
        self.val_losses = {'kld_loss': [],
                      'logpx_z': [],
                      'recomb_loss': [],
                      'pair_loss': []}

        self.scaler_list = []
        self.optimizer = optim.Adam(self.svae.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.lr_decay_rate)

        self.pair_dl_iter = iter(self.audio_loader_pair['train'])


    def model_train(self, run):
        for epoch in range(self.epochs):
            for batch in self.audio_loader_triplet['train']:
                self.optimizer.zero_grad()
                outputs, self.loss_dict = self.train_step_triplet(self.svae, batch, loss_fn,
                                                recomb_loss_func=run[0],
                                                recomb_loss_weight=run[1])

                try:
                    batch_pair = next(self.pair_dl_iter)
                except:
                    #                 epoch_a+=1
                    self.pair_dl_iter = iter(self.audio_loader_pair['train'])
                    batch_pair = next(self.pair_dl_iter)

                _, loss_dict_pair = self.train_step_pair(self.svae, batch_pair,
                                                    pair_loss_func=run.pair_loss_func,
                                                    pair_loss_weight=run.pair_loss_weight)
                # take a step after 2 losses get backpropgated
                self.optimizer.step()

                loss_dict = {**self.loss_dict, **loss_dict_pair}

                for k in loss_dict:
                    self.train_losses[k].append(loss_dict[k])
                    self.writer.add_scalar(f'Loss/Train/{k}', loss_dict[k], self.itern)
                self.itern += 1
                if self.itern % self.plot_every == 1:
                    batch_val, outputs_val, loss_val_dict = self.validate_triplet(self.svae, self.audio_loader_triplet['val'], loss_fn,
                                                                     recomb_loss_func=run[0])
                    _, _, loss_val_dict_pair = self.validate_pair(self.svae, self.audio_loader_pair['val'],
                                                             pair_loss_func=run.pair_loss_func,
                                                             pair_loss_weight=run.pair_loss_weight)
                    loss_val_dict = {**loss_val_dict, **loss_val_dict_pair}

                    '''save log info'''
                    loss_val = 0.0
                    for k in loss_val_dict:
                        self.val_losses[k].append(loss_val_dict[k])
                        loss_val += loss_val_dict[k]
                        self.writer.add_scalar(f'Loss/Val/{k}', loss_val_dict[k], self.itern)
                    self.writer.add_scalar(f'Loss/Val/total', loss_val, self.itern)

                    if run.pair_loss_func is None:
                        self.scaler_list.append(0)
                    else:
                        self.scaler_list.append(self.svae.scaler.weight.item())
                    self.writer.add_scalar(f'Weight/scaler', self.scaler_list[-1], self.itern)

                    print(f'-{run}')
                    print("Plot after epoch {} ++++ iteration {}), last validation loss = {}, ++++ current lr = {} ++++ scaler = {},".format(
                            epoch, self.itern,
                            [(k, v[-1]) for k, v in self.val_losses.items()],
                            [group['lr'] for group in self.optimizer.param_groups],
                            self.scaler_list[-1]))

                    '''save image results'''
                    # vis_sz = 16*3 #outputs_val['targets'].shape[0]
                    # N = outputs_val['targets'].shape[0] // 3
                    # vis_idx = list(chain(*[[i, i + N, i + 2 * N] for i in range(vis_sz // 3)]))
                    # vis_idx_2 = list(chain(*[[i, i, i] for i in range(vis_sz // 3)]))
                    #
                    # #     print(vis_idx)
                    # ori_samples = outputs_val['targets'][vis_idx].squeeze().detach().cpu().numpy()
                    # gen_samples = outputs_val["px_z"][0][vis_idx].squeeze().detach().cpu().numpy()
                    # recomb_samples = outputs_val["px_z_recomb"][0][vis_idx_2].squeeze().detach().cpu().numpy()
                    #
                    # #outputs,outputs_val
                    # list_img_temp = []
                    # for A, B, C in zip(#outputs['targets'], outputs['px_z'], outputs['px_z_recomb'],
                    #                             ori_samples, gen_samples, recomb_samples):
                    #     list_img_temp.append(A)
                    #     list_img_temp.append(B)
                    #     list_img_temp.append(C)
                    #     # list_img_temp.append(D)
                    #     # list_img_temp.append(E)
                    #     # list_img_temp.append(F)
                    #
                    # array_img_out = np.array(list_img_temp, dtype=np.float32)
                    # num = 3
                    # save_images(array_img_out, [int(num * vis_sz), int(num)], '{}/{:02d}_{:06d}.jpg'.format(self.dir_img_result, epoch, self.itern))

                    self.train_vis(outputs_val, self.train_losses, self.val_losses, self.plot_every, self.itern, 50)

            self.scheduler.step(loss_val)
            if self.optimizer.param_groups[0]['lr'] < 1e-7:
                break

        vl = 0
        for k, v in self.val_losses.items():
            vl += v[-1]

        model_name = f'SRPVAE_{run[2]}_{run[0].__name__}_{run[1]}_{run[3]}_' + f'{run.pair_loss_func.__name__}_{run.pair_loss_weight}_{epoch + 1}_{vl}.pth'
        model_path = os.path.join(self.dir_model, model_name)

        torch.save(self.svae.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        self.writer.close()

    def train(self):
        for run in self.runs:
            '''load model'''
            self.model_init(run)
            self.model_train(run)

    def train_step_triplet(self, model, batch, loss_func,
                   recomb_loss_func, recomb_loss_weight=1):

        '''
        training step for recombination
        '''
        model.train()

        outputs = model(batch)

        vae_loss, loss_dict = loss_func(outputs, outputs["targets"])
        #     recomb_loss, outputs['px_z_recomb'] = recomb_loss_func(model,outputs,xp_weight)
        recomb_loss, outputs['px_z_recomb'] = recomb_loss_func(model, outputs)
        if isinstance(recomb_loss, list):
            recomb_loss[0] *= recomb_loss_weight
            recomb_loss[1] *= recomb_loss_weight
            loss = recomb_loss[0] + recomb_loss[1] + torch.mean(torch.sum(kld(*outputs['qz_x']), dim=1))

            loss_dict['recomb_loss'] = recomb_loss[0].item() + recomb_loss[1].item()
        else:
            recomb_loss *= recomb_loss_weight
            #     loss = vae_loss + recomb_loss
            loss = recomb_loss + torch.mean(torch.sum(kld(*outputs['qz_x']), dim=1))
            loss_dict['recomb_loss'] = recomb_loss.item()
        #     optimizer.zero_grad()
        loss.backward()
        #     optimizer.step()

        return outputs, loss_dict

    def train_step_pair(self, model, batch, pair_loss_func, pair_loss_weight=1e3):
        model.train()

        outputs = model(batch)
        pair_loss = pair_loss_func(outputs, batch) * pair_loss_weight
        #     reconstrction loss is calculated in the recombined way, so we just calculate the pair loss
        #     vae_loss, loss_dict = loss_func(outputs, outputs["targets"])
        loss = pair_loss  # + vae_loss

        self.loss_dict['pair_loss'] = pair_loss.item()

        #     optimizer.zero_grad()
        loss.backward()
        #     optimizer.step()

        return outputs, self.loss_dict

    def validate_triplet(self, model, loader, loss_func, recomb_loss_func):
        '''
        validation step for recombination
        '''
        model.eval()

        loss_dict = {'kld_loss': 0,
                     'logpx_z': 0,
                     'recomb_loss': 0}

        with torch.no_grad():
            for batch in loader:
                outputs = model(batch)
                #         loss += (loss_func(outputs, outputs['targets']) + \
                #                  pair_loss(outputs,batch)).item() * batch['s1'].shape[0]

                recomb_loss, outputs['px_z_recomb'] = recomb_loss_func(model, outputs)
                loss_dict['recomb_loss'] += recomb_loss.item() * batch['s_sp'].shape[0]  # *recomb_loss_weight

                _, ld = loss_func(outputs, outputs["targets"])
                for k in ld:
                    loss_dict[k] += ld[k] * batch['s_sp'].shape[0]

        #     loss/=len(loader.dataset)
        for k in loss_dict:
            loss_dict[k] /= len(loader.dataset)

        return batch, outputs, loss_dict

    def validate_pair(self, model, loader,
                      pair_loss_func, pair_loss_weight=1e3):
        model.eval()

        loss_dict = {'pair_loss': 0}

        with torch.no_grad():
            for batch in loader:
                outputs = model(batch)
                #         loss += (loss_func(outputs, outputs['targets']) + \
                #                  pair_loss(outputs,batch)).item() * batch['s1'].shape[0]

                loss_dict['pair_loss'] += pair_loss_func(outputs, batch).item() * batch['s1'].shape[
                    0] * pair_loss_weight

        #             _, ld = loss_func(outputs, outputs["targets"])
        #             for k in ld:
        #                 loss_dict[k] += ld[k] * batch['s1'].shape[0]

        #     loss/=len(loader.dataset)
        for k in loss_dict:
            loss_dict[k] /= len(loader.dataset)

        return batch, outputs, loss_dict

    def train_vis(self, gen_batch,
                  train_losses, val_losses, plot_every, itern,
                  recent=30, vis_sz=15):
        '''
        plot the recombination
        '''
        fig = plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
        fig.clf()

        grid = plt.GridSpec(4, vis_sz, hspace=0.05, wspace=0.05)

        N = gen_batch['targets'].shape[0] // 3
        vis_idx = list(chain(*[[i, i + N, i + 2 * N] for i in range(vis_sz // 3)]))
        vis_idx_2 = list(chain(*[[i, i, i] for i in range(vis_sz // 3)]))

        #     print(vis_idx)
        ori_samples = gen_batch['targets'][vis_idx].squeeze().detach().cpu().numpy()
        gen_samples = gen_batch["px_z"][0][vis_idx].squeeze().detach().cpu().numpy()
        recomb_samples = gen_batch["px_z_recomb"][0][vis_idx_2].squeeze().detach().cpu().numpy()

        for i in range(vis_sz):
            axes = [fig.add_subplot(grid[0, i]),
                    fig.add_subplot(grid[1, i]),
                    fig.add_subplot(grid[2, i])]

            axes[0].imshow(ori_samples[i].T, cmap='viridis', vmin=self.gmin, vmax=self.gmax, origin='lower')
            axes[1].imshow(gen_samples[i].T, cmap='viridis', vmin=self.gmin, vmax=self.gmax, origin='lower')
            axes[2].imshow(recomb_samples[i].T, cmap='viridis', vmin=self.gmin, vmax=self.gmax, origin='lower')

            axes[0].set_xticks([])
            axes[1].set_xticks([])
            axes[2].set_xticks([])
            if i > 0:
                axes[0].set_yticks([])
                axes[1].set_yticks([])
                axes[2].set_yticks([])

        axes = [0, 0]
        if itern >= recent * plot_every:
            axes.append(fig.add_subplot(grid[-1, :10]))
            axes.append(fig.add_subplot(grid[-1, 10:]))
        else:
            axes.append(fig.add_subplot(grid[-1, :]))

        for k in train_losses:
            axes[2].plot(train_losses[k], label='train_' + k)
            #     print(itern,plot_every)
            axes[2].plot(range(0, itern, plot_every), val_losses[k], label='val_' + k)
        axes[2].set_yscale('log')
        #     axes[2].set_title('Losses')
        axes[2].set_xlabel("Gradient iterations")

        if itern >= (recent + 1) * plot_every:
            for k in train_losses:
                axes[3].plot(range(itern - recent * plot_every, itern), train_losses[k][-(plot_every * recent):],
                             label='train_' + k)
                #         print(itern,plot_every)
                axes[3].plot(range(itern - recent * plot_every - 1, itern, plot_every), val_losses[k][-(recent + 1):],
                             label='val_loss' + k)
            axes[3].set_yscale('log')
            axes[3].legend()
            #     axes[3].set_title('Recent Losses')
            axes[3].set_xlabel("Gradient iterations")
        else:
            axes[2].legend()

        fig.savefig(os.path.join(self.dir_img_result, str(self.itern)+'.jpg'))















