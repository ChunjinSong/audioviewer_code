import matplotlib.pyplot as plt
import torch

def vis_train_mappingA2V(vis_data, path_save, vis_sz=10):
    # vis_data = {'audio': outputs_audio,
    #             'visual': outputs_visual,
    #             'losses': self.losses_dict
    #             'scale_audio': [self.gmin, self.gmax]}
    fig = plt.figure(figsize=(40, 20), dpi=80, facecolor='w', edgecolor='k')
    fig.clf()
    grid = plt.GridSpec(12, vis_sz, hspace=0.05, wspace=0.05)
    gen_samples_audio = vis_data['audio']["px_z"][0][:vis_sz].squeeze().detach().cpu().numpy()
    ori_samples_audio = vis_data['audio']['targets'][:vis_sz].squeeze().detach().cpu().numpy()

    gen_samples_visual = vis_data['visual'][:vis_sz] * 0.5 + 0.5
    # gen_samples_visual = (gen_samples_visual*255).astype('uint8')
    gen_samples_visual = gen_samples_visual.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    # gen_samples_visual = gen_samples_visual.mul(255).add_(0.5).clamp_(0, 255).astype('unit8')
    # .squeeze().permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(vis_sz):
        axes = [fig.add_subplot(grid[0:2, i]),
                fig.add_subplot(grid[2:4, i])]

        axes[0].imshow(ori_samples_audio[i].T, cmap='viridis', vmin=vis_data['scale_audio'][0], vmax=vis_data['scale_audio'][1], origin='lower')
        axes[1].imshow(gen_samples_audio[i].T, cmap='viridis', vmin=vis_data['scale_audio'][0], vmax=vis_data['scale_audio'][1], origin='lower')

        axes[0].set_xticks([])
        axes[1].set_xticks([])
        if i > 0:
            axes[0].set_yticks([])
            axes[1].set_yticks([])

        axes = [fig.add_subplot(grid[4:6, i])]

        axes[0].imshow(gen_samples_visual[i])

        axes[0].set_xticks([])
        axes[0].set_yticks([])

    # axes = []
    # axes.append(fig.add_subplot(grid[6:10, :7]))

    # axes = [fig.add_subplot(grid[6:, :int(vis_sz/2)]),
    #         fig.add_subplot(grid[6:, int(vis_sz/2):])]
    # axes[0].plot([x for x in vis_data['losses']['loss_cycle']], label='train_cycle')
    # axes[0].set_yscale('log')
    # axes[0].set_xlabel("Gradient iterations")
    # axes[0].legend()
    #
    # axes[1].plot([x for x in vis_data['losses']['loss_smooth']], label='train_smooth')
    # axes[1].set_yscale('log')
    # axes[1].set_xlabel("Gradient iterations")
    # axes[1].legend()

    fig.savefig(path_save)
    plt.close(fig)


    # print()







def vis_test_audioRecon(audio_in, audio_recon, path_save, vis_sz=2):
    # vis_data = {'audio': outputs_audio,
    #             'visual': outputs_visual,
    #             'losses': self.losses_dict
    #             'scale_audio': [self.gmin, self.gmax]}
    fig = plt.figure(figsize=(40, 20), dpi=80, facecolor='w', edgecolor='k')
    fig.clf()
    grid = plt.GridSpec(12, vis_sz, hspace=0.05, wspace=0.05)

    gen_samples_audio = audio_recon[:vis_sz].squeeze().detach().cpu().numpy()
    ori_samples_audio = audio_in[:vis_sz].squeeze().detach().cpu().numpy()

    for i in range(vis_sz):
        axes = [fig.add_subplot(grid[0:2, i]),
                fig.add_subplot(grid[2:4, i])]

        axes[0].imshow(ori_samples_audio[i].T, cmap='viridis', vmin=-100, vmax=13, origin='lower')
        axes[1].imshow(gen_samples_audio[i].T, cmap='viridis', vmin=-100, vmax=13, origin='lower')

        axes[0].set_xticks([])
        axes[1].set_xticks([])
        if i > 0:
            axes[0].set_yticks([])
            axes[1].set_yticks([])

    fig.savefig(path_save)
    plt.close(fig)


    # print()





