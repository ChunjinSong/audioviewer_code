import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D 

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
import sklearn.metrics.pairwise as pw

from losses import featureLoss
from utils import *


def s2f_latent_dist(model_speech, model_face, sample, step=1, sr=16000):
    '''
    generate an video showing the animated faces, vowels, sound(if possible)
    '''

    if isinstance(model_speech, nn.Module):
        model_speech.eval();
    elif not isinstance(model_speech,sklearn.decomposition._base._BasePCA):
        raise TypeError("Audio model type is not supported.") 

    if isinstance(model_face,nn.Module):
        model_face.eval();
    elif not isinstance(model_face,sklearn.decomposition._base._BasePCA):
        raise TypeError("Visual model type is not supported.") 



#     load the sample
    # y, sr = librosa.load(sample['path'], sr=sr)
#     print(len(y)/sr)
    ori_mel = sample['spectrogram'].squeeze()
    ms = torch.from_numpy(ori_mel).unsqueeze(0).unsqueeze(0)

    sl=[]
    el=[]
    pl=[]

    with open(sample['path'].replace('.WAV','.TXT'), 'r') as f:
        _,_,sentence = f.readlines()[0].split(' ',2)
#         print(f'Sentence: {x}')
    with open(sample['path'].replace('.WAV','.PHN'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            s,e,p = line[:-1].split(' ',2)
    #         print(s,e,p)
            sl.append(float(s))
            el.append(float(e))
            pl.append(p)

    sl = np.asarray(sl)/sr
    el = np.asarray(el)/sr


    if isinstance(model_speech,nn.Module):
        mss= seg_data(ms,20,step).permute(0,1,3,2).cuda()
        N = mss.size()[0]
        mss_np = mss.squeeze().detach().cpu().numpy().reshape(N,-1)
        

        latent_mel = model_speech.encode(mss)[0][0]
        latent_np = latent_mel.squeeze().detach().cpu().numpy().reshape(N,-1)

    elif isinstance(model_speech,sklearn.decomposition._base._BasePCA):
        mss= seg_data(ms,20,step).permute(0,1,3,2).cuda()
        N = mss.size()[0]
        mss_np = mss.squeeze().view(N,-1).detach().cpu().numpy()
        
        latent_np = model_speech.transform(mss_np)
        latent_mel = torch.from_numpy(latent_np).float().cuda()


    if isinstance(model_face,nn.Module):
        face_mel = model_face.decode(latent_mel)
        face_np = face_mel.squeeze().view(N,-1).detach().cpu().numpy()

    elif isinstance(model_face,sklearn.decomposition._base._BasePCA):
        face_np = model_face.inverse_transform(latent_np)
        face_mel = torch.from_numpy(face_np.reshape(-1,3,64,64)).float().cuda()
    

    vects_list=[mss_np,latent_np,face_np]
    vname_list=["original_mel_segment","latent","face"]


    fig, axes = plt.subplots(4, 2, figsize=(10,15),sharex='all', sharey='all')

    fig.suptitle(sentence)
    axes[0,0].set_title("cosine similarity")
    axes[0,1].set_title("euclidian distance")

    for i in range(3):

        tmp = pw.cosine_similarity(vects_list[i])
        im=axes[i,0].imshow(tmp)
        axes[i,0].set_ylabel(vname_list[i])
        fig.colorbar(im, ax=axes[i,0])
        

        tmp = pw.euclidean_distances(vects_list[i])
        im=axes[i,1].imshow(tmp)
        fig.colorbar(im, ax=axes[i,1])



    fLoss=featureLoss()
    feature_dist = fLoss.pairwise(face_mel)
    im = axes[3,0].imshow(feature_dist)
    fig.colorbar(im, ax=axes[3,0])

    im=axes[3,1].imshow(feature_dist+
                        pw.euclidean_distances(vects_list[2])**2/vects_list[2].shape[-1])
    fig.colorbar(im, ax=axes[3,1])

    axes[3,0].set_title("feture loss")
    axes[3,1].set_title("feature loss + reconstruction loss")

    fLoss=None
    torch.cuda.empty_cache();
    return fig


def s2f_demo_gen(model_speech, model_face, sample, fps=25, sr=16000,mode='pad'):
    '''
    generate an video showing the animated faces, vowels, sound(if possible)
    '''
    if isinstance(model_speech,nn.Module):
        model_speech.eval();
    elif not isinstance(model_speech,sklearn.decomposition._base._BasePCA):
        raise TypeError("Audio model type is not supported.") 

    if isinstance(model_face,nn.Module):
        model_face.eval();
    elif not isinstance(model_face,sklearn.decomposition._base._BasePCA):
        raise TypeError("Visual model type is not supported.") 

#     load the sample
    # y, sr = librosa.load(sample['path'], sr=sr)
#     print(len(y)/sr)
    ori_mel = sample['spectrogram'].squeeze()
    ms = torch.from_numpy(ori_mel).unsqueeze(0).unsqueeze(0)

    FC,L = ori_mel.shape
    step = 100//fps
    ffps=float(fps)
    
    sl=[]
    el=[]
    pl=[]

    with open(sample['path'].replace('.WAV','.TXT'), 'r') as f:
        _,_,sentence = f.readlines()[0].split(' ',2)
#         print(f'Sentence: {x}')
    with open(sample['path'].replace('.WAV','.PHN'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            s,e,p = line[:-1].split(' ',2)
    #         print(s,e,p)
            sl.append(float(s))
            el.append(float(e))
            pl.append(p)
    
    sl = np.asarray(sl)/sr
    el = np.asarray(el)/sr
    
    mss= seg_data(ms,20,step).permute(0,1,3,2).cuda()
    N = mss.size()[0]
    mss_np = mss.squeeze().detach().cpu().numpy().reshape(N,-1)
    if isinstance(model_speech,nn.Module):
        
        latent_mel = model_speech.encode(mss)[0][0]
        latent_np = latent_mel.squeeze().detach().cpu().numpy().reshape(N,-1)

    elif isinstance(model_speech,sklearn.decomposition._base._BasePCA):

        latent_np = model_speech.transform(mss_np)
        latent_mel = torch.from_numpy(latent_np).float().cuda()


    if isinstance(model_face,nn.Module):
        face_mel = model_face.decode(latent_mel)

    elif isinstance(model_face,sklearn.decomposition._base._BasePCA):
        face_np = model_face.inverse_transform(latent_np)
        face_mel = torch.from_numpy(face_np.reshape(-1,3,64,64)).float().cuda()
    
    
    fig=plt.figure(figsize=(6,6), dpi= 80, facecolor='w', edgecolor='k')
    
    axes = []
    grid = plt.GridSpec(3,3, hspace=0.05, wspace=0.05)
    
    axes.append(fig.add_subplot(grid[:, 0]))
    axes.append(fig.add_subplot(grid[0:2,1:3]))
    axes.append(fig.add_subplot(grid[2, 1:3]))
    
    mel_seg = axes[0].imshow(np.zeros((80,20))-80,
                             cmap='viridis',
                             vmin=np.min(ori_mel),vmax=np.max(ori_mel),
                             origin='lower',)
    face  = axes[1].imshow(np.zeros((64,64,3)))
    phone_list = []
#     phone = axes[2].text(0,0,[],animated=True)


    fig.suptitle(sentence)


    # Define the animation function, which is called for each new frame:

    def animate(i, mel_dur=0.2, shift_hw=0.01):
        
        
        for ph in phone_list:
            ph.remove()
        phone_list[:]=[]
    
        st = i/ffps
        et = i/ffps+0.2
        
        ph_i = np.where((sl>=st) & (sl<et))[0]
#         print(ph_i)
        
            
        mel_seg.set_data(mss[i].squeeze().detach().cpu().numpy().T)
#         mel_seg.set_extent([i/ffps,i/ffps+0.2,0,80])
#         print(mel_seg.get_extent())
    
        for idx,pi in enumerate(ph_i):
            x = (sl[pi]-st)/shift_hw
#             print(idx,pi,sl[pi],x,pl[pi])
            ph = axes[0].text(x=x, y=65+(pi%3)*5,
                         s=pl[pi], color='red',fontsize=20)
            phone_list.append(ph)

        axes[2].clear()
        axes[2].axis('off')
        
        if len(ph_i)>0:
            axes[2].text(0.5,0.5,', '.join(pl[ph_i[0]:ph_i[-1]+1]),fontsize=40,
                         ha='center',va='center')

        face_data = face_mel[i].permute(1,2,0).squeeze().detach().cpu().numpy()/2+0.5
        face_data[face_data<0]=0
        face_data[face_data>1]=1
        face.set_data(face_data)
#     #     axes[0].imshow(mss[i].squeeze().detach().cpu().numpy().T)
#     #     axes[1].imshow(face_mel[i].permute(1,2,0).squeeze().detach().cpu().numpy()/2+0.5)

#         axes[0].set_xticks(np.linspace(i/ffps,i/ffps+0.2,2))
        axes[1].set_xticks([])
#         axes[0].set_yticks([])
        axes[1].set_yticks([])
        return (mel_seg,face,)

    # Compile the animation. Setting blit=True will only re-draw the parts that have changed.

    anim = animation.FuncAnimation(fig, animate, init_func=None,
                                   frames=N, interval=1000.0/ffps, 
                                   blit=True)


    
    return anim, face_mel



def plot_latent_traj(sample,model,
                     embed_step = 1,
                     vis_step = 1,
                     vec_range = None):

    '''
    plot the trajectory in the latent sapce of an utterance encoded by the model  
    '''

    if isinstance(model,nn.Module):
        model.eval();
    elif not isinstance(model,sklearn.decomposition._base._BasePCA):
        raise TypeError("Audio model type is not supported.") 

    
    ori_mel = sample['spectrogram'].squeeze()
    ms = torch.from_numpy(ori_mel).unsqueeze(0).unsqueeze(0)

    if isinstance(model,nn.Module):
        mss= seg_data(ms,20,embed_step).permute(0,1,3,2).cuda()
        
        latent_mel = model.encode(mss)[0][0].detach().cpu().numpy()

    elif isinstance(model,sklearn.decomposition._base._BasePCA):
        mss= seg_data(ms,20,embed_step).permute(0,1,3,2).squeeze().detach().numpy()
        N = mss.shape[0]
        mss_np = mss.squeeze().reshape(N,-1)
        
        latent_mel = model.transform(mss_np)


    if vec_range is not None:
        latent_mel =latent_mel[:,vec_range[0]:vec_range[1]]


    manifolds = [PCA(n_components=3, whiten=True),
                 TSNE(n_components=3),
                 MDS(n_components=3),
                 Isomap(n_components=3)]
    m_name = ['PCA','t-SNE','MDS','Isomap']


    fig = plt.figure(figsize=(10,10))
    cm = plt.get_cmap('viridis')
    for i in range(4):

        ax = fig.add_subplot(2,2,i+1,projection='3d')
        y = manifolds[i].fit_transform(latent_mel)
        NPOINTS = y[::vis_step].shape[0]

        ax.set_prop_cycle('color',[cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
        for j in range(NPOINTS-1):
    #         ax1.plot(x[i:i+2],y[i:i+2])
            ax.plot3D(y[vis_step*j:(vis_step*(j+2)),0],
                      y[vis_step*j:(vis_step*(j+2)),1],
                      y[vis_step*j:(vis_step*(j+2)),2])
    #     ax.plot3D(y[::vis_step,0],y[::vis_step,1],y[::vis_step,2])
        ax.scatter(y[0,0],y[0,1],y[0,2],c='r')
        ax.set_title(m_name[i])
    return fig

def s2f_gen(model_speech, model_face, sample, fps=25, sr=16000, mode='pad'):
    '''
    generate an video showing the animated faces
    '''
    if isinstance(model_speech,nn.Module):
        model_speech.eval();
    elif not isinstance(model_speech,sklearn.decomposition._base._BasePCA):
        raise TypeError("Audio model type is not supported.") 

    if isinstance(model_face,nn.Module):
        model_face.eval();
    elif not isinstance(model_face,sklearn.decomposition._base._BasePCA):
        raise TypeError("Visual model type is not supported.") 

#     print(len(y)/sr)
    ori_mel = sample['spectrogram'].squeeze()
    ms = torch.from_numpy(ori_mel).unsqueeze(0).unsqueeze(0)

    FC,L = ori_mel.shape
    step = 100//fps
    ffps=float(fps)
    
    
    mss= seg_data(ms,20,step,mode).permute(0,1,3,2).cuda()
    N = mss.size()[0]
    mss_np = mss.squeeze().detach().cpu().numpy().reshape(N,-1)

    if isinstance(model_speech,nn.Module):
        
        latent_mel = model_speech.encode(mss)[0][0]
        latent_np = latent_mel.squeeze().detach().cpu().numpy().reshape(N,-1)

    elif isinstance(model_speech,sklearn.decomposition._base._BasePCA):
        
        latent_np = model_speech.transform(mss_np)
        latent_mel = torch.from_numpy(latent_np).float().cuda()


    if isinstance(model_face,nn.Module):
        face_mel = model_face.decode(latent_mel)

    elif isinstance(model_face,sklearn.decomposition._base._BasePCA):
        face_np = model_face.inverse_transform(latent_np)
        face_mel = torch.from_numpy(face_np.reshape(-1,3,64,64)).float().cuda()
    
    
    fig=plt.figure(figsize=(6,6), dpi= 80, facecolor='w', edgecolor='k')
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax  = fig.add_subplot(111)
    
    face  = ax.imshow(np.zeros((64,64,3)))
    
    # Define the animation function, which is called for each new frame:

    def animate(i, mel_dur=0.2, shift_hw=0.01):

        face_data = face_mel[i].permute(1,2,0).squeeze().detach().cpu().numpy()/2+0.5
        face_data[face_data<0]=0
        face_data[face_data>1]=1
        face.set_data(face_data)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return (face,)

    # Compile the animation. Setting blit=True will only re-draw the parts that have changed.

    anim = animation.FuncAnimation(fig, animate, init_func=None,
                                   frames=N, 
                                   interval=1000.0/ffps, 
                                   blit=True)


    
    return anim, face_mel.detach()
