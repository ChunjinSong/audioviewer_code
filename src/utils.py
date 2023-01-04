import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.cluster
import librosa
import sklearn
import imageio
import time, os, yaml


def seg_data(indata, seg_len, step=None,mode='tight'):

    '''
        segemnt specturm into patches by a specified step length
    '''
    seg_len = int(seg_len)
    if step is None:
        step=seg_len
    B,C,F,fn = indata.shape
    # print(B,C,F,fn)
    if fn < seg_len:
        return torch.nn.functional.pad(input=indata,pad=(0,seg_len-fn),
            mode='constant',value=-80)
    rem = (fn-seg_len)%step

    if mode == 'pad' and rem > 0:
        indata = torch.nn.functional.pad(input=indata,pad=(0,step-rem),
            mode='constant',value=-80)
        rem = 0;

    if rem==0:
        return indata.unfold(3,seg_len,step).permute(0,3,1,2,4).view(-1,C,F,seg_len)
    return torch.cat([indata[:,:,:,:int(fn-rem)].unfold(3,seg_len,step).permute(0,3,1,2,4).view(-1,C,F,seg_len),
                      indata[:,:,:,-seg_len:]],0)

def seg_data_si(indata, seg_len, step=None,mode='tight'):

    '''
        segemnt specturm into patches, retrun the indecis of starting frames as well
    '''
     
    seg_len = int(seg_len)
    if step is None:
        step=seg_len
    B,C,F,fn = indata.shape
    # print(B,C,F,fn)
    if fn < seg_len:
        return torch.nn.functional.pad(input=indata,pad=(0,seg_len-fn),
            mode='constant',value=-80),np.asarray([0])
    rem = (fn-seg_len)%step

    if mode == 'pad' and rem > 0:
        indata = torch.nn.functional.pad(input=indata,pad=(0,step-rem),
            mode='constant',value=-80)
        rem = 0;
    fn2=indata.shape[-1]
    if rem==0:
        return indata.unfold(3,seg_len,step).permute(0,3,1,2,4).view(-1,C,F,seg_len),\
               np.arange(0,fn2-seg_len+1,step)
    return torch.cat([indata[:,:,:,:int(fn-rem)].unfold(3,seg_len,step).permute(0,3,1,2,4).view(-1,C,F,seg_len),
                      indata[:,:,:,-seg_len:]],0), \
           np.append(np.arange(0, fn-seg_len,step),[fn-seg_len])


def seg_data_anno(indata,seg_len,phones, frame_dur=0.01, phone_step=1,margin=5,
    align_mode='start'):
    '''
     segemnt specturm into patches,centering at each annotated phone 
    '''
    seg_len = int(seg_len)

    B,C,F,fn = indata.shape

    if align_mode=='center':
        sib_list = np.floor((phones['start'][1:-1:phone_step] + 
                            phones['end'][1:-1:phone_step])/2/frame_dur
            ).astype(np.int32)-margin-seg_len//2
    else:
        sib_list = np.floor(phones['start'][:-1:phone_step]/frame_dur
            ).astype(np.int32)-margin
        

    sib_list[sib_list<0]=0
    sib_list[sib_list>fn-margin*2-seg_len]=fn-margin*2-seg_len
    
    mss=[indata[:,:,:,si_b:(si_b+seg_len+margin*2)] for si_b in sib_list]
    
    return torch.cat(mss),sib_list  

def visualize(patches,n=None):
    '''
        Plot spectrogram segments
    '''

    fig = plt.figure(figsize=(15, 4))
    if n is None:
        n = patches.shape[0]
    else:
        n = min(patches.shape[0],n)
    for i in range(n):
        inp = patches[i,:,:,:].squeeze()
        ax = fig.add_subplot(1, n, i+1, xticks=[], yticks=[])
        ax.imshow(inp.squeeze().detach().numpy(),cmap='viridis',vmin=inp.min(),vmax=inp.max())

def detect_filling(D_mel, ratio = 0.9, med_filt_sz=7):
    '''
    modified from 554X project, detecting the signal segment
    '''
    Df = scipy.signal.medfilt(np.mean(D_mel.squeeze().detach().numpy(),axis=-2),med_filt_sz)

    # cluseter with initial guess
    [centroids, l] = scipy.cluster.vq.kmeans2(Df, np.array([np.min(Df),np.max(Df)]));

    thr = np.min(centroids) * ratio + np.max(centroids)*(1-ratio)
    pi,_ = scipy.signal.find_peaks(Df,np.max(centroids))

    si = max(0,np.where(Df[:pi[0]]<thr)[0][-1]-10)
    ei = min(Df.shape[0], np.where(Df[pi[-1]:]>thr)[0][-1]+pi[-1]+10);
    # plt.plot(Df)
    # plt.axvspan(si, ei, facecolor='#2ca02c', alpha=0.1)

    return si, ei

def fold_mel(mss,target_len=None):
    
    '''
    the method fold a 3D tensor of mel spectrogram segments (B * 1 * seg_len * n_mel)
    back to a big 2D one (target_len * n_mel)
    '''
    B,C,seg_len, n_mel = mss.size()
    if target_len is None:
        target_len = seg_len*B
    rem = target_len % seg_len
    if rem==0:
        return mss.view(-1,n_mel)
    
    return torch.cat([mss[:-1].view(-1,n_mel),
                      mss[-1,:,-rem:,:].view(-1,n_mel)],0)        
        

def fold_mel_kernel(mss,target_len=None,step_len=None,temporal_kernel=None,eps=1e-5):
    '''
    The method fold a 3D tensor of mel spectrogram segments (B * 1 * seg_len * n_mel)
    back to a big 2D one (target_len * n_mel)
    The method support a weight kernel for smooth reconstruction
    '''
    B,C,seg_len, n_mel = mss.size()
    device = mss.device
    # set the default step length to 0
    if step_len is None:
        step_len = seg_len
        
    if target_len is None:
        target_len = seg_len+step_len*(B-1)
    elif target_len > seg_len + step_len*(B-1) \
        or target_len <= seg_len+step_len*(B-2) \
        or target_len < seg_len:
        raise ValueError(("target_len (%s) is not compatible with seg_len(%s) and step_len(%s) ")%(
            target_len, seg_len,step_len))
     
    # confirme the kernel is a vetor at the length of a segment
    if temporal_kernel is None:
        kernel = torch.ones(seg_len)
    else:
        kernel = temporal_kernel.flatten()
        if kernel.shape[0] != seg_len:
            raise ValueError(("kernel_length (%s) and seglen (%s) does not match" % (
              kernel.shape, seg_len)))
        
    
    # make it numaerically robust to zeros
    kernel = kernel.to(device)
    kernel = kernel.unsqueeze(-1)
    kernel += eps
    kernel /= kernel.sum()
    
    ret_mel = torch.zeros(target_len,n_mel).to(device)
    weight_sum = torch.zeros(target_len,1).to(device)

    for i in range(B):
        ei = min(target_len, seg_len + i*step_len)
        si = ei-seg_len
        
        ret_mel[si:ei,:] += mss[i,0,:]*kernel
        weight_sum[si:ei] += kernel
    ret_mel /= weight_sum
    
    return ret_mel      
        
def gauss_kernel_1d(mu=None, std=None, size=20):

    '''
        generate a gaussian kernel
    '''

    if mu is None:
        mu = (size-1)/2
    if std is None:
        std = size/4
    f = torch.exp(-((torch.arange(0,size) - mu) / std) ** 2 / 2)
    return f
        

def recon_mel(ms,model,seg_fn=20, step_len=None, kernel=None):

    '''
    input a melspectrogram, reconstruct it with the model
    '''

    if step_len is None:
        step_len = seg_fn

    if isinstance(model, nn.Module):
        model.eval();
    elif not isinstance(model,sklearn.decomposition._base._BasePCA):
        raise TypeError("Audio model type is not supported.") 

       
    ms = torch.from_numpy(ms).unsqueeze(0).unsqueeze(0)
#     print(ms.size())
    
    # print(ms.shape)
    mss = seg_data(ms,seg_fn,step_len).permute(0,1,3,2).cuda()
    
    if isinstance(model,nn.Module):
        mss_r = model(mss)['px_z'][0]

    elif isinstance(model,sklearn.decomposition._base._BasePCA):
        N = mss.size()[0]
        mss_np = mss.squeeze().detach().cpu().numpy().reshape(N,-1)
        z = model.transform(mss_np)
        mss_r = torch.from_numpy(model.inverse_transform(z)).view(-1,1,20,80)
    
    # ms_r = fold_mel(mss_r,ms.shape[-1]).permute(1,0).detach().cpu().numpy()
    ms_r = fold_mel_kernel(mss_r,ms.shape[-1],step_len,kernel).permute(1,0).detach().cpu().numpy()
    
    
    return ms_r


def mel_phase_to_sound(ms,P,**kwargs):

    '''
    generate sound with mel spectrogram and phase information
    '''
#     print(kwargs)
    M = librosa.feature.inverse.mel_to_stft(ms, sr=kwargs['sr'], n_fft=kwargs['n_fft'],
                                           fmin=kwargs['fmin'], fmax=kwargs['fmax'])
#     print(ms.shape,P.shape,M.shape)
    D = M*P
    y = librosa.core.istft(D, win_length=kwargs['win_length'],
                            hop_length=kwargs['hop_length'],)
    return y

def recon_sound(model, sample, sr=16000, n_fft=400, hop_len=160):

    '''
    rconstruct sound with the input model
    '''
    y, _ = librosa.load(sample['path'], sr=sr)
    
    ori_mel = sample['spectrogram'].squeeze()
    
    dec_mel = recon_mel(ori_mel,model)

    y_dec_mel = librosa.feature.inverse.mel_to_audio(
                librosa.core.db_to_power(dec_mel),sr=sr,
                n_fft=n_fft,win_length=n_fft,hop_length=hop_len,
                fmin=20,fmax=8000)

    D = librosa.core.stft(y,
                n_fft=n_fft,win_length=n_fft,hop_length=hop_len)
    M,P = librosa.core.magphase(D)

    y_dec_mel_phase = mel_phase_to_sound(librosa.core.db_to_power(dec_mel),P,sr=sr,
                n_fft=n_fft,win_length=n_fft,hop_length=hop_len,
                fmin=20, fmax=8000)
    
    return {'ys':{'y':y,
           'y_dec_mel':y_dec_mel,
           'y_dec_mel_phase':y_dec_mel_phase},
            'mels':{ 'ori_mel':ori_mel,
           'dec_mel':dec_mel}}

def mel_gen(wav,sr=16000, n_fft=400, hop_len=160, n_mels = 80,
                   f_min=20, f_max=8000, top_db=80):

    '''
        generate mel spectrogram of input sound wave
    '''

    S = librosa.feature.melspectrogram(y=wav,sr=sr,
                                n_fft=n_fft,
                                win_length=n_fft,
                                hop_length=hop_len,
                                n_mels=n_mels,
                                fmin=f_min,
                                fmax=f_max)
    S = librosa.core.power_to_db(S, ref=1.0, top_db=top_db) 
    return {'spectrogram': S}

def parse_utter_info(path):

    '''
    get the utterance information from its path
    '''
    uinfo_list = path.split('/')
    # print(uinfo_list)
    uinfo_dict = {'train': uinfo_list[-4].lower()=='train',
                'dialect_region': int(uinfo_list[-3][-1]),
                'sex': uinfo_list[-2][0],
                'ID': uinfo_list[-2],
                'sentence':uinfo_list[-1][:-4],
                }
    # print(uinfo_dict)
    return uinfo_dict

#===============add by scj====================
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0] // (size[1]), w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:j*(j+1), w*i:w*(i+1), :] = image
    return img

def save_images(images, size, image_path):
    # return scipy.misc.imsave(image_path, merge(images, size))
    return imageio.imwrite(image_path, merge(images, size))


def get_current_time():
    now = int(round(time.time()*1000))
    current_time = time.strftime('%H%M%S', time.localtime(now/1000))
    current_day = time.strftime('%Y_%m_%d', time.localtime(now/1000))
    return current_time, current_day

def write_config(config, outfile):
    with open(outfile, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def mkdir_output_train(args):
    GPU_ID = args['trainer']['GPU_ID']
    dir_output = args['output']['dir_output']

    current_time, current_day = get_current_time()

    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    dir_output = os.path.join(dir_output, current_day, str(gpu_id)+'_'+current_time)

    dir_log = os.path.join(dir_output, 'log')
    dir_model = os.path.join(dir_output, 'model')
    dir_img_result = os.path.join(dir_output, 'image')
    dir_config = os.path.join(dir_output, 'config')

    os.makedirs(dir_log, exist_ok=True)
    os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_img_result, exist_ok=True)
    os.makedirs(dir_config, exist_ok=True)

    write_config(args, os.path.join(dir_config,'test_config.yaml'))

    return dir_log, dir_model, dir_img_result

def mkdir_output_test(args):
    GPU_ID = args['GPU_ID']
    dir_output = args['output']['dir_output']

    current_time, current_day = get_current_time()

    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    dir_output = os.path.join(dir_output, current_day, str(gpu_id)+'_'+current_time)

    # dir_log = os.path.join(dir_output, 'log')
    # dir_model = os.path.join(dir_output, 'model')
    dir_img_result = os.path.join(dir_output, 'image')
    dir_img_video = os.path.join(dir_output, 'video')
    dir_config = os.path.join(dir_output, 'config')

    # os.makedirs(dir_log, exist_ok=True)
    # os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_img_result, exist_ok=True)
    os.makedirs(dir_img_video, exist_ok=True)
    os.makedirs(dir_config, exist_ok=True)

    write_config(args, os.path.join(dir_config,'test_config.yaml'))

    return dir_img_result, dir_img_video




