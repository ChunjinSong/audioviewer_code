from utils import *
import h5py
import torch
import os, sys
import numpy as np
import pickle as pkl
import sys
from melDatasets import TIMIT_RawMel_Dataset 

import random

# save the unterance information from annotation
def gen_utter_info(rawMel_dataset):
    utter_info=[]
    for sample in rawMel_dataset:
        uinfo_dict = parse_utter_info(sample['path'])
        sl=[]
        el=[]
        pl=[]

        with open(sample['path'].replace('.WAV','.PHN'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                s,e,p = line[:-1].split(' ',2)
                sl.append(float(s))
                el.append(float(e))
                pl.append(p)
        st = np.array(sl)/sr
        et = np.array(el)/sr

        uinfo_dict['phones'] = {'start':st,
                                'end':et,
                                'phone':np.array(pl)}
        utter_info.append(uinfo_dict)
     
    return utter_info

# generate mel spactrogram datasets of a specified split
def mel_to_patches(argv):

    if len(argv)<=1:
        mode = 'ALL'
    elif argv[1]=='test':
        mode = 'TEST'
    elif argv[1]=='train':
        mode = 'TRAIN'
    elif argv[1] == 'dev':
        mode = 'KALDI_DEV'
        with open("../data/processed/config/dev_spk.list") as f:
            spk_lst = f.read().split()
    elif argv[1] == 'ktest':
        mode = 'KALDI_TEST'
        with open("../data/processed/config/test_spk.list") as f:
            spk_lst = f.read().split()
    else:
        mode = 'ALL'

    if len(argv)<=2:
        align_mode = 'start'
    elif argv[2]=='center':
        align_mode = 'center'
    else:
        align_mode ='start'

    min_dB = -20
    len_hw = 25/1000.0
    shift_hw = 10/1000.0
    n_FBank = 80
    sr = 16000
    z_dim = 128
    margin = 5

    data_folder = '../data/processed/TIMIT_spec/specs/'

    segment_length = np.array([0.2,
                 # 1, 1.2
                 ])
    segment_length = np.array([0.2,
     # 1, 1.2
     ])
    seg_fn = (segment_length/shift_hw)
    n_fft = int(sr*len_hw)
    hop_len = int(sr*shift_hw)

    print('Loading raw Mel data....')
    tms_ds = TIMIT_RawMel_Dataset(os.path.join(data_folder,'data.pkl'))

    info_path='../datasets/TIMIT_spec/specs/utter_info_timit.pkl'

    if not os.path.exists(info_path):
        print('Utterance infomation file not found. Generating from annotation...')
        utter_info = gen_utter_info(tms_ds)

        with open(info_path, 'wb') as outfile:
            pkl.dump(utter_info,outfile)

        print(f'Utterance info written to: {info_path}')
    else:
        with open(info_path,'rb') as infile:
            utter_info = pkl.load(infile)

        print('Utterance info loaded.')

    for sf in seg_fn:

        print(f'segment length: {sf*shift_hw} s')
        MSData=[]
        utter_idx=np.zeros((0,),dtype=int)
        st_list=np.zeros((0,),dtype=float)
        for i, sample in enumerate(tms_ds):

            if sample['path'].split('/')[-4]==mode \
            or mode=='ALL'\
            or  ('KALDI' in mode and \
                sample['path'].split('/')[-2].lower() in spk_lst):
                # print(utter_info[i])
                # print(sample['path'])
              # print(wav.squeeze().shape[-1]/freq.to(torch.float))
            #     print(sample['spectrogram'].shape)
                ms = torch.from_numpy(sample['spectrogram']).unsqueeze(0).unsqueeze(0)
            #     print(ms.size())
                
                # print(ms.shape)
                # mss,si = seg_data_si(ms,sf,int(sf/2),'pad')
                mss, si  = seg_data_anno(ms,sf,utter_info[i]['phones'],
                    shift_hw,margin,align_mode=align_mode)
                if mss.shape[2]!=80:
                    print(sample['path'],ms.shape,mss.shape)
                MSData.append(mss)
                # print(ms.shape,mss.shape,si)
                utter_idx=np.append(utter_idx,np.ones_like(si)*i)
                st_list=np.append(st_list,si*shift_hw)
                # print(utter_idx,st_list)
                # print(mss.shape,utter_idx.shape,st_list.shape)

        #     print(mss.shape)
        #     visualize(mss)
        #     print(mss.max(),mss.min())
        #     break
        MSData = torch.cat(MSData,0).permute(0,1,3,2)
        print(MSData.shape)

        data_file = os.path.join(data_folder,
         f'TIMITMel_anno_{int(sf)}_{mode}_{align_mode}.hdf5')

        with h5py.File(data_file, 'w') as hf:
            hf.create_dataset("melspectrogram",data=MSData.detach().numpy())
            hf.create_dataset("start_time",data=st_list)
            hf.create_dataset("utterance_idx",data=utter_idx)

            hf.attrs['margin'] = margin
            hf.attrs['segment_len'] = sf*shift_hw
            hf.attrs['shift_hw'] = shift_hw
            hf.attrs['align_mode'] = align_mode

            # print(hf.keys())
            for key in hf.keys():
                # print(key, hf[key][...])
                print(key, hf[key])

            print("Data is written to ",data_file)

            
if __name__ == "__main__":
    mel_to_patches(sys.argv)
