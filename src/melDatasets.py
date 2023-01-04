from utils import *
import h5py
import torch
import os, sys
import numpy as np
import pickle as pkl
import sys

import random

# definition ofdataset classes


class TIMIT_RawMel_Dataset(torch.utils.data.Dataset):

    '''
        return the mel spectrogram and path of an utterance 
    '''

    def __init__(self, data_path):
        super(TIMIT_RawMel_Dataset).__init__()
        with open(data_path, 'rb') as infile:
            print('loading data from', data_path)
            data = pkl.load(infile)
            self.spectrogram = data['Melspec']
            self.path = data['path']
        
    def __len__(self):
        return len(self.spectrogram)
    
    def __getitem__(self, idx):
        sample = {'spectrogram':self.spectrogram[idx],
                  'path':self.path[idx]}
        return sample

class TIMIT_RawMel_Dataset_Triplet(TIMIT_RawMel_Dataset):

    '''
       the return value sample_list is a tuple consists of:
          the mel spectrograms of corresponding information of 3 random utterances where:
           sample_list[0] and sample_list[1] are the same sentence spoken by different speakers,
           and sample_list[0] and sample_list[2] are different sentences spoken by the same speaker
    '''
    def __init__(self, data_path, info_path ):
        super().__init__(data_path)
                    
        with open(info_path,'rb') as infile:
            self.utter_info = pkl.load(infile)
        
        self.sentence_idx = {}
        self.sentence_content = {}
        self.ID_idx = {}
        
        self.valid_idx=list(range(len(self.spectrogram)))
        
        for i in range(len(self.spectrogram)):
            uinfo= self.utter_info[i]
            
            if uinfo['sentence'] not in self.sentence_idx:
                self.sentence_idx[uinfo['sentence']]=[i]

                with open(self.path[i].replace('.WAV','.TXT'), 'r') as f:
                    _,_,x = f.readlines()[0].split(' ',2)    
                    self.sentence_content[uinfo['sentence']] = x
            else:
                self.sentence_idx[uinfo['sentence']].append(i)

            spkID = uinfo['ID'] 
            if spkID not in self.ID_idx:
                self.ID_idx[spkID]=[i]
            else:
                self.ID_idx[spkID].append(i)
        for k,idx in self.sentence_idx.items():
            if len(idx)==1:
#                     print(k,idx)
                self.valid_idx.remove(idx[0])
        
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        vidx = self.valid_idx[idx]
        sentence = self.utter_info[vidx]['sentence']
        
        spk = self.utter_info[vidx]['ID']
        spk_occur = len(self.ID_idx[spk])
        v_in_spk = self.ID_idx[spk].index(vidx)
        s_in_spk = v_in_spk - np.random.randint(spk_occur-1) - 1
        sidx = self.ID_idx[spk][s_in_spk]
                            
        sentence_occur = len(self.sentence_idx[sentence])
        v_in_sentence = self.sentence_idx[sentence].index(vidx)
        r_in_sentence = v_in_sentence - np.random.randint(sentence_occur-1) - 1
        ridx = self.sentence_idx[sentence][r_in_sentence]

        sample_list = []
        for idx in [vidx,ridx,sidx]:
            sample_list.append({'spectrogram':self.spectrogram[idx],
                                'path':self.path[idx],
                                'ID':self.utter_info[idx]['ID'],
                                'sentence':self.sentence_content[self.utter_info[idx]['sentence']],
                                'sentence_ID':self.utter_info[idx]['sentence']})
        return tuple(sample_list)

class YNDataset_Mel(torch.utils.data.Dataset):
    
    '''
        return a mel spectrogram segment
    '''
    def __init__(self, data_path, device='cuda'):
        super(YNDataset_Mel).__init__()
        self.device=device
        with h5py.File(data_path, 'r') as hf:
          # hf.visit(print_name)
            print('loading data from', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...])#.to(device)
        
    def __len__(self):
        return self.spectrogram.shape[0]
    
    def __getitem__(self, idx):
        sample = self.spectrogram[idx].to(self.device) 
        return sample

class YNDataset_MelPair(torch.utils.data.Dataset):
    '''
        return a pair of mel spectrogram segments from the same uterrance 
          with a time difference of dt 
    '''
    def __init__(self, data_path, seg_length=20, device='cuda'):
        super(YNDataset_Mel).__init__()
        self.sl=seg_length;
        self.device=device
        with h5py.File(data_path, 'r') as hf:
          # hf.visit(print_name)
            print('loading data from', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...])#.to(device)
        self.valid_range = self.spectrogram.shape[2]-seg_length
    def __len__(self):
        return self.spectrogram.shape[0]
    
    def __getitem__(self, idx):
        sample = self.spectrogram[idx]
        
        idx = torch.randperm(self.valid_range)[:2]
        
        return {'s1': sample[:,idx[0]:idx[0]+self.sl].to(self.device),
               's2': sample[:,idx[1]:idx[1]+self.sl].to(self.device),
               'dt': (idx[1]-idx[0]).to(self.device).view(1)/100.0}

class YNDataset_Mel_annoPair(torch.utils.data.Dataset):
    '''
        return a pair of mel spectrogram segments 
          with the same phone sequence and offsets
    '''
    def __init__(self, data_path, info_path, device='cuda'):
        super(YNDataset_Mel).__init__()
        self.device=device
        with h5py.File(data_path, 'r') as hf:
            print('Loading data from: ', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...])#.to(device)
            self.utter_idx = hf['utterance_idx'][...]
            self.st = hf['start_time'][...]

            self.margin = hf.attrs['margin']
            self.shift_hw = hf.attrs['shift_hw']
            self.seg_len =int(hf.attrs['segment_len'][...]/self.shift_hw)
        with open(info_path,'rb') as infile:
            self.utter_info = pkl.load(infile)
        
        self.phone_comb_idx = {}
        self.phone_anno = []

        for i in range(self.spectrogram.shape[0]):
            uinfo= self.utter_info[self.utter_idx[i]]
    
            phone_indices = np.argwhere((uinfo['phones']['start']<self.st[i]+self.seg_len+
                                         self.margin*self.shift_hw)*\
                         (uinfo['phones']['end']>self.st[i]+self.margin*self.shift_hw))[:,0]
#             phone_indices = np.argwhere((uinfo['phones']['end']<self.st[i]+self.seg_len+
#                                          self.margin*shift_hw*2)*\
#                          (uinfo['phones']['start']>self.st[i]))[:,0]
        
            phones = tuple(uinfo['phones']['phone'][phone_indices[:3]])
            self.phone_anno.append(phones)
            if phones not in self.phone_comb_idx:
                self.phone_comb_idx[phones]=[i]
            else:
                self.phone_comb_idx[phones].append(i)

        
        
    def __len__(self):
        return self.spectrogram.shape[0]
    
    def __getitem__(self, idx):
        phones = self.phone_anno[idx]
        phone_occur = len(self.phone_comb_idx[phones])
        ri = np.random.randint(phone_occur)
        
        if self.phone_comb_idx[phones][ri]==idx:
            ri-=1
        ridx = self.phone_comb_idx[phones][ri]
        
        offset =  np.random.randint(int(self.margin*2))
        
        return {'s1': self.spectrogram[idx,:,offset:offset+self.seg_len,:].to(self.device),
                's2': self.spectrogram[ridx,:,offset:offset+self.seg_len,:].to(self.device),
                'idx':(idx,ridx),
                'phones':phones,
                'offset':offset,
               }


class YNDataset_Mel_anno_triplet(torch.utils.data.Dataset):

    '''
       return a triplet of mel spectrogram segments where:
           s_sp and s_xp are the same phone sequence spoken by different speakers,
           s_sp and s_sx  are different phone sequence spoken by different speakers
    '''
    def __init__(self, data_path, info_path, sample_margin=None, phone_num=3,
                 no_silence=True, no_duplicate=True, device='cuda'):
        super(YNDataset_Mel).__init__()
        self.device=device
        with h5py.File(data_path, 'r') as hf:
            print('Loading data from: ', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...])#.to(device)
            self.utter_idx = hf['utterance_idx'][...]
            self.st = hf['start_time'][...]

            self.margin = hf.attrs['margin']
            self.shift_hw = hf.attrs['shift_hw']
            self.seg_len =int(hf.attrs['segment_len'][...]/self.shift_hw)
            
        if sample_margin is None:
            self.sample_margin=self.margin
        else:
            self.sample_margin = sample_margin
                
        self.no_duplicate = no_duplicate
        self.phone_num=phone_num
        with open(info_path,'rb') as infile:
            self.utter_info = pkl.load(infile)
        
        self.phone_comb_idx = {}
        self.phone_anno = []
        self.ID_idx = {}
        
        self.valid_idx=[]
        seg_dur = self.seg_len*self.shift_hw
        for i in range(self.spectrogram.shape[0]):
#         for i in range(50):
            uinfo= self.utter_info[self.utter_idx[i]]
#             print(uinfo)
#             print(self.st[i]+seg_dur+\
#                                          self.margin*self.shift_hw)
#             print(self.st[i]+self.margin*self.shift_hw)
            phone_indices = np.argwhere((uinfo['phones']['start']<self.st[i]+seg_dur+\
                                         self.margin*self.shift_hw)&\
                         (uinfo['phones']['end']>self.st[i]+self.margin*self.shift_hw))[:,0]
#             phone_indices = np.argwhere((uinfo['phones']['end']<self.st[i]+self.seg_len+
#                                          self.margin*shift_hw*2)*\
#                          (uinfo['phones']['start']>self.st[i]))[:,0]
            if self.phone_num:
                phones = tuple(uinfo['phones']['phone'][phone_indices[:self.phone_num]])
            else:
                phones = tuple(uinfo['phones']['phone'][phone_indices[:]])
                
#             print(phone_indices,phones)
            
            self.phone_anno.append(phones)
            if phones not in self.phone_comb_idx:
                self.phone_comb_idx[phones]=[i]
            else:
                self.phone_comb_idx[phones].append(i)

            if phones != ('h#',):
                spkID = uinfo['ID'] 
                if spkID not in self.ID_idx:
                    self.ID_idx[spkID]=[i]
                else:
                    self.ID_idx[spkID].append(i)
#             else:
#                 print(i)

            if not (no_silence and phones==('h#',)):
                self.valid_idx.append(i)
        if no_duplicate:
            for k,idx in self.phone_comb_idx.items():
                if len(idx)==1:
#                     print(k,idx)
                    self.valid_idx.remove(idx[0])
        
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        vidx = self.valid_idx[idx]
        phones = self.phone_anno[vidx]
        
        spk = self.utter_info[self.utter_idx[vidx]]['ID']
        spk_occur = len(self.ID_idx[spk])
        v_in_spk = self.ID_idx[spk].index(vidx)
        s_in_spk = v_in_spk - np.random.randint(spk_occur-1) - 1
        sidx = self.ID_idx[spk][s_in_spk]
                
        if phones == ('h#',):
            phones = self.phone_anno[sidx]
            sidx, vidx = vidx, sidx
            
        phone_occur = len(self.phone_comb_idx[phones])
        v_in_phone = self.phone_comb_idx[phones].index(vidx)
        r_in_phone = v_in_phone - np.random.randint(phone_occur-1) - 1
        ridx = self.phone_comb_idx[phones][r_in_phone]
#         ri = np.random.randint(phone_occur)
        
#         if self.phone_comb_idx[phones][ri]==idx:
#             ri-=1
#         ridx = self.phone_comb_idx[phones][ri]
        
        offset =  np.random.randint(int(self.sample_margin*2)+1)+self.margin-self.sample_margin
        offset_s =  np.random.randint(int(self.sample_margin*2)+1)
        
        return {'s_sp': self.spectrogram[vidx,:,offset:offset+self.seg_len,:].to(self.device),
                's_xp': self.spectrogram[ridx,:,offset:offset+self.seg_len,:].to(self.device),
                's_sx':self.spectrogram[sidx,:,offset_s:offset_s+self.seg_len,:].to(self.device),
                'idx':(vidx,ridx,sidx),
                'phones':(self.phone_anno[vidx],self.phone_anno[ridx],self.phone_anno[sidx]),
                'ID':(self.utter_info[self.utter_idx[vidx]]['ID'],
                     self.utter_info[self.utter_idx[ridx]]['ID'],
                      self.utter_info[self.utter_idx[sidx]]['ID']),
                'offset':(offset,offset_s),
               }