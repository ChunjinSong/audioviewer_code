from utils import *
import h5py
import torch
import os, sys
import numpy as np
import pickle as pkl
import sys


class TIMIT_RawMel_Dataset(torch.utils.data.Dataset):
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
        sample = {'spectrogram': self.spectrogram[idx],
                  'path': self.path[idx]}
        return sample


class TIMIT_RawMel_Dataset_Triplet(TIMIT_RawMel_Dataset):
    def __init__(self, data_path, info_path):
        super().__init__(data_path)

        with open(info_path, 'rb') as infile:
            self.utter_info = pkl.load(infile)

        self.sentence_idx = {}
        self.sentence_content = {}
        self.ID_idx = {}

        self.valid_idx = list(range(len(self.spectrogram)))

        for i in range(len(self.spectrogram)):
            uinfo = self.utter_info[i]

            if uinfo['sentence'] not in self.sentence_idx:
                self.sentence_idx[uinfo['sentence']] = [i]

                with open(self.path[i].replace('.WAV', '.TXT'), 'r') as f:
                    _, _, x = f.readlines()[0].split(' ', 2)
                    self.sentence_content[uinfo['sentence']] = x
            else:
                self.sentence_idx[uinfo['sentence']].append(i)

            spkID = uinfo['ID']
            if spkID not in self.ID_idx:
                self.ID_idx[spkID] = [i]
            else:
                self.ID_idx[spkID].append(i)
        for k, idx in self.sentence_idx.items():
            if len(idx) == 1:
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
        s_in_spk = v_in_spk - np.random.randint(spk_occur - 1) - 1
        sidx = self.ID_idx[spk][s_in_spk]

        sentence_occur = len(self.sentence_idx[sentence])
        v_in_sentence = self.sentence_idx[sentence].index(vidx)
        r_in_sentence = v_in_sentence - np.random.randint(sentence_occur - 1) - 1
        ridx = self.sentence_idx[sentence][r_in_sentence]

        sample_list = []
        for idx in [vidx, ridx, sidx]:
            sample_list.append({'spectrogram': self.spectrogram[idx],
                                'path': self.path[idx],
                                'ID': self.utter_info[idx]['ID'],
                                'sentence': self.sentence_content[self.utter_info[idx]['sentence']],
                                'sentence_ID': self.utter_info[idx]['sentence']})
        return tuple(sample_list)


class YNDataset_Mel(torch.utils.data.Dataset):
    def __init__(self, data_path, device='cuda'):
        super(YNDataset_Mel).__init__()
        with h5py.File(data_path, 'r') as hf:
            # hf.visit(print_name)
            print('loading data from', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...]).to(device)

    def __len__(self):
        return self.spectrogram.shape[0]

    def __getitem__(self, idx):
        sample = self.spectrogram[idx]
        return sample


class YNDataset_MelPair(torch.utils.data.Dataset):
    def __init__(self, data_path, seg_length=20, device='cuda'):
        super(YNDataset_Mel).__init__()
        self.sl = seg_length;
        self.device = device
        with h5py.File(data_path, 'r') as hf:
            # hf.visit(print_name)
            print('loading data from', data_path)
            self.spectrogram = torch.from_numpy(hf['melspectrogram'][...]).to(device)
        self.valid_range = self.spectrogram.shape[2] - seg_length

    def __len__(self):
        return self.spectrogram.shape[0]

    def __getitem__(self, idx):
        sample = self.spectrogram[idx]

        idx = torch.randperm(self.valid_range)[:2]

        return {'s1': sample[:, idx[0]:idx[0] + self.sl],
                's2': sample[:, idx[1]:idx[1] + self.sl],
                'dt': (idx[1] - idx[0]).to(self.device).view(1) / 100.0}


def mel_to_patches(argv):
    if len(argv) <= 1:
        mode = 'ALL'
    elif argv[1] == 'test':
        mode = 'TEST'
    elif argv[1] == 'train':
        mode = 'TRAIN'
    elif argv[1] == 'dev':
        mode = 'KALDI_DEV'
        with open("./configs/dev_spk.list") as f:
            spk_lst = f.read().split()
    elif argv[1] == 'ktest':
        mode = 'KALDI_TEST'
        with open("./configs/test_spk.list") as f:
            spk_lst = f.read().split()
    else:
        mode = 'ALL'

    min_dB = -20
    len_hw = 25 / 1000.0
    shift_hw = 10 / 1000.0
    n_FBank = 80
    sr = 16000
    z_dim = 128
    data_folder = '../datasets/TIMIT_spec/specs/'

    # segment_length = np.array([0.2, 1, 1.2])
    segment_length = np.array([0.6, 1.8])
    seg_fn = (segment_length / shift_hw)
    n_fft = int(sr * len_hw)
    hop_len = int(sr * shift_hw)

    tms_ds = TIMIT_RawMel_Dataset(os.path.join(data_folder, 'data.pkl'))

    for sf in seg_fn:

        MSData = []

        for sample in tms_ds:

            if sample['path'].split('/')[4] == mode \
                    or mode == 'ALL' \
                    or ('KALDI' in mode and sample['path'].split('/')[-2].lower() in spk_lst):
                # print(wav.squeeze().shape[-1]/freq.to(torch.float))
                #     print(sample['spectrogram'].shape)
                ms = torch.from_numpy(sample['spectrogram']).unsqueeze(0).unsqueeze(0)
                #     print(ms.size())

                # print(ms.shape)
                mss = seg_data(ms, sf, int(sf / 2))
                if mss.shape[2] != 80:
                    print(sample['path'], ms.shape, mss.shape)
                MSData.append(mss)
            # break

        #     print(mss.shape)
        #     visualize(mss)
        #     print(mss.max(),mss.min())
        #     break
        MSData = torch.cat(MSData, 0).permute(0, 1, 3, 2)
        print(MSData.shape)

        data_file = os.path.join(data_folder, f'TIMITMel_{int(sf)}_{mode}.hdf5')

        def print_name(name):
            print(name)

        # print(data_file)
        with h5py.File(data_file, 'w') as hf:
            hf.create_dataset("melspectrogram", data=MSData.detach().numpy())
            hf.visit(print_name)
            print("Data is written to ", data_file)


if __name__ == "__main__":
    mel_to_patches(sys.argv)
