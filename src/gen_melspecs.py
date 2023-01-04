#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
from hparam import Hparam
import pickle as pkl
import sys
from tqdm import tqdm

def save_spectrogram(hp):
    """ 
        Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    # downloaded dataset path
    audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))                                        
    print(audio_path)
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.out_path, exist_ok=True)   # make folder to save train file

    total_speaker_num = len(audio_path)
    # train_speaker_num = 1
    print("total speaker number : %d"%total_speaker_num)

    utterances_spec = []
    path_list = []

    for i, folder in enumerate(audio_path):
        print("%dth speaker processing(%s)..."%(i+1, folder))
        for utter_name in tqdm(os.listdir(folder)):
            if utter_name[-4:].lower() == '.WAV'.lower():
                # print(utter_name)
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                '''
                  save: melspectrums, path
                '''
                # S = librosa.core.stft(y=utter,
                #                       n_fft=hp.data.nfft,
                #                       win_length=int(hp.data.window * sr),
                #                       hop_length=int(hp.data.hop * sr))
                # S = np.abs(S)
                # mel_basis = librosa.filters.mel(sr=hp.data.sr, 
                #                                 n_fft=hp.data.nfft,
                #                                 n_mels=hp.data.nmels,
                #                                 fmin=20,
                #                                 fmax=8000)
                # # S = np.log10(np.dot(mel_basis, S))           # log mel spectrogram of utterances
                # # S = librosa.core.power_to_dB(np.dot(mel_basis, S)) 
                # S = np.dot(mel_basis, S)          # log mel spectrogram of utterances
                # S = np.log10(S)           # log mel spectrogram of utterances
                # S[S<hp.data.flooring_dB] = hp.data.flooring_dB
                # S = S*20          # log mel spectrogram of utterances

                S = librosa.feature.melspectrogram(y=utter,sr=sr,
                                n_fft=hp.data.nfft,
                                win_length=int(hp.data.window * sr),
                                hop_length=int(hp.data.hop * sr),
                                n_mels=hp.data.nmels,
                                fmin=20,
                                fmax=8000)
                S = librosa.core.power_to_db(S,ref=1.0, top_db=80) 

                utterances_spec.append(S)
                path_list.append(utter_path)
        # break

    data = {'Melspec':utterances_spec,
                'path':path_list}
    # print(data_train)
    with open(os.path.join(hp.data.out_path,'data.pkl'), 'wb') as outfile:
        pkl.dump(data, outfile)


if __name__ == "__main__":

    if len(sys.argv) <=1:
        hp=Hparam()
    else:
        hp=Hparam(sys.argv[1])
    save_spectrogram(hp)
