from melDatasets import YNDataset_Mel, YNDataset_MelPair, YNDataset_Mel_anno_triplet
import torch, os
from torchvision import datasets, transforms

def read_train_audio_single(data_file, batch_size_single):
    timit_train_single = YNDataset_Mel(data_file)
    timit_val_single = YNDataset_Mel(data_file.replace('TRAIN', 'KALDI_DEV'))
    timit_test_single = YNDataset_Mel(data_file.replace('TRAIN', 'KALDI_TEST'))

    # batch_size_single = 144
    # train, val, test
    train_loader_timit_single = torch.utils.data.DataLoader(timit_train_single,
                                                            batch_size=batch_size_single,
                                                            shuffle=True,
                                                            drop_last=True)
    val_loader_timit_single = torch.utils.data.DataLoader(timit_val_single,
                                                          batch_size=batch_size_single,
                                                          shuffle=True)
    # test_loader_timit_single = torch.utils.data.DataLoader(timit_test_single,
    #                                                        batch_size=1,
    #                                                        shuffle=True)

    print(f"TIMIT dataset size: train:{len(timit_train_single)}," + \
          f' val:{len(timit_val_single)}')

    return train_loader_timit_single, val_loader_timit_single#, test_loader_timit_single


def read_train_audio_pair(data_file, batch_size_pair):
    timit_train_pair = YNDataset_MelPair(data_file)
    timit_val_pair = YNDataset_MelPair(data_file.replace('TRAIN', 'KALDI_DEV'))
    timit_test_pair = YNDataset_MelPair(data_file.replace('TRAIN', 'KALDI_TEST'))

    train_loader_timit_pair = torch.utils.data.DataLoader(timit_train_pair,
                                                          batch_size=batch_size_pair,
                                                          shuffle=True,
                                                          drop_last=True)
    val_loader_timit_pair = torch.utils.data.DataLoader(timit_val_pair,
                                                        batch_size=batch_size_pair,
                                                        shuffle=True)
    # test_loader_timit_pair = torch.utils.data.DataLoader(timit_test_pair,
    #                                                      batch_size=1,
    #                                                      shuffle=True)

    print(f"TIMIT dataset size: train:{len(timit_train_pair)}, "
          f"val:{len(timit_val_pair)}")

    return train_loader_timit_pair, val_loader_timit_pair#, test_loader_timit_pair


def read_train_audio_triplet(data_file, uinfo_file, batch_size_triplet):

    train_set_triplet = YNDataset_Mel_anno_triplet(data_file, uinfo_file, 3, None)
    val_set_triplet = YNDataset_Mel_anno_triplet(data_file.replace('TRAIN', 'KALDI_DEV'), uinfo_file, 3, None)
    test_set_triplet = YNDataset_Mel_anno_triplet(data_file.replace('TRAIN', 'KALDI_TEST'), uinfo_file, 3, None)

    # train, val, test
    train_loader_timit_triplet = torch.utils.data.DataLoader(train_set_triplet,
                                                             batch_size=batch_size_triplet,
                                                             shuffle=True)

    val_loader_timit_triplet = torch.utils.data.DataLoader(val_set_triplet,
                                                           batch_size=batch_size_triplet,
                                                           shuffle=True)

    # test_loader_timit_triplet = torch.utils.data.DataLoader(test_set_triplet,
    #                                                         batch_size=1,
    #                                                         shuffle=True)

    print(f"Whole dataset size: train:{len(train_set_triplet)}, "
          f"val:{len(val_set_triplet)}")

    return train_loader_timit_triplet, val_loader_timit_triplet#, test_loader_timit_triplet

def get_max_min_triplet(args):
    path_audio_timit_spec_triplet_train = args['data']['path_audio_timit_spec_triplet_train']
    path_audio_timit_spec_triplet_unifo = args['data']['path_audio_timit_spec_triplet_unifo']

    train_set = YNDataset_Mel_anno_triplet(path_audio_timit_spec_triplet_train, path_audio_timit_spec_triplet_unifo, 3, None)
    gmin = train_set.spectrogram.min()
    gmax = train_set.spectrogram.max()
    return gmin, gmax

def get_max_min_single(args):
    path_audio_timit_spec_single_train = os.path.join(args['data']['dir_data'],
                                                      args['data']['path_audio_timit_spec_single_train'])
    # path_audio_timit_spec_single_train = args['data']['path_audio_timit_spec_single_train']
    train_set = YNDataset_Mel(path_audio_timit_spec_single_train)
    gmin = train_set.spectrogram.min()
    gmax = train_set.spectrogram.max()
    return gmin, gmax

def get_audio_loader_dict(args):
    batch_size_single = get_path(args['data']['batch_size_single'])
    batch_size_pair = args['data']['batch_size_pair']
    batch_size_triplet = args['data']['batch_size_triplet']

    path_audio_timit_spec_single_train = os.path.join(args['data']['dir_data'],
                                                      args['data']['path_audio_timit_spec_single_train'])

    path_audio_timit_spec_pair_train = os.path.join(args['data']['dir_data'],
                                                      args['data']['path_audio_timit_spec_pair_train'])

    path_audio_timit_spec_triplet_train = os.path.join(args['data']['dir_data'],
                                                      args['data']['path_audio_timit_spec_triplet_train'])

    path_audio_timit_spec_triplet_unifo = os.path.join(args['data']['dir_data'],
                                                      args['data']['path_audio_timit_spec_triplet_unifo'])

    train_loader_timit_single, val_loader_timit_single = read_train_audio_single(path_audio_timit_spec_single_train, batch_size_single)
    train_loader_timit_pair, val_loader_timit_pair = read_train_audio_pair(path_audio_timit_spec_pair_train, batch_size_pair)
    train_loader_timit_triplet, val_loader_timit_triplet = read_train_audio_triplet(path_audio_timit_spec_triplet_train, path_audio_timit_spec_triplet_unifo, batch_size_triplet)

    audio_loader_dict = {'single': {'train': train_loader_timit_single,
                                         'val': val_loader_timit_single},
                              'pair': {'train': train_loader_timit_pair,
                                       'val': val_loader_timit_pair},
                              'triplet': {'train': train_loader_timit_triplet,
                                          'val': val_loader_timit_triplet},
                              }

    return audio_loader_dict

def get_audio_loader_triplet(args):
    batch_size_triplet = args['data']['batch_size_triplet']

    path_audio_timit_spec_triplet_train = args['data']['path_audio_timit_spec_triplet_train']
    path_audio_timit_spec_triplet_unifo = args['data']['path_audio_timit_spec_triplet_unifo']

    train_loader_timit_triplet, val_loader_timit_triplet = read_train_audio_triplet(path_audio_timit_spec_triplet_train, path_audio_timit_spec_triplet_unifo, batch_size_triplet)

    audio_loader_triplet= {'train': train_loader_timit_triplet,
                                          'val': val_loader_timit_triplet}


    return audio_loader_triplet

def get_audio_loader_pair(args):
    batch_size_pair = args['data']['batch_size_pair']
    path_audio_timit_spec_pair_train = args['data']['path_audio_timit_spec_pair_train']

    train_loader_timit_pair, val_loader_timit_pair = read_train_audio_pair(path_audio_timit_spec_pair_train, batch_size_pair)

    audio_loader_pair = {'train': train_loader_timit_pair,
                                       'val': val_loader_timit_pair}

    return audio_loader_pair


def get_audio_loader_single(args):
    batch_size_single = args['data']['batch_size_single']
    path_audio_timit_spec_single_train = args['data']['path_audio_timit_spec_single_train']
    train_loader_timit_single, val_loader_timit_single = read_train_audio_single(path_audio_timit_spec_single_train, batch_size_single)

    audio_loader_single = {'train': train_loader_timit_single,
                                         'val': val_loader_timit_single}

    return audio_loader_single


def get_video_loader_dict(args):

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(args['data']['img_size_video']),
                                    transforms.ToTensor(),
                                    SetRange])

    raw_data_folder = args['data']['dir_video_raw']

    celeba_train = datasets.CelebA(root=raw_data_folder,
                                   split="train",
                                   transform=transform,
                                   download=True)
    celeba_val = datasets.CelebA(root=raw_data_folder,
                                 split="test",
                                 transform=transform,
                                 download=True)

    train_loader_celeba = torch.utils.data.DataLoader(celeba_train,
                                                      batch_size=args['data']['batch_size_video'],
                                                      shuffle=True,
                                                      drop_last=True)
    val_loader_celeba = torch.utils.data.DataLoader(celeba_val,
                                                    batch_size=args['data']['batch_size_video'],
                                                    shuffle=True,
                                                    drop_last=True)

    visual_loader_dict = {'face': {'train': train_loader_celeba,
                                   'val': val_loader_celeba}
                          }

    return visual_loader_dict, celeba_train, celeba_val

def get_path(path):
    return path



