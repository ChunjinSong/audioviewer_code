import argparse
import os

import yaml

from training_mappingA2V import Train

# dir_current = os.path.dirname(__file__)
# path = os.path.join(dir_current, 'config/train_mapping.yaml')

parses = argparse.ArgumentParser()
parses.add_argument('--config', type=str, default='./config/training_mapping.yaml', help='path to the configs file.')
opts = parses.parse_args()

def get_config(config):
    with open(config,'r') as stream:
        return yaml.load(stream)


def train():
    args = get_config(opts.config)
    if args is None:
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args['trainer']['GPU_ID']))
    model = Train(args)
    model.train()
    pass

if __name__ == '__main__':
    train()