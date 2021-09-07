# from models import get_cell_based_tiny_net, get_search_spaces
# from nas_201_api import NASBench201API as API
# from nasbench import api as nasbench101api
# from nas_101_api.model import Network
# from nas_101_api.model_spec import ModelSpec
import itertools
import random
import numpy as np
# from models.cell_searchs.genotypes import Structure
from copy import deepcopy
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import json
import torch
import os

class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x), x
                

def return_feature_layer(network, prefix=''):
    #for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')
             

class NDS:
    def __init__(self, searchspace, path = './data/nds_data/'):
        self.searchspace = searchspace
        data = json.load(open(os.path.join(path,'%s.json'%searchspace), 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        #print(config)
        if 'genotype' in config:
#             print('geno')
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(config['width'], 1, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.
            #print(config)
            #print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                if len(config['bot_muls']) > 0:
                    if config['bot_muls'][0] == 0:
                        config['bot_muls'][0] = 1
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            if 'ss' in config and config['ss'][0] == 8:
#                 print(config['ss'][0])
                config['ss'][0] = 1
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_cifar'#'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            #"res_stem_cifar": ResStemCifar,
            #"res_stem_in": ResStemIN,
            #"simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
#         return_feature_layer(network)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def get_final_accuracy(self, uid, acc_type, trainval):
        return 100.-self.data[uid]['test_ep_top1'][-1]


def get_search_space(args):
    if args.nasspace == 'nds_resnet':
        return NDS('ResNet')
    elif args.nasspace == 'nds_amoeba':
        return NDS('Amoeba')
    elif args.nasspace == 'nds_amoeba_in':
        return NDS('Amoeba_in')
    elif args.nasspace == 'nds_darts_in':
        return NDS('DARTS_in')
    elif args.nasspace == 'nds_darts':
        return NDS('DARTS')
    elif args.nasspace == 'nds_darts_fix-w-d':
        return NDS('DARTS_fix-w-d')
    elif args.nasspace == 'nds_darts_lr-wd':
        return NDS('DARTS_lr-wd')
    elif args.nasspace == 'nds_enas':
        return NDS('ENAS')
    elif args.nasspace == 'nds_enas_in':
        return NDS('ENAS_in')
    elif args.nasspace == 'nds_enas_fix-w-d':
        return NDS('ENAS_fix-w-d')
    elif args.nasspace == 'nds_pnas':
        return NDS('PNAS')
    elif args.nasspace == 'nds_pnas_fix-w-d':
        return NDS('PNAS_fix-w-d')
    elif args.nasspace == 'nds_pnas_in':
        return NDS('PNAS_in')
    elif args.nasspace == 'nds_nasnet':
        return NDS('NASNet')
    elif args.nasspace == 'nds_nasnet_in':
        return NDS('NASNet_in')
    elif args.nasspace == 'nds_resnext-a':
        return NDS('ResNeXt-A')
    elif args.nasspace == 'nds_resnext-a_in':
        return NDS('ResNeXt-A_in')
    elif args.nasspace == 'nds_resnext-b':
        return NDS('ResNeXt-B')
    elif args.nasspace == 'nds_resnext-b_in':
        return NDS('ResNeXt-B_in')
    elif args.nasspace == 'nds_vanilla':
        return NDS('Vanilla')
    elif args.nasspace == 'nds_vanilla_lr-wd':
        return NDS('Vanilla_lr-wd')
    elif args.nasspace == 'nds_vanilla_lr-wd_in':
        return NDS('Vanilla_lr-wd_in')
