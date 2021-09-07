from fbnas import NDS
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from model_wrapper import *
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import numpy as np

class ModelBuilder():
    def __init__(self, args):
        self.search_space = args.search_space
        self.device = args.device
        self.output_size = args.output_size
        self.config = args.config
        self.nds_path = args.nds_path
        
        
        
        
        if 'nlp' in self.search_space:
            self.dimension = self.config['dimension']
        else:
            self.last_channels = np.asarray([self.config['last_channel_l0']*args.last_channels, self.config['last_channel_l1']*args.last_channels, self.config['last_channel_l2']*args.last_channels]).astype(int)
        
        if args.init_channels:
            self.init_channels = args.init_channels
        assert self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in','ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in', 'nasbench101', 'nasbench201','nasbenchnlp'], \
        "search space is not supported!"
        
        try: self.NDS = NDS(self.search_space, self.nds_path)
        except: self.NDS = None
        
    def get_model(self, arch_config):
        if self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in']:
            if '_in' in self.search_space:
                imagenet = True
            else:
                imagenet = False
            model_config = self.NDS.get_network_config(arch_config)
            gen = model_config['genotype']
            
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            width = model_config['width']
            depth = model_config['depth']
            model = DARTSWrapper(genotype, width, depth, self.last_channels, imagenet = imagenet, output_size = self.output_size)
        
        elif self.search_space in ['ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
            model_config = self.NDS.get_network_config(arch_config)
            model = ResNetSeriesWrapper(model_config, self.last_channels, output_size = self.output_size)
        
        elif self.search_space == 'nasbench101':
            model = NB101Wrapper(arch_config, self.init_channels, self.last_channels, self.output_size)
        elif self.search_space == 'nasbench201':
            model = NB201Wrapper(arch_config, self.init_channels, self.last_channels, self.output_size)
        elif self.search_space == 'nasbenchnlp':
            model = NBNLPWrapper(arch_config[1], self.dimension)
        
        """
        to do list: more search space maybe?
        """
        return model.to(self.device)
            
        

        
