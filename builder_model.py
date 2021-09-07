from fbnas import NDS
from copy import deepcopy
from utils.tricks import *
from pynbs import nasbench2,model_spec
from pynbs.model_nb101_tri_bn import Network as NB101Network
from pynbs.nbmacro import Network as NBMacroNetwork
from pynbs.nlp_model import AWDRNNModelREG

import torch.nn as nn
import torch.nn.functional as F

#currently support NB101,NB201,NDS

class ModelBuilder():
    def __init__(self, search_space , config, device = 'cpu',last_channels = 64,init_channels = 16, input_size = 32, output_size = 8):
        self.search_space = search_space
        self.device = device
        self.init_channels = init_channels
        self.last_channels = last_channels
        self.input_size = input_size
        self.output_size = output_size
        
        if 'nasbench' not in self.search_space and 'nlp' not in self.search_space:
            self.NDS = NDS(self.search_space)
            assert self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in','ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in'], \
        "search space is not supported!"
        else:
            self.NDS = None
        self.config = config
    def get_model_nlp(self,modelargs):
        device = self.device
        custom_model = AWDRNNModelREG(modelargs.model, 
                               1, 
                               modelargs.emsize, 
                               modelargs.nhid, 
                               modelargs.nlayers, 
                               self.config['dimension'],
                               modelargs.dropout, 
                               modelargs.dropouth, 
                               modelargs.dropouti, 
                               modelargs.dropoute, 
                               modelargs.wdrop, 
                               modelargs.tied,
                               modelargs.recepie,
                               verbose=False)
        return custom_model.to(device)
    def get_model(self,arch):
        #NB101: description is matrix & ops
        #NB201: description is str
        #NDS: description is INDEX
        device = self.device
        config = self.config
        if self.NDS:
            original_config = deepcopy(self.NDS.get_network_config(arch))# save the original config since it is modified
            model = self.NDS.get_network(arch)
            model_config = self.NDS.get_network_config(arch)
            self.NDS.data[arch]['net'] = original_config
            
            if self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in']:
                concat_len = len(model_config['genotype']['reduce_concat'])
                final_channels = model.classifier.classifier.in_features
                try: in_channel = model.cells[model._layers//3].preprocess1.op[1].in_channels
                except: in_channel = model.cells[model._layers//3].preprocess1.conv_1.in_channels
                model.h2 = FullConvBN(in_channel,int(config['last_channel_l0']*self.last_channels))
                try: in_channel = model.cells[2*model._layers//3].preprocess1.op[1].in_channels
                except: in_channel = model.cells[2*model._layers//3].preprocess1.conv_1.in_channels
                model.h3 = FullConvBN(in_channel,int(config['last_channel_l1']*self.last_channels))
                model.h4 = FullConvBN(final_channels,int(config['last_channel_l2']*self.last_channels))  
            elif self.search_space in ['ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
                width = model_config['stem_w']
                ws = model_config['ws']
                depth = model_config['ds']
#                 model.stem = nn.Sequential(nn.Conv2d(3, width, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
#                                            nn.BatchNorm2d(width, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                                            nn.ReLU(inplace=True))
                model.h2 = FullConvBN(ws[1],int(config['last_channel_l0']*self.last_channels))
                model.h3 = FullConvBN(ws[2],int(config['last_channel_l1']*self.last_channels))
                model.h4 = FullConvBN(ws[3],int(config['last_channel_l2']*self.last_channels))            
                exec('model.s2.b%d.relu= DoNothing()'%(depth[-3]))
                exec('model.s3.b%d.relu= DoNothing()'%(depth[-2]))
                exec('model.s4.b%d.relu= DoNothing()'%(depth[-1]))
            
        else:
            if self.search_space == 'nasbench101':
                matrix, ops = arch
                spec = model_spec._ToModelSpec(matrix, ops)
                model = NB101Network(spec, last_channels = [int(config['last_channel_l0']*self.last_channels),\
                                                       int(config['last_channel_l1']*self.last_channels),\
                                                       int(config['last_channel_l2']*self.last_channels)],\
                                stem_out_channels = self.init_channels, num_stacks = 3, num_modules_per_stack = 3)
            elif self.search_space == 'nasbenchmacro':
                arch = [int(i) for i in list(arch)]
                model = NBMacroNetwork(arch,[int(config['last_channel_l0']*self.last_channels),\
                                                       int(config['last_channel_l1']*self.last_channels),\
                                                       int(config['last_channel_l2']*self.last_channels)], init_channels = self.init_channels)
            else:
                if isinstance(arch,list):
                    arch = generate_arch(arch)
                model = nasbench2.get_model_from_arch_str(arch, 10, self.init_channels)
                model.top0 = FullConvBN(self.init_channels,int(config['last_channel_l0']*self.last_channels))
                model.top1 = FullConvBN(self.init_channels*2,int(config['last_channel_l1']*self.last_channels))
                model.top2 = FullConvBN(self.init_channels*4,int(config['last_channel_l2']*self.last_channels))
                
                
        return model.to(device)
    
    def learn(self,model,x):
        output_size = self.output_size
        if self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in']:
            s0 = s1 = model.stem(x)
            for i, cell in enumerate(model.cells):
                if i == 1*model._layers//3:
                    x2 = model.h2(F.interpolate((s1),(output_size,output_size)))
                if i == 2*model._layers//3:
                    x3 = model.h3(F.interpolate((s1),(output_size,output_size)))
                s0, s1 = s1, cell(s0, s1, model.drop_path_prob)
            x4 = model.h4(s1)
            return [x2,x3,x4]
        elif self.search_space in ['ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
            x = model.stem(x)
            x1 = model.s1(x)
            x2 = model.s2((x1))
            x3 = model.s3(F.relu(x2))
            x4 = model.s4(F.relu(x3))
            x2 = model.h2(F.interpolate((x2),(output_size,output_size)))
            x3 = model.h3(F.interpolate((x3),(output_size,output_size)))
            x4 = model.h4(x4)
#             print(x3.shape,x4.shape)
            return [x2,x3,x4]
        elif self.search_space == 'nasbench201':
            x = model.stem(x)        
            x = model.stack_cell1(x)
            x0 = model.top0(F.interpolate((x),(output_size,output_size)))
            x = model.reduction1(x)
            x = model.stack_cell2(x)
            x1 = model.top1(F.interpolate((x),(output_size,output_size)))
            x = model.reduction2(x)
            x = model.stack_cell3(x)
            x2 = model.top2(x)
            return [x0,x1,x2]
        else:
            return model(x)
        