from copy import deepcopy
from utils.tricks import *
from pynbs.nasbench101.model import Network as NB101Network
from pynbs.nasbench101 import model_spec
from pynbs.nasbench201.nasbench2 import NAS201Model as NAS201Network
from pynbs.nasbench201.nasbench2 import get_model_from_arch_str
from pynbs.nasbenchmacro.nbmacro import Network as NBMacroNetwork
from pynbs.nasbenchnlp.models import AWDRNNModel as NBNLPNetwork

from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.nas.operations import ReLUConvBN
from pycls.models.anynet import AnyNet

import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace


class NB101Wrapper(nn.Module):
    """
    wrap the NB101 model for regression
    e.g. arch:
    [[[0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 1, 0],
  [0, 0, 0, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0]],
 ['input',
  'conv1x1-bn-relu',
  'conv1x1-bn-relu',
  'conv3x3-bn-relu',
  'conv1x1-bn-relu',
  'maxpool3x3',
  'output']]
    """
    def __init__(self, arch, init_channels,last_channels,output_size = 8, num_stacks = 3,num_modules_per_stack = 3,num_labels = 10):
        super(NB101Wrapper, self).__init__()
        self.init_channels = init_channels
        self.last_channels = last_channels
        self.output_size = output_size
        self.num_stacks = num_stacks
        self.num_modules_per_stack = num_modules_per_stack
        self.num_labels = num_labels
        
        matrix, ops = arch
        spec = model_spec.ModelSpec(matrix, ops)
        self.model = NB101Network(spec, stem_out_channels = init_channels, num_stacks = num_stacks, num_modules_per_stack = num_modules_per_stack,num_labels = num_labels)
        self.out0 = nn.Sequential(nn.BatchNorm2d(init_channels),nn.Conv2d(init_channels,last_channels[0],1))
        self.out1 = nn.Sequential(nn.BatchNorm2d(init_channels*2),nn.Conv2d(init_channels*2,last_channels[1],1))
        self.out2 = nn.Sequential(nn.BatchNorm2d(init_channels*4),nn.Conv2d(init_channels*4,last_channels[2],1))
        
    def forward(self, x):
        count = 0
        for _, layer in enumerate(self.model.layers):
            if isinstance(layer,nn.MaxPool2d) and count == 0:
                x0 = self.out0(F.interpolate(x,(self.output_size,self.output_size)))
                count += 1
            elif isinstance(layer,nn.MaxPool2d) and count == 1:
                x1 = self.out1(F.interpolate(x,(self.output_size,self.output_size)))
                count += 1
            x = layer(x)
        x2 = self.out2(x)
        return [x0,x1,x2]

class NB201Wrapper(nn.Module):
    """
    wrap the NB201 model for regression
    e.g. arch: [[0],[0,1],[0,1,2]] -> arch_str: '|none~0|+|none~0|skip_connect~1|+|none~0|skip_connect~1|nor_conv_1x1~2|'
    """
    def __init__(self, arch, init_channels, last_channels, output_size = 8, num_labels = 10):
        super(NB201Wrapper, self).__init__()
        self.init_channels = init_channels
        self.last_channels = last_channels
        self.output_size = output_size
        self.num_labels = num_labels
        self.arch_str = generate_arch(arch)
        self.model = get_model_from_arch_str(self.arch_str, num_classes = num_labels, use_bn=True, init_channels=init_channels)
        self.out0 = nn.Sequential(nn.BatchNorm2d(init_channels),nn.Conv2d(init_channels,last_channels[0],1))
        self.out1 = nn.Sequential(nn.BatchNorm2d(init_channels*2),nn.Conv2d(init_channels*2,last_channels[1],1))
        self.out2 = nn.Sequential(nn.BatchNorm2d(init_channels*4),nn.Conv2d(init_channels*4,last_channels[2],1))
    def forward(self, x):
        x = self.model.stem(x)        
        x = self.model.stack_cell1(x)
        x0 = self.out0(F.interpolate((x),(self.output_size,self.output_size)))
        x = self.model.reduction1(x)
        x = self.model.stack_cell2(x)
        x1 = self.out1(F.interpolate((x),(self.output_size,self.output_size)))
        x = self.model.reduction2(x)
        x = self.model.stack_cell3(x)
        x2 = self.out2(x)
        return [x0,x1,x2]

class NBNLPWrapper(nn.Module):
    """
    wrap the NBNLP model for regression
    """
    def __init__(self, arch, last_channels):
        super(NBNLPWrapper, self).__init__()
        modelargs = Namespace(**arch)        
        self.model = NBNLPNetwork(modelargs.model, 
                               1, 
                               modelargs.emsize, 
                               modelargs.nhid, 
                               modelargs.nlayers, 
                               modelargs.dropout, 
                               modelargs.dropouth, 
                               modelargs.dropouti, 
                               modelargs.dropoute, 
                               modelargs.wdrop, 
                               modelargs.tied,
                               modelargs.recepie,
                               verbose=False)
        self.modelargs = modelargs
        self.nreg = last_channels
        self.decoder = torch.nn.Linear(modelargs.emsize, self.nreg)
        self.encoder = torch.nn.Linear(self.nreg, modelargs.emsize)
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.init_hidden = self.model.init_hidden
    def forward(self, input, hidden, return_h=False):
        N,B,C = input.shape
        raw_output = self.encoder(input.reshape(N*B,C)).reshape(N,B,-1)
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.model.rnns):
            raw_output, new_h = rnn(raw_output, hidden[i])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.model.nlayers - 1:
                #self.hdrop(raw_output) add??? 
                raw_output = self.model.lockdrop(raw_output, self.model.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        
        output = self.model.lockdrop(raw_output, self.model.dropout)
        
        outputs.append(output)
        
        result = self.decoder(output.reshape(N*B,-1)).reshape(N,B,-1)#.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden
    
class DARTSWrapper(nn.Module):
    """
    wrap the DARTS model for regression
    """
    def __init__(self, genotype, width, depth, last_channels, imagenet = False, output_size = 8):
        super(DARTSWrapper, self).__init__()
        self.genotype = genotype
        self.width = width
        self.depth = depth
        self.last_channels = last_channels 
        self.imagenet = imagenet
        self.output_size = output_size
        
        if imagenet:
            model = NetworkImageNet(width, 1, depth, False,  genotype)
        else:
            model = NetworkCIFAR(width, 1, depth, False, genotype)
            
        model.drop_path_prob = 0.
        if imagenet:
            model.stem0[0].stride = 1 # stem0 conv1
            model.stem0[3].stride = 1
            model.stem1[1].stride = 1 # stem1 conv1
            C_in, C_out = model.cells[0].preprocess0.conv_1.in_channels, model.cells[0].preprocess0.conv_1.out_channels * 2
            model.cells[0].preprocess0 = ReLUConvBN(C_in, C_out, 1, 1, 0)
        
        
        final_channels = model.classifier.classifier.in_features
        
        try: in_channel = model.cells[model._layers//3].preprocess1.op[1].in_channels
        except: in_channel = model.cells[model._layers//3].preprocess1.conv_1.in_channels
        model.out0 = nn.Sequential(nn.BatchNorm2d(in_channel),nn.Conv2d(in_channel,last_channels[0],1))
        try: in_channel = model.cells[2*model._layers//3].preprocess1.op[1].in_channels
        except: in_channel = model.cells[2*model._layers//3].preprocess1.conv_1.in_channels
        model.out1 = nn.Sequential(nn.BatchNorm2d(in_channel),nn.Conv2d(in_channel,last_channels[1],1))
        model.out2 = nn.Sequential(nn.BatchNorm2d(final_channels),nn.Conv2d(final_channels,last_channels[2],1))
        
        self.model = model
        
    def forward(self, x):
        if self.imagenet:
            s0 = self.model.stem0(x)
            s1 = self.model.stem1(s0)
        else:
            s0 = s1 = self.model.stem(x)
            
        for i, cell in enumerate(self.model.cells):
            if i == 1* self.model._layers//3:
                x2 = self.model.out0(F.interpolate((s1),(self.output_size,self.output_size)))
            if i == 2* self.model._layers//3:
                x3 = self.model.out1(F.interpolate((s1),(self.output_size,self.output_size)))
            s0, s1 = s1, cell(s0, s1, self.model.drop_path_prob)
        x4 = self.model.out2(s1)
        return [x2,x3,x4]
        
        
class ResNetSeriesWrapper(nn.Module):
    """
    wrap the ResNet, ResNeXt-A, ResNeXt-B model for regression
    """
    def __init__(self, config, last_channels, output_size = 8):
        super(ResNetSeriesWrapper, self).__init__()
        
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
        config['stem_type'] = 'res_stem_cifar' #'res_stem_in'
#         config['stem_type'] = 'simple_stem_in'
        #"res_stem_cifar": ResStemCifar,
        #"res_stem_in": ResStemIN,
        #"simple_stem_in": SimpleStemIN,
        if config['block_type'] == 'double_plain_block':
            config['block_type'] = 'vanilla_block'
            
        self.config = config
        self.last_channels = last_channels
        self.output_size = output_size
        
        width = config['stem_w']
        ws = config['ws']
        depth = config['ds']
        
        self.model = AnyNet(**config)
        
        
        self.out0 = nn.Sequential(nn.BatchNorm2d(ws[1]),nn.Conv2d(ws[1],last_channels[0],1))
        self.out1 = nn.Sequential(nn.BatchNorm2d(ws[2]),nn.Conv2d(ws[2],last_channels[1],1))
        self.out2 = nn.Sequential(nn.BatchNorm2d(ws[3]),nn.Conv2d(ws[3],last_channels[2],1))
        
        
        exec('self.model.s2.b%d.relu= DoNothing()'%(depth[-3]))
        exec('self.model.s3.b%d.relu= DoNothing()'%(depth[-2]))
        exec('self.model.s4.b%d.relu= DoNothing()'%(depth[-1]))
        
    def forward(self, x):
        x = self.model.stem(x)
        x1 = self.model.s1(x)
        x2 = self.model.s2((x1))
        x3 = self.model.s3(F.relu(x2))
        x4 = self.model.s4(F.relu(x3))
        
        x2 = self.out0(F.interpolate((x2),(self.output_size,self.output_size)))
        x3 = self.out1(F.interpolate((x3),(self.output_size,self.output_size)))
        x4 = self.out2(x4)
        return [x2,x3,x4]
        
        