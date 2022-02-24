from foresight.dataset import *
from utils.tricks import *
from configs import *
import torch.nn as nn
import torch.nn.functional as F

class CVTask():
    def __init__(self, args):#currently we support C10/C100/IN16
        self.dataset = args.dataset
        self.config = args.config
        self.workers = args.workers
        self.pad = args.pad
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.device = args.device
        self.last_channels = args.last_channels
        if args.batch_size is not None:
            self.config['batchsize'] = args.batch_size #batch_size can be overrided
            
        if self.dataset == 'cifar10':
            self.if_split = True
        else:
            self.if_split = False
        if 'cifar' in self.dataset:
            self.meanstd = [[x / 255.0 for x in [125.3, 123.0, 113.9]],[x / 255.0 for x in [63.0, 62.1, 66.7]]]
        else:
            self.meanstd = None
            
        train_loader, val_loader = get_cifar_dataloaders(int(self.config['batchsize']), int(self.config['batchsize']), self.dataset, self.workers,if_split = self.if_split, pad=False, meanstd = self.meanstd)
        
        train_dataprovider = DataIterator(train_loader)
        data, _ = train_dataprovider.next()
        if self.dataset == 'ImageNet16-120':
            data = F.interpolate(data,(self.input_size,self.input_size))
        self.data = data
        self.targets = self._build_target()
        self.data = self.data.to(self.device)
        
        ###############################build target####################################
    def get_data(self):
        config = self.config
        device = self.device
        data = self.data
        if config['noise']['type'] == 'uniform':
            noise = torch.rand(data.shape).uniform_(-1*config['noise']['noise_level'],config['noise']['noise_level'])
        elif config['noise']['type'] == 'gaussian':
            noise = torch.rand(data.shape).normal_() * config['noise']['noise_level']
        noise = noise.to(device)
        targets = self.targets
        
        return data + noise, targets
    def _build_target(self):
        config = self.config
        device = self.device
        data = self.data
        targets = []
        for layer in range(3):
            target = torch.zeros(config['batchsize'],int(config['last_channel_l%d'%layer]*self.last_channels),self.output_size,self.output_size)
            if config['feature'][str(layer)]['sin1d'] is not None:
                target += generate_wave(config['feature'][str(layer)]['sin1d']['range'],config['batchsize'],int(config['last_channel_l%d'%layer]*self.last_channels), if_2d = False, local=config['feature'][str(layer)]['sin1d']['local']) * config['feature'][str(layer)]['sin1d']['level']
            if config['feature'][str(layer)]['sin2d'] is not None:
                target += generate_wave(config['feature'][str(layer)]['sin2d']['range'],config['batchsize'],int(config['last_channel_l%d'%layer]*self.last_channels), if_2d = True, local=config['feature'][str(layer)]['sin2d']['local']) * config['feature'][str(layer)]['sin2d']['level']
            if config['feature'][str(layer)]['resize'] is not None:
                resizedata = F.interpolate(data,(8,8))
                resizedata = resizedata - resizedata.min()
                resizedata = (resizedata /resizedata.max() - 0.5) * 2
                perm = torch.randperm(1000)
                idx = perm[:int(config['last_channel_l%d'%layer]*self.last_channels)]
                target += resizedata.repeat(1,1000,1,1)[:,idx] * config['feature'][str(layer)]['resize']['level']

            if config['feature'][str(layer)]['dot'] is not None:
                target += config['feature'][str(layer)]['dot']['level'] * generate_gaussian_dot(config['batchsize'],c = int(config['last_channel_l%d'%layer]*self.last_channels),n = int(config['feature'][str(layer)]['dot']['partial']* self.output_size * self.output_size),local = config['feature'][str(layer)]['dot']['local'], gaussian = False)

            if config['feature'][str(layer)]['dot_gaussian'] is not None:
                target += config['feature'][str(layer)]['dot_gaussian']['level'] * generate_gaussian_dot(config['batchsize'],c = int(config['last_channel_l%d'%layer]*self.last_channels),n = int(config['feature'][str(layer)]['dot_gaussian']['partial']* self.output_size * self.output_size),local = config['feature'][str(layer)]['dot_gaussian']['local'], gaussian = True)

            targets.append(target.to(device))    
        return targets
        
class NLPTask():
    def __init__(self,args):#currently we support NASBENCH-NLP
        self.config = args.config
        self.length = args.length
        self.device = args.device
        
        if args.batch_size is not None:
            self.config['batchsize'] = args.batch_size #batch_size can be overrided

        self.data, self.targets = self._build_target()
        self.data = self.data.squeeze(1).permute(1,0,2)
        self.targets = self.targets.squeeze(1).permute(1,0,2)
        ###############################build target####################################
    def get_data(self):
        config = self.config
        device = self.device
        data = self.data
        if config['noise']['type'] == 'uniform':
            noise = torch.rand(data.shape).uniform_(-1*config['noise']['noise_level'],config['noise']['noise_level'])
        elif config['noise']['type'] == 'gaussian':
            noise = torch.rand(data.shape).normal_() * config['noise']['noise_level']
        noise = noise.to(device)
        targets = self.targets
        
        return data + noise, targets
    
    def _build_target(self):
        config = self.config
        device = self.device
        targets = []
        for layer in range(2):
            target = torch.zeros(config['batchsize'],1,self.length,config['dimension'])
            if config['feature'][str(layer)]['sin1d'] is not None:
                target += generate_wave(config['feature'][str(layer)]['sin1d']['range'],config['batchsize'],1, if_2d = False, local=config['feature'][str(layer)]['sin1d']['local'], output_size = [self.length, config['dimension']]) * config['feature'][str(layer)]['sin1d']['level']
            if config['feature'][str(layer)]['sin2d'] is not None:
                target += generate_wave(config['feature'][str(layer)]['sin2d']['range'],config['batchsize'],1, if_2d = True, local=config['feature'][str(layer)]['sin2d']['local'], output_size = [self.length, config['dimension']]) * config['feature'][str(layer)]['sin2d']['level']


            if config['feature'][str(layer)]['dot'] is not None:
                target += config['feature'][str(layer)]['dot']['level'] * generate_gaussian_dot(config['batchsize'],c = 1,n = int(config['feature'][str(layer)]['dot']['partial'] * self.length * config['dimension']),local = config['feature'][str(layer)]['dot']['local'], gaussian = False, output_size = [self.length, config['dimension']])

            if config['feature'][str(layer)]['dot_gaussian'] is not None:
                target += config['feature'][str(layer)]['dot_gaussian']['level'] * generate_gaussian_dot(config['batchsize'],c = 1,n = int(config['feature'][str(layer)]['dot_gaussian']['partial'] * self.length * config['dimension']),local = config['feature'][str(layer)]['dot_gaussian']['local'], gaussian = True, output_size = [self.length, config['dimension']])

            targets.append(target.to(device))    
        return targets
        
        
        