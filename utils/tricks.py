#some are abandoned
import torch
import torch.nn as nn
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter 
import random

def generate_arch(structure):
    NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    strings = []
    for i in range(3):
        
        string = '|'.join([NAS_BENCH_201[structure[i][k]]+'~{:}'.format(k) for k in range(i+1)])
        string = '|{:}|'.format(string)
        strings.append( string )
    return '+'.join(strings)

def generate_wave(betas,batchsize,channel, if_2d = False, local = True, input_size = 32, output_size = 8, num_pos_feats = 32):
    isnlp = False
    if isinstance(output_size, list):
        assert len(output_size) == 2
        isnlp = True
        l,d = output_size
        output_size = max(output_size)
    if local == False:
        num_pos_feats = num_pos_feats
        poses = []
        if if_2d is False:
            for beta in betas:
                not_mask = torch.ones(1,output_size,output_size)
                y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 2 * beta #PI/8
                x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 2 * beta
                step = torch.rand(num_pos_feats,dtype=torch.float32) * math.pi * 2
                pos_x = x_embed[:, :, :, None] + step
                pos_y = y_embed[:, :, :, None] + step
                pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
                poses.append(pos.squeeze(0))
        if if_2d:
            for beta1 in betas:
                for beta2 in betas:
                    not_mask = torch.ones(1,output_size,output_size)
                    y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 2 * beta1#PI/8
                    x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 2 * beta2
                    step = torch.rand(num_pos_feats,dtype=torch.float32) * math.pi * 2
                    pos_x = x_embed[:, :, :, None] + step
                    pos_y = y_embed[:, :, :, None] + step
                    pos = pos_x + pos_y
                    inv_idx = torch.arange(output_size-1, -1, -1).long()
                    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos = torch.cat((pos, pos[:,inv_idx,:,:]), dim=3).permute(0, 3, 1, 2)
                    poses.append(pos.squeeze(0))
        poses = torch.cat(poses,dim=0)
        perm = torch.randperm(len(poses))
        idx = perm[:channel]
        poses = poses[idx]
        if isnlp:
            return poses.unsqueeze(0).repeat(batchsize,1,1,1)[:,:,:l,:d]
        return poses.unsqueeze(0).repeat(batchsize,1,1,1)
    else:
        posess = []
        for bt in range(batchsize):
            num_pos_feats = 32
            poses = []
            if if_2d is False:
                for beta in betas:
                    not_mask = torch.ones(1,output_size,output_size)
                    y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 2 * beta#PI/8
                    x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 2 * beta
                    step = torch.rand(num_pos_feats,dtype=torch.float32) * math.pi * 2
                    pos_x = x_embed[:, :, :, None] + step
                    pos_y = y_embed[:, :, :, None] + step
                    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
                    poses.append(pos.squeeze(0))
            if if_2d:
                for beta1 in betas:
                    for beta2 in betas:
                        not_mask = torch.ones(1,output_size,output_size)
                        y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 2 * beta1#PI/8
                        x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 2 * beta2
                        step = torch.rand(num_pos_feats,dtype=torch.float32) * math.pi * 2
                        pos_x = x_embed[:, :, :, None] + step
                        pos_y = y_embed[:, :, :, None] + step
                        pos = pos_x + pos_y
                        inv_idx = torch.arange(output_size-1, -1, -1).long()

                        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(3)
                        pos = torch.cat((pos, pos[:,inv_idx,:,:]), dim=3).permute(0, 3, 1, 2)
                        poses.append(pos.squeeze(0))
            poses = torch.cat(poses,dim=0)
            perm = torch.randperm(len(poses))
            idx = perm[:channel]
            poses = poses[idx]
            posess.append(poses.unsqueeze(0))
        if isnlp:
            return torch.cat(posess,dim=0)[:,:,:l,:d]
        return torch.cat(posess,dim=0)
    
def generate_gaussian_dot(b,c = 36,n = 16, local = False, gaussian = False, input_size = 32, output_size = 8):
    isnlp = False
    if isinstance(output_size, list):
        assert len(output_size) == 2
        isnlp = True
        l,d = output_size
        output_size = max(output_size)
    if local is True:
        if isnlp:
            target = np.zeros((b,c,l,d))
        else:
            target = np.zeros((b,c,output_size,output_size))
        
        for tb in range(b):
            for tc in range(c):
                i = 0
                while i < n:
                    if isnlp:
                        x1 = np.random.randint(0,l)
                        x2 = np.random.randint(0,d)
                    else:
                        x1 = np.random.randint(0,output_size)
                        x2 = np.random.randint(0,output_size)
                    if target[tb,tc,x1,x2] == 0:
                        if random.random() > 0.5:
                            target[tb,tc,x1,x2] = 1
                        else:
                            target[tb,tc,x1,x2] = -1
                        i += 1
                if gaussian:
                    target[tb,tc] = gaussian_filter(target[tb,tc],1)
                    target[tb,tc] = target[tb,tc]/max(abs(target[tb,tc].min()),abs(target[tb,tc].max()))
        
        target = torch.from_numpy(target)
#         target = target/max(abs(target.min()),abs(target.max()))
    else:
        if isnlp:
            target = np.zeros((c,l,d))
        else:
            target = np.zeros((c,output_size,output_size))
        for tc in range(c):
            i = 0
            while i < n:
                if isnlp:
                    x1 = np.random.randint(0,l)
                    x2 = np.random.randint(0,d)
                else:
                    x1 = np.random.randint(0,output_size)
                    x2 = np.random.randint(0,output_size)
                if target[tc,x1,x2] == 0:
                    if random.random() > 0.5:
                        target[tc,x1,x2] = 1
                    else:
                        target[tc,x1,x2] = -1
                    i += 1
            if gaussian:
                target[tc] = gaussian_filter(target[tc],1)
                target[tc] = target[tc]/max(abs(target[tc].min()),abs(target[tc].max()))
        target = torch.from_numpy(target)
        target = target.repeat(b,1,1,1)
#         target = target/max(abs(target.min()),abs(target.max()))
    
    return target.float()


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class FullConv(nn.Module):
    def __init__(self, in_dims, last_channels, use_bn=True):
        super(FullConv, self).__init__()
        if use_bn:
            self.lastact = nn.Sequential(nn.BatchNorm2d(in_dims), nn.ReLU(inplace=True))
        else:
            self.lastact = nn.ReLU(inplace=True)
        self.classifier  = nn.Conv2d(in_dims, last_channels,1)

    def forward(self, x):
        x = self.lastact(x)
        logits = self.classifier(x)
        return logits
    
class FullConvBN(nn.Module):
    def __init__(self, in_dims, last_channels, use_bn=True):
        super(FullConvBN, self).__init__()
        self.classifier  = nn.Sequential(nn.BatchNorm2d(in_dims), nn.Conv2d(in_dims, last_channels,1))

    def forward(self, x):
        logits = self.classifier(x)
        return logits    
class DoNothing(nn.Module):
    def __init__(self):
        super(DoNothing, self).__init__()
    def forward(self, x):
        return x
class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]
def generate_pos(betas,batchsize,channel, if_2d = False):
    num_pos_feats = 36
#     betas = np.arange(start,end,0.01)   
#     betas = random.sample(list(betas),10)
    
    poses = []
    if if_2d is False:
        for beta in betas:
            not_mask = torch.ones(1,8,8)
            y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 1/8 * beta#PI/8
            x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 1/8 * beta

            step = torch.arange(num_pos_feats,dtype=torch.float32) * math.pi/num_pos_feats
            pos_x = x_embed[:, :, :, None] + step
            pos_y = y_embed[:, :, :, None] + step
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            poses.append(pos.squeeze(0))
    if if_2d:
        for beta1 in betas:
            for beta2 in betas:
                not_mask = torch.ones(1,8,8)
                y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 1/8 * beta1#PI/8
                x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 1/8 * beta2
                step = torch.arange(num_pos_feats,dtype=torch.float32) * math.pi/num_pos_feats
                pos_x = x_embed[:, :, :, None] + step
                pos_y = y_embed[:, :, :, None] + step
                pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos = torch.cat((pos_y*pos_x, (pos_x*pos_y).transpose(1,2)), dim=3).permute(0, 3, 1, 2)
                poses.append(pos.squeeze(0))
    poses = torch.cat(poses,dim=0)
    perm = torch.randperm(len(poses))
    idx = perm[:channel]
    poses = poses[idx]
    return poses.unsqueeze(0).repeat(batchsize,1,1,1)

def generate_position(betas,batchsize,channel, if_2d = False):
    num_pos_feats = 32

    poses = []
    if if_2d is False:
        for beta in betas:
            not_mask = torch.ones(1,8,8)
            y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 1/8 * beta#PI/8
            x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 1/8 * beta

            step = torch.arange(num_pos_feats,dtype=torch.float32) * math.pi/num_pos_feats
            pos_x = x_embed[:, :, :, None] + step
            pos_y = y_embed[:, :, :, None] + step
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            poses.append(pos.squeeze(0))
    if if_2d:
        for beta1 in betas:
            for beta2 in betas:
                not_mask = torch.ones(1,8,8)
                y_embed = not_mask.cumsum(1, dtype=torch.float32)* math.pi * 1/8 * beta1#PI/8
                x_embed = not_mask.cumsum(2, dtype=torch.float32)* math.pi * 1/8 * beta2
                step = torch.arange(num_pos_feats,dtype=torch.float32) * math.pi/num_pos_feats
                pos_x = x_embed[:, :, :, None] + step
                pos_y = y_embed[:, :, :, None] + step
                pos = pos_x + pos_y
                pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(3)
                pos = torch.cat((pos, (pos).transpose(1,2)), dim=3).permute(0, 3, 1, 2)
                poses.append(pos.squeeze(0))
    poses = torch.cat(poses,dim=0)
    perm = torch.randperm(len(poses))
    idx = perm[:channel]
    poses = poses[idx]
    return poses.unsqueeze(0).repeat(batchsize,1,1,1)

def generate_dot(b,c = 36,n = 10):
    x = torch.ones(b,c,8,8) * -1
    for tb in range(b):
        for tc in range(c):
            i = 0
            while i < n:
                x1 = np.random.randint(0,8)
                x2 = np.random.randint(0,8)
                if x[tb,tc,x1,x2] == -1:
                    x[tb,tc,x1,x2] += 2
                    i += 1
    return x

def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups