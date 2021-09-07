import torch.nn as nn
import torch
import random
from foresight.weight_initializers import init_net
from utils.tricks import *
from argparse import Namespace

class Evaluator():
    def __init__(self, args):
        self.total_iters = args.total_iters
        self.eval_interval = args.eval_interval
        self.init_w_type = args.init_w_type
        self.init_b_type = args.init_b_type
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.device = args.device
        if 'train_weights' in args.config: 
            self.train_weights = args.config['train_weights']
        else: 
            self.train_weights = args.train_weights
        if 'eval_weights' in args.config: 
            self.eval_weights = args.config['eval_weights']
        else: 
            self.eval_weights = args.eval_weights
    def evaluate(self,task,model_builder,arch):
        if 'nlp' in model_builder.search_space:
            return self.evaluate_nlp(task,model_builder,arch)
        else :
            return self.evaluate_cv(task,model_builder,arch)
        
    def evaluate_cv(self,task,model_builder,arch):
        model = model_builder.get_model(arch)
        init_net(model, self.init_w_type, self.init_b_type)
        optimizer = torch.optim.SGD(get_parameters(model),
                            lr=self.learning_rate,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay)
        loss_function = nn.MSELoss().to(self.device)
        losses = []
        train_weights = self.train_weights
        eval_weights = self.eval_weights
        for iters in range(1,self.total_iters+1):
            model.train()
            data,targets = task.get_data()
            output = model(data)
            loss = train_weights[2] * loss_function(output[2], targets[2])
            if train_weights[1] != 0:
                loss += train_weights[1] * loss_function(output[1], targets[1])
            if train_weights[0] != 0:
                loss += train_weights[0] * loss_function(output[0], targets[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iters%self.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    data,targets = task.get_data()
                    output = model(data)
                    loss = eval_weights[2] * loss_function(output[2], targets[2])
                    if train_weights[1] != 0:
                        loss += eval_weights[1] * loss_function(output[1], targets[1])
                    if train_weights[0] != 0:
                        loss += eval_weights[0] * loss_function(output[0], targets[0])
                        
                if np.isnan(float(loss.item())):
                    losses.append(1e3 + random.random())
                else:
                    losses.append(float(loss.item()))
        return losses
    
    def evaluate_nlp(self,task,model_builder,arch):
        model = model_builder.get_model(arch)
        init_net(model, self.init_w_type, self.init_b_type)
        optimizer = torch.optim.Adam(params = list(model.parameters()), lr=self.learning_rate, weight_decay=model.modelargs.wdecay)
        loss_function = nn.MSELoss().to(self.device)
        losses = []
        for iters in range(1,self.total_iters+1):
            model.train()
            data,targets = task.get_data()
            hidden = model.init_hidden(model_builder.config['batchsize'])
            output, hidden = model(data, hidden)
            
            loss = loss_function(output, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iters%self.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    data,targets = task.get_data()
                    hidden = model.init_hidden(model_builder.config['batchsize'])
                    output, hidden = model(data, hidden)

                    loss = loss_function(output, targets)
                if np.isnan(float(loss.item())):
                    losses.append(1e3 + random.random())
                else:
                    losses.append(float(loss.item()))
        return losses
    
