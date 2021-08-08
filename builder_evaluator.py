import torch.nn as nn
import torch
import random
from foresight.weight_initializers import init_net
from utils.tricks import *
from argparse import Namespace

class Evaluator():
    def __init__(self, learning_rate = 1e-1, weight_decay = 4e-5, momentum = 0.9, init_w_type = 'none', init_b_type = 'none', device = 'cpu', total_iters = 100, eval_interval = 10, train_weights = [1.,1.,1.], eval_weights = [0.25, 0.5, 1.]):
        self.total_iters = total_iters
        self.eval_interval = eval_interval
        self.init_w_type = init_w_type
        self.init_b_type = init_b_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.train_weights = train_weights
        self.eval_weights = eval_weights
    def evaluate(self,task,model_builder,arch):
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
            output = model_builder.learn(model,data)
            loss = train_weights[0] * loss_function(output[0], targets[0])  + train_weights[1] * loss_function(output[1], targets[1])  + train_weights[2] * loss_function(output[2], targets[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iters%self.eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    data,targets = task.get_data()
                    output = model_builder.learn(model,data)
                    loss = loss_function(output[0], targets[0]) * eval_weights[0] + loss_function(output[1], targets[1]) * eval_weights[1] + loss_function(output[2], targets[2]) * eval_weights[2]
                if np.isnan(float(loss.item())):
                    losses.append(1e3 + random.random())
                else:
                    losses.append(float(loss.item()))
        return losses

class EvaluatorNLP():
    def __init__(self, learning_rate = 1e-3, weight_decay = 1.2e-06, momentum = 0.9, init_w_type = 'none', init_b_type = 'none', device = 'cpu', total_iters = 100, eval_interval = 10, train_weights = [1.,1.,1.], eval_weights = [0.25, 0.5, 1.]):
        self.total_iters = total_iters
        self.eval_interval = eval_interval
        self.init_w_type = init_w_type
        self.init_b_type = init_b_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.train_weights = train_weights
        self.eval_weights = eval_weights
    def evaluate(self,task,model_builder,arch):
        modelargs = Namespace(**arch)
        
        model = model_builder.get_model_nlp(modelargs)
        
        init_net(model, self.init_w_type, self.init_b_type)
        optimizer = torch.optim.Adam(params = list(model.parameters()), lr=self.learning_rate, weight_decay=modelargs.wdecay)
        
        loss_function = nn.MSELoss().to(self.device)
        losses = []
        train_weights = self.train_weights
        eval_weights = self.eval_weights
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