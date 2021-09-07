import json
import argparse
import random
import time
import logging
from builder_task import *
from builder_model import *
from builder_evaluator import *
from utils.config_generator import *
from utils.config_generator_nlp import *

from configs import *
import torch
import numpy as np
import scipy.stats as stats
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sampling experiment')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--total_iters', type=int, default=100, help='total iters')
    parser.add_argument('--eval_interval', type=int, default=10, help='evaluate interval')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_channels', default=16, type=int, help='init channels')
    parser.add_argument('--last_channels', default=64, type=int, help='last channels')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--samples', type=int, default=1000, help='samples for searching')
    parser.add_argument('--seed', type=int, default=111, help='pytorch manual seed')
    parser.add_argument('--search_space', default='nasbench101',type=str, help='search space')
    parser.add_argument('--json_loc', default='data/nasbench1_500_fb.json',
                        type=str, help='path to json file for nasbench')
    parser.add_argument('--json_description',default='nasbench1_500_fb',type = str,help = 'default of json file')
    parser.add_argument('--config', default='CONF_COMBO',
                        type=str, help='config')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda:"+ "0" if torch.cuda.is_available() else "cpu")    
    return args


def trainval(args):
    preds = []
    archs = args.archs
    config = eval(args.config)
    model_builder = args.model_builder
    model_builder.config = config
    if 'nlp' in args.search_space:
        task = NLPTask(config, device = args.device,length = 8, batch_size = args.batch_size)
        evaluator = EvaluatorNLP(learning_rate = 1e-3, weight_decay = 1.2e-06, momentum = 0.9, init_w_type = 'none', init_b_type = 'none', device = 'cpu', total_iters = 100, eval_interval = 10, train_weights = [1.,1.,1.], eval_weights = [0.25, 0.5, 1.])
        
    else:
        task = CVTask(dataset = args.dataset,config = config, device = args.device,last_channels = args.last_channels, batch_size = args.batch_size)
        evaluator = Evaluator(learning_rate = args.learning_rate, weight_decay = args.weight_decay, momentum = args.momentum, init_w_type = args.init_w_type, init_b_type = args.init_b_type, device = args.device, total_iters = args.total_iters, eval_interval = args.eval_interval)
    for indx,arch in enumerate(archs):
        arch_info = arch
        losses = evaluator.evaluate(task,model_builder,arch_info)
        if 'nlp' in args.search_space:
            logging.info("%d %s %s",indx,str(arch['recepie_id']), str(losses))
        else:
            logging.info("%d %s %s",indx,str(arch), str(losses))
        preds.append([arch_info,losses])
    return preds

if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed) 
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if 'nasbench' in args.search_space:
        json_description = args.json_description
    else:
        json_description = 'none'
    path = './exp/sample_task_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.search_space,json_description,args.config,args.samples,args.init_w_type,args.init_b_type,args.init_channels,args.last_channels,args.dataset,args.total_iters,args.batch_size,args.seed)
    os.mkdir(path)
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    fh = logging.FileHandler(os.path.join(path,'train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)
    ###################################################################
    model_builder = ModelBuilder(search_space = args.search_space , config = None, device =args.device,last_channels = args.last_channels,init_channels = args.init_channels)
    if 'nlp' in args.search_space:
        all_stats = []
        if 'wiki' in args.search_space:
            for fn in os.listdir('./data/train_logs_wikitext-2'):
                if fn.endswith('.json'):
                    all_stats.append(json.load(open(os.path.join('./data/train_logs_wikitext-2', fn), 'r')))
        else:
            for fn in os.listdir('./data/train_logs_single_run'):
                if fn.endswith('.json'):
                    all_stats.append(json.load(open(os.path.join('./data/train_logs_single_run', fn), 'r')))
        
        
        ok_stats = [x for x in all_stats if x['status'] == 'OK']
        sampled_stat = []
        for stat in ok_stats:
            sampled_stat.append(stat)
        if args.samples < len(sampled_stat):
            sampled_stat = random.sample(sampled_stat,args.samples)
        recipes = []
        for inds,stat in enumerate(sampled_stat):
            recipes.append(stat['recepie_id'])
        with open(os.path.join(path,'subsample.json'),'w') as t:
            json.dump(recipes,t)
        args.archs = sampled_stat
    else:
        if 'nasbench' in args.search_space:
            with open(args.json_loc,'r') as t:
                args.archs = json.load(t)
            if len(args.archs) > args.samples:
                args.archs = random.sample(args.archs,args.samples)
        else:
            if len(model_builder.NDS) > args.samples:
                args.archs = random.sample(np.arange(len(model_builder.NDS)).tolist(),args.samples)
            else:
                args.archs = np.arange(len(model_builder.NDS)).tolist()

    with open(os.path.join(path,'subsample.json'),'w') as t:
        json.dump(args.archs,t)
    args.model_builder = model_builder
    #BUILD CONFIG###################################################################
    start = time.time()
    results = trainval(args)
    with open(os.path.join(path,'record.json'),'w') as t:
        json.dump(results,t)
    logging.info('end:%f'%(time.time()-start))