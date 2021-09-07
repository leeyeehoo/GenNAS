import json
import argparse
import random
import time
import logging
from builder_task import *
from builder_model import *
from builder_evaluator import *
from builder_explorer import *
from utils.config_generator import *
from utils.config_generator_nlp import *

import torch
import numpy as np
import scipy.stats as stats

def parse_arguments():
    parser = argparse.ArgumentParser(description='Search for a task')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--total_iters', type=int, default=100, help='total iters')
    parser.add_argument('--eval_interval', type=int, default=100, help='evaluate interval')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_channels', default=16, type=int, help='init channels')
    parser.add_argument('--last_channels', default=64, type=int, help='last channels')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--seed', type=int, default=111, help='pytorch manual seed')
    parser.add_argument('--search_space', default='nlp',type=str, help='search space')
    
    parser.add_argument('--config', default='CONF_NLP',
                        type=str, help='config')
    
    parser.add_argument('--population_size', default=50, type=int, help='population size')
    parser.add_argument('--tournament_size', default=10, type=int, help='tournament size')
    parser.add_argument('--evolve_size', default=400, type=int, help='evolve size')
    parser.add_argument('--mutation_rate', type=float, default=1., help='mutation rate')
    parser.add_argument('--neighbor', type=int, default=200, help='simulated sample neighbor')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda:"+ "0" if torch.cuda.is_available() else "cpu") 
    return args


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed) 
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = './exp/explore_task_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.search_space,args.config,args.init_w_type,args.init_b_type,args.init_channels,args.last_channels,args.dataset,args.total_iters,args.batch_size,args.seed)
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
    config = eval(args.config)
    model_builder = ModelBuilder(search_space = args.search_space , config = config, device =args.device,last_channels = args.last_channels,init_channels = args.init_channels)
    
    explorer = Explorer(model_builder,mutate_ratio = args.mutation_rate, neighbor = args.neighbor)
    if 'nlp' in args.search_space:
        task = NLPTask(config, device = args.device,length = 8, batch_size = args.batch_size)
        evaluator = EvaluatorNLP(learning_rate = 1e-3, weight_decay = 1.2e-06, momentum = 0.9, init_w_type = 'none', init_b_type = 'none', device = 'cpu', total_iters = 100, eval_interval = 10, train_weights = [1.,1.,1.], eval_weights = [0.25, 0.5, 1.])
    else:
        task = CVTask(dataset = args.dataset,config = config, device = args.device,last_channels = args.last_channels,batch_size = args.batch_size)
        evaluator = Evaluator(learning_rate = args.learning_rate, weight_decay = args.weight_decay, momentum = args.momentum, init_w_type = args.init_w_type, init_b_type = args.init_b_type, device = args.device, total_iters = args.total_iters, eval_interval = args.eval_interval)
    #BUILD CONFIG###################################################################
    population = []
    benchmarks = []
    start = time.time()
    for ite in range(args.population_size):
        if args.search_space== 'nlp':
            arch,arch_info = explorer.random_spec()
        else:
            arch = explorer.random_spec()
            arch_info = arch
        losses = evaluator.evaluate(task,model_builder,arch_info)
        population.append([arch,losses[-1]])
        logging.info("%d %s %f",ite,str(arch),losses[-1])
        benchmarks.append([str(arch),losses[-1]])
    for ite in range(args.evolve_size):
        sample = random.sample(population, args.tournament_size)
        best_arch = sorted(sample, key=lambda i:i[1])[0][0]
        if args.search_space== 'nlp':
            new_arch,arch_info = explorer.mutate_spec(best_arch)
        else:
            new_arch = explorer.mutate_spec(best_arch)
            arch_info = new_arch
        losses = evaluator.evaluate(task,model_builder,arch_info)
        population.append([new_arch,losses[-1]])
        population.pop(0)
        logging.info("%d %s %f",ite,str(new_arch),losses[-1])
        benchmarks.append([str(new_arch),losses[-1]])
    results = benchmarks
    with open(os.path.join(path,'record.json'),'w') as t:
        json.dump(results,t)
    logging.info('end:%f'%(time.time()-start))