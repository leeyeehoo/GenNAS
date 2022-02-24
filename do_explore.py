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
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_channels', default=16, type=int, help='init channels')
    parser.add_argument('--last_channels', default=64, type=int, help='last channels')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--seed', type=int, default=111, help='pytorch manual seed')
    parser.add_argument('--search_space', default='nasbenchnlp',type=str, help='search space')
    
    parser.add_argument('--nds_path', default='../../GenNAS/data/nds_data/',
                        type=str, help='path to nds dataset')
    parser.add_argument('--nlp_path', default='data/nasbenchnlp',
                        type=str, help='path to nlp dataset')
    
    parser.add_argument('--config', default='CONF_NLP',
                        type=str, help='config')
    
    parser.add_argument('--population_size', default=50, type=int, help='population size')
    parser.add_argument('--tournament_size', default=10, type=int, help='tournament size')
    parser.add_argument('--evolve_size', default=400, type=int, help='evolve size')
    parser.add_argument('--mutate_ratio', type=float, default=1., help='mutate rate')
    parser.add_argument('--neighbor', type=int, default=200, help='simulated sample neighbor')
    
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--train_weights', default=[0.25,0.5,1.], type=float, nargs='+')
    parser.add_argument('--eval_weights', default=[0.25,0.5,1.], type=float, nargs='+')
    parser.add_argument('--workers', default=2, type=int, help='workers')
    parser.add_argument('--pad', action='store_true', help='add padding for real images random crop')
    parser.add_argument('--input_size', default=32, type=int, help='input size for cv task')
    parser.add_argument('--output_size', default=8, type=int, help='output size for cv task')
    parser.add_argument('--length', default=8, type=int, help='length for nlp task')
    parser.add_argument('--job_description',default='nasbench2_1000_0',type = str,help = 'job description')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = "cuda:"+ "0" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed) 
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #BUILD LOG##################################################################
    path = './exp/explore_task_{}_{}'.format(args.job_description,time.strftime("%Y%m%d-%H%M%S"))
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
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(vars(args),f)
    #BUILD CONFIG###################################################################
    args.config = eval(args.config)
    model_builder = ModelBuilder(args)
    evaluator = Evaluator(args)
    explorer = Explorer(model_builder,args)
    if 'nlp' in args.search_space:
        task = NLPTask(args)
    else:
        task = CVTask(args)
    #BUILD CONFIG###################################################################
    population = []
    benchmarks = []
    start = time.time()
    logging.info('begin generating the polulation')
    for ite in range(args.population_size):
        arch = explorer.random_spec()
        losses = evaluator.evaluate(task,model_builder,arch)
        population.append([arch,losses[-1]])
        benchmarks.append([str(arch),losses[-1]])
        logging.info("%d %f",ite,losses[-1])
    logging.info('begin mutation')
    for ite in range(args.evolve_size):
        sample = random.sample(population, args.tournament_size)
        best_arch = sorted(sample, key=lambda i:i[1])[0][0]
        arch = explorer.mutate_spec(best_arch)
        losses = evaluator.evaluate(task,model_builder,arch)
        population.append([arch,losses[-1]])
        population.pop(0)
        logging.info("%d %f",ite,losses[-1])
        benchmarks.append([str(arch),losses[-1]])
    results = benchmarks
    with open(os.path.join(path,'record.json'),'w') as t:
        json.dump(results,t)
    logging.info('end:%f'%(time.time()-start))