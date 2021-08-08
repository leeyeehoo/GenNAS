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

import torch
import numpy as np
import scipy.stats as stats

def parse_arguments():
    parser = argparse.ArgumentParser(description='Search for a task')
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
    parser.add_argument('--samples', type=int, default=20, help='samples for searching')
    parser.add_argument('--seed', type=int, default=1, help='pytorch manual seed')
    parser.add_argument('--search_space', default='nlp',type=str, help='search space')
    parser.add_argument('--json_loc', default='data/nasbench1_search_20samples.json',
                        type=str, help='path to json file for nasbench')
    parser.add_argument('--json_description',default='nasbench1_search_20samples',type = str,help = 'default of json file')
    parser.add_argument('--population_size', default=50, type=int, help='population size')
    parser.add_argument('--tournament_size', default=10, type=int, help='tournament size')
    parser.add_argument('--evolve_size', default=400, type=int, help='evolve size')
    parser.add_argument('--mutation_rate', type=float, default=0.8, help='mutation rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda:"+ "0" if torch.cuda.is_available() else "cpu") 
    return args


def trainval(args, config):
    preds = []
    accs = []
    archs = args.archs
    model_builder = args.model_builder
    model_builder.config = config
    if 'nlp' in args.search_space:
        task = NLPTask(config, device = args.device,length = 8, batch_size = args.batch_size)
        evaluator = EvaluatorNLP(learning_rate = 1e-3, weight_decay = 1.2e-06, momentum = 0.9, init_w_type = 'none', init_b_type = 'none', device = 'cpu', total_iters = 100, eval_interval = 10, train_weights = [1.,1.,1.], eval_weights = [0.25, 0.5, 1.])
        
    else:
        task = CVTask(dataset = args.dataset,config = config, device = args.device,last_channels = args.last_channels, batch_size = args.batch_size)
        evaluator = Evaluator(learning_rate = args.learning_rate, weight_decay = args.weight_decay, momentum = args.momentum, init_w_type = args.init_w_type, init_b_type = args.init_b_type, device = args.device, total_iters = args.total_iters, eval_interval = args.eval_interval)
    for indx,arch in enumerate(archs):
        if 'nlp' in args.search_space:
            
            arch_info = arch
            acc = np.exp(np.min(arch['test_losses']))
        else:
            if 'nasbench' in args.search_space:
                arch_info = arch[0]
                acc = arch[1]
            else:
                arch_info = arch
                acc = model_builder.NDS.get_final_accuracy(arch,None,None)
        accs.append(acc)
        losses = evaluator.evaluate(task,model_builder,arch_info)
        preds.append(losses[-1])
    return abs(stats.kendalltau(preds,accs)[0]),abs(stats.spearmanr(preds,accs)[0]), preds

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
        
    path = './exp/search_task_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.search_space,json_description,args.samples,args.init_w_type,args.init_b_type,args.init_channels,args.last_channels,args.dataset,args.total_iters, args.batch_size,args.seed)
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
        for fn in os.listdir('./data/train_logs_single_run'):
            if fn.endswith('.json'):
                all_stats.append(json.load(open(os.path.join('./data/train_logs_single_run', fn), 'r')))
        ok_stats = [x for x in all_stats if x['status'] == 'OK']
        sampled_stat = []
        for stat in ok_stats:
            sampled_stat.append(stat)
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
                archs = json.load(t)
            if isinstance(archs,list):
                args.archs = random.sample(archs,args.samples)
            else:
                perms = random.sample(list(archs),args.samples)
                args.archs = []
                for perm in perms:
                    args.archs.append([perm,archs[perm]['mean_acc']])
        else:
            if len(model_builder.NDS) > args.samples:
                args.archs = random.sample(np.arange(len(model_builder.NDS)).tolist(),args.samples)
            else:
                args.archs = np.arange(len(model_builder.NDS)).tolist()
        with open(os.path.join(path,'subsample.json'),'w') as t:
            json.dump(args.archs,t)
    args.model_builder = model_builder
    #BUILD CONFIG###################################################################
    population = []
    benchmarks = []
    start = time.time()
    for ite in range(args.population_size):
        if 'nlp' in args.search_space:
            config = generate_config_nlp(global_config_nlp)
        else:
            config = generate_config(global_config)
            
        tau,spearmanr,preds = trainval(args,config)
        population.append([config,spearmanr])
        logging.info("%d %s %f %f",ite,str(config),tau,spearmanr)
        benchmarks.append([str(config),spearmanr])
    for ite in range(args.evolve_size):
        sample = random.sample(population, args.tournament_size)
        best_config = sorted(sample, key=lambda i:i[1],reverse=True)[0][0]
        if 'nlp' in args.search_space:
            new_config = mutate_config_nlp(global_config_nlp,best_config, p=args.mutation_rate)
        else:
            new_config = mutate_config(global_config,best_config, p=args.mutation_rate)
        tau,spearmanr,preds = trainval(args,new_config)
        population.append([new_config,spearmanr])
        population.pop(0)
        logging.info("%d %s %f %f",ite,str(new_config),tau,spearmanr)
        benchmarks.append([str(new_config),spearmanr])
    results = benchmarks
    with open(os.path.join(path,'record.json'),'w') as t:
        json.dump(results,t)
    logging.info('end:%f'%(time.time()-start))