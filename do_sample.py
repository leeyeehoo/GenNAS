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
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_channels', default=16, type=int, help='init channels')
    parser.add_argument('--last_channels', default=64, type=int, help='last channels')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--samples', type=int, default=1000, help='samples for sampling')
    parser.add_argument('--seed', type=int, default=111, help='pytorch manual seed')
    parser.add_argument('--search_space', default='nasbenchnlp',type=str, help='search space')
    parser.add_argument('--json_path', default='data/nasbench201/nasbench2_1000_0.json',
                        type=str, help='path to json file for nasbench101/201')
    parser.add_argument('--nds_path', default='../../GenNAS/data/nds_data/',
                        type=str, help='path to nds dataset')
    parser.add_argument('--nlp_path', default='data/nasbenchnlp',
                        type=str, help='path to nlp dataset')
    parser.add_argument('--config', default='CONF_NLP',
                        type=str, help='config')
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

def trainval(archs, model_builder, task, evaluator):
    preds = []
    for indx,arch in enumerate(archs):
        losses = evaluator.evaluate(task,model_builder,arch)
        logging.info("%d %s",indx, str(losses))
        preds.append([str(arch),losses])
    return preds

if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed) 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True #reproducibility
    torch.backends.cudnn.benchmark = False
    
    
    #BUILD LOG##################################################################
    path = './exp/sample_task_{}_{}'.format(args.job_description,time.strftime("%Y%m%d-%H%M%S"))
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
    if 'nlp' in args.search_space:
        task = NLPTask(args)
    else:
        task = CVTask(args)
        
    if 'nlp' in args.search_space:
        all_stats = []
        if 'wiki' in args.search_space:
            for fn in os.listdir(os.path.join(args.nlp_path,'train_logs_wikitext-2')):
                if fn.endswith('.json'):
                    all_stats.append(json.load(open(os.path.join(args.nlp_path,'train_logs_wikitext-2', fn), 'r')))
        else:
            for fn in os.listdir(os.path.join(args.nlp_path,'train_logs_single_run')):
                if fn.endswith('.json'):
                    all_stats.append(json.load(open(os.path.join(args.nlp_path,'train_logs_single_run', fn), 'r')))

        ok_stats = [x for x in all_stats if x['status'] == 'OK']
        sampled_stat = []
        for index, stat in enumerate(ok_stats):
            sampled_stat.append([index,stat])
        if args.samples < len(sampled_stat):
            sampled_stat = random.sample(sampled_stat,args.samples)
        archs = sampled_stat
    elif args.search_space == 'nasbench101' or args.search_space == 'nasbench201':
        with open(args.json_path,'r') as t:
                archs = json.load(t)
        if len(archs) > args.samples:
            archs = random.sample(archs,args.samples)
    elif args.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in','ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
        
        if len(model_builder.NDS) > args.samples:
            archs = random.sample(np.arange(len(model_builder.NDS)).tolist(),args.samples)
        else:
            archs = np.arange(len(model_builder.NDS)).tolist()
    with open(os.path.join(path,'subsample.json'),'w') as t:
        json.dump(archs,t)
        
    
    #BUILD CONFIG###################################################################
    start = time.time()
    results = trainval(archs, model_builder, task, evaluator)
    with open(os.path.join(path,'record.json'),'w') as t:
        json.dump(results,t)
    logging.info('end:%f'%(time.time()-start))