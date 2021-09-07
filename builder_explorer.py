#support NB101, NB201, Simulated NDS ResN-series
from pynbs.nasbench101 import nb101_generator,model_spec
from pynbs.nasbenchnlp.nas_environment import Environment
import copy
import random
import numpy as np
from scipy.spatial import distance
import pandas as pd
import os
# some functions
def mutate_embedded(e, std=1, axes_bounds=None):
    e_new = e + np.random.randn(len(e)) * std
    if axes_bounds is not None:
        e_new = np.clip(e_new, axes_bounds[0], axes_bounds[1])
    return e_new
def find_closest(E, e):
    #dists = np.linalg.norm(E - e.reshape(1, -1), axis=1)
    dists = distance.cdist([e], E, "cosine")[0]
    return np.argmin(dists)
def get_diff(conf_A,conf_B):
    count = 0
    for item in conf_A:
        if isinstance(conf_A[item],list):
            for it_A,it_B in zip(conf_A[item],conf_B[item]):
                if it_A != it_B:
                    count += 1
        else:
            if conf_A[item] != conf_B[item]:
                count += 1
    return count

class Explorer():
    def __init__(self, model_builder, args):
        self.mutate_ratio = args.mutate_ratio
        self.neighbor = args.neighbor
        self.NDS = model_builder.NDS
        self.search_space = model_builder.search_space
        self.env = None
        self.nlp_path = args.nlp_path #default data/nasbenchnlp
        
        if self.search_space == 'nasbenchnlp':
            df_recepie_vectors_lowdim = pd.read_csv(os.path.join(self.nlp_path,'doc2vec_features_lowdim.csv')).set_index('recepie_id')
            self.env = Environment(os.path.join(self.nlp_path,'train_logs_single_run/'))
            self.search_set_recepie_ids = np.array(self.env.get_recepie_ids())
            self.X_lowdim = df_recepie_vectors_lowdim.loc[self.search_set_recepie_ids].values
            self.axes_bounds = (np.min(self.X_lowdim, axis=0), np.max(self.X_lowdim, axis=0))
        assert self.search_space in ['nasbenchnlp','nasbench101','nasbench201','ResNeXt-A','ResNeXt-B','ResNet'], \
        "search space is not supported!"
    def random_spec(self):
        search_space = self.search_space
        if search_space == "nasbench101":
            spec = nb101_generator.random_spec()
            return [spec.original_matrix,spec.original_ops]
        elif search_space == "nasbench201":
            spec = []
            for i in range(3):
                node = []
                for k in range(i+1):
                    node.append(random.choice([0,1,2,3,4]))
                spec.append(node)
            return spec
        elif search_space == "nasbenchnlp":
            arch_id = np.random.randint(len(self.search_set_recepie_ids))
            return arch_id,self.env._logs[arch_id]
        else:
            return random.randint(0,len(self.NDS)-1)
    def mutate_spec(self,spec):
        search_space = self.search_space
        if search_space == "nasbench101":
            matrix,ops = spec
            spec = model_spec.ModelSpec(matrix,ops)
            new_spec = nb101_generator.mutate_spec(spec,self.mutate_ratio)
            return [new_spec.original_matrix,new_spec.original_ops]
        elif search_space == "nasbench201":
            new_spec = copy.deepcopy(spec)
            num_node = 5
            mutate_ratio = self.mutate_ratio/num_node
            for k in range(3):
                for i in range(k+1):
                    if random.random()<(mutate_ratio):
                        new_spec[k][i] = random.choice([0,1,2,3,4])
                    else:
                        continue
            return new_spec
        elif search_space == "nasbenchnlp":
            parent = spec[0]
            for std in [0.5, 1.0, 2.0, 4.0, 8.0]:
                e_new = mutate_embedded(self.X_lowdim[parent], std, self.axes_bounds)
                child = find_closest(self.X_lowdim, e_new)
                if child != parent:
                    break
            new_arch = self.env._logs[child]
            return [child,new_arch]
        elif search_space in ['ResNet','ResNeXt-A','ResNeXt-B']:
            query_list = random.sample(np.arange(len(self.NDS)).tolist(),self.neighbor)
            conf_A = self.NDS.get_network_config(spec)
            diffs = []
            for b in query_list:
                conf_B = self.NDS.get_network_config(b)
                diff = get_diff(conf_A,conf_B)
                diffs.append(diff)
            return query_list[np.argmin(diffs)]
        else:
            raise NotImplementedError
    