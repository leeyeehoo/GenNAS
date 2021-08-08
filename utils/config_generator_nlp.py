import numpy as np
import random
from copy import deepcopy

global_config_nlp = {'batchsize': [16],
 'noise': ['gaussian', 'uniform'],
 'dimension': [16, 32, 48, 64, 80, 96],
 'feature': ['sin1d', 'sin2d', 'dot', 'dot_gaussian']}

def generate_config_nlp(global_config, output_size = 8):
    config = {}
    for item in global_config:
        if 'feature' not in item:
            if 'noise' in item:
                config['noise'] = {}
                config['noise']['type'] = np.random.choice(global_config[item])
                config['noise']['noise_level'] = np.random.choice(np.asarray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
            else:
                config[item] = np.random.choice(global_config[item])
        else:
            config[item] = {}
            for layer in range(2):
                config[item][str(layer)] = {}
                for feat in global_config[item]:
                    if np.random.random() > 0.5:
                        config[item][str(layer)][feat] = {}
                        config[item][str(layer)][feat]['level'] = np.random.choice(np.arange(0.5,3.,0.1))
                        if feat in ['sin1d','sin2d']:
                            start,end = np.sort([np.random.random()*0.5,np.random.random()*0.5])
                            config[item][str(layer)][feat]['range'] = np.random.uniform(start,end,10).tolist()
                            if np.random.random() > 0.5:
                                config[item][str(layer)][feat]['local'] = True
                            else:
                                config[item][str(layer)][feat]['local'] = True
                        if feat in ['dot','dot_gaussian']:
                            config[item][str(layer)][feat]['partial'] = np.random.random()
                            if np.random.random() > 0.5:
                                config[item][str(layer)][feat]['local'] = True
                            else:
                                config[item][str(layer)][feat]['local'] = True
                    else:
                        config[item][str(layer)][feat] = None
    return config
def mutate_config_nlp(global_config,config,p = 0.5 , output_size = 8):
    new_config = deepcopy(config)
    for item in global_config:
        if 'feature' not in item:
            if np.random.random() > p:
                if 'noise' in item:
                    new_config['noise']['type'] = np.random.choice(global_config[item])
                    new_config['noise']['noise_level'] = np.random.choice(np.asarray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
                else:
                    new_config[item] = np.random.choice(global_config[item])
#                 new_config[item] = np.random.choice(global_config[item])
        else:
            for layer in range(2):
                for feat in global_config[item]:
                    if np.random.random() < p:
                        pass
                    else:
                        if np.random.random() > 0.5:
                            new_config[item][str(layer)][feat] = {}
                            new_config[item][str(layer)][feat]['level'] = np.random.choice(np.arange(0.5,3,0.1))
                            if feat in ['sin1d','sin2d']:
                                start,end = np.sort([np.random.random()*0.5,np.random.random()*0.5])
                                new_config[item][str(layer)][feat]['range'] = np.random.uniform(start,end,10).tolist()
                                if np.random.random() > 0.5:
                                    new_config[item][str(layer)][feat]['local'] = True
                                else:
                                    new_config[item][str(layer)][feat]['local'] = True

                            if feat in ['dot','dot_gaussian']:
                                new_config[item][str(layer)][feat]['partial'] = np.random.random()
                                if np.random.random() > 0.5:
                                    new_config[item][str(layer)][feat]['local'] = True
                                else:
                                    new_config[item][str(layer)][feat]['local'] = True

                        else:
                            new_config[item][str(layer)][feat] = None
    return new_config