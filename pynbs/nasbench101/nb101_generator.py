import numpy as np
from .model_spec import ModelSpec
import torch
import random
import copy

config = {'train_data_files': [],
 'valid_data_file': '',
 'test_data_file': '',
 'sample_data_file': '',
 'data_format': 'channels_last',
 'num_labels': 10,
 'module_vertices': 7,
 'max_edges': 9,
 'available_ops': ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
 'stem_filter_size': 128,
 'num_stacks': 3,
 'num_modules_per_stack': 3,
 'batch_size': 256,
 'train_epochs': 108,
 'train_seconds': 14400.0,
 'learning_rate': 0.1,
 'lr_decay_method': 'COSINE_BY_STEP',
 'momentum': 0.9,
 'weight_decay': 0.0001,
 'max_attempts': 5,
 'intermediate_evaluations': ['0.5'],
 'num_repeats': 3,
 'use_tpu': True,
 'tpu_iterations_per_loop': 100,
 'tpu_num_shards': 2}
class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""
def check_spec(config, model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
        raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)
    if num_vertices > config['module_vertices']:
        raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                             % (num_vertices, config['module_vertices']))

    if num_edges > config['max_edges']:
        raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                             % (num_edges, config['max_edges']))

    if model_spec.ops[0] != 'input':
        raise OutOfDomainError('first operation should be \'input\'')
    if model_spec.ops[-1] != 'output':
        raise OutOfDomainError('last operation should be \'output\'')
    for op in model_spec.ops[1:-1]:
        if op not in config['available_ops']:
            raise OutOfDomainError('unsupported op %s (available ops = %s)'
                           % (op, config['available_ops']))
def is_valid(config,model_spec):
    """Checks the validity of the model_spec.

    For the purposes of benchmarking, this does not increment the budget
    counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      True if model is within space.
    """
    try:
        check_spec(config,model_spec)
    except OutOfDomainError:
        return False
    return True

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

def random_spec():
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = ModelSpec(matrix=matrix, ops=ops)
        if is_valid(config,spec):
            return spec
        
def mutate_spec(old_spec, mutation_rate=1.0):
  """Computes a valid mutated spec from the old_spec."""
  while True:
    new_matrix = copy.deepcopy(old_spec.original_matrix)
    new_ops = copy.deepcopy(old_spec.original_ops)

    # In expectation, V edges flipped (note that most end up being pruned).
    edge_mutation_prob = mutation_rate / NUM_VERTICES
    for src in range(0, NUM_VERTICES - 1):
      for dst in range(src + 1, NUM_VERTICES):
        if random.random() < edge_mutation_prob:
          new_matrix[src, dst] = 1 - new_matrix[src, dst]
          
    # In expectation, one op is resampled.
    op_mutation_prob = mutation_rate / OP_SPOTS
    for ind in range(1, NUM_VERTICES - 1):
      if random.random() < op_mutation_prob:
        available = [o for o in config['available_ops'] if o != new_ops[ind]]
        new_ops[ind] = random.choice(available)
        
    new_spec = ModelSpec(new_matrix, new_ops)
    if is_valid(config, new_spec):
      return new_spec

def random_combination(iterable, sample_size):
  """Random selection from itertools.combinations(iterable, r)."""
  pool = tuple(iterable)
  n = len(pool)
  indices = sorted(random.sample(range(n), sample_size))
  return tuple(pool[i] for i in indices)