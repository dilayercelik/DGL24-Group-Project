'''
import numpy as np


def weight_variable_glorot(output_dim):

    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,(input_dim, output_dim))
    
    return initial
'''
import torch

def weight_variable_glorot(output_dim):
    input_dim = output_dim
    init_range = torch.sqrt(torch.tensor(6.0) / (input_dim + output_dim))
    # Use torch.rand to generate uniform distribution samples and scale them to the desired range
    initial = torch.rand(input_dim, output_dim) * (2 * init_range) - init_range
    
    return initial



   