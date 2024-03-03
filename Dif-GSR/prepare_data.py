import numpy as np
import torch
import pandas as pd
import scipy
from scipy.io import loadmat
import yaml
from utils import anti_vectorize,h_i
from sklearn.model_selection import KFold




def prepare_cross_validation_data(config):
    func_data = pd.read_csv(config["Data"]["functional_data"]) #EDITED Moritz
    morph_data = pd.read_csv(config["Data"]["morphological_data"]) #EDITED Moritz
    func_data = anti_vectorize(func_data, num_nodes=config["Diffusion"]["target_dim"], h=True) #EDITED Moritz
    morph_data = anti_vectorize(morph_data, num_nodes=config["Diffusion"]["source_dim"], h=False) #EDITED Moritz
    func_data = torch.from_numpy(func_data).float()
    morph_data = torch.from_numpy(morph_data).float()
    func_data[func_data < 0] = 0
    #morph_data = torch.nn.functional.normalize(morph_data, p=2, dim=0)
    morph_data[morph_data < 0] = 0
    kf = KFold(n_splits=3, shuffle=True, random_state=config["Seed"]["seed"])
    indices = np.arange(func_data.shape[0])
    data_dict = {}
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(indices)):
        data_dict[fold_idx] = ((func_data[train_indices], morph_data[train_indices]),(func_data[test_indices], morph_data[test_indices]))



    return data_dict




if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    prepare_cross_validation_data(config)






