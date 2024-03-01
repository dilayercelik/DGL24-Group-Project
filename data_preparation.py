# import relevant packages
import numpy as np
import pandas as pd
from torch import Tensor
from MatrixVectorizer import MatrixVectorizer

def multi_anti_vectorize(arr, vectorizer, matrix_size): 
    return np.array([vectorizer.anti_vectorize(v, matrix_size) for v in arr])


def load_data_tensor():
    # import data from .csv file
    hr_train_raw = pd.read_csv('dgl-icl/hr_train.csv')
    lr_train_raw = pd.read_csv('dgl-icl/lr_train.csv')
    lr_test_raw = pd.read_csv('dgl-icl/lr_test.csv')

    # anti-vectorize 
    lr_n = 160
    hr_n = 268
    vectorizer = MatrixVectorizer()
    hr_train = multi_anti_vectorize(hr_train_raw.values, vectorizer, hr_n)
    lr_train = multi_anti_vectorize(lr_train_raw.values, vectorizer, lr_n)
    lr_test = multi_anti_vectorize(lr_test_raw.values, vectorizer, lr_n)
    return Tensor(lr_train), Tensor(lr_test), Tensor(hr_train)