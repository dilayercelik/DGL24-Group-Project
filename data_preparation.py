# import relevant packages
import numpy as np
import pandas as pd
from torch import Tensor
from MatrixVectorizer import MatrixVectorizer

def multi_anti_vectorize(arr, vectorizer, matrix_size): 
    return np.array([vectorizer.anti_vectorize(v, matrix_size) for v in arr])

# added an input so that the function can be adjusted to everyone's path to the dataset
def load_data_tensor(path_to_datasets):
    # import data from .csv file
    hr_train_raw = pd.read_csv(path_to_datasets + '/hr_train.csv')
    lr_train_raw = pd.read_csv(path_to_datasets + '/lr_train.csv')
    lr_test_raw = pd.read_csv(path_to_datasets + '/lr_test.csv')

    # anti-vectorize 
    lr_n = 160
    hr_n = 268
    vectorizer = MatrixVectorizer()
    hr_train = multi_anti_vectorize(hr_train_raw.values, vectorizer, hr_n)
    lr_train = multi_anti_vectorize(lr_train_raw.values, vectorizer, lr_n)
    lr_test = multi_anti_vectorize(lr_test_raw.values, vectorizer, lr_n)

    # NOTE the order of return is low res train, low res test, high res train
    return Tensor(lr_train), Tensor(lr_test), Tensor(hr_train)

def split_train_data(data, test_ratio=0.2):
    n = data.size(0)
    split_idx = int(n * (1-test_ratio))
    train_data, val_data = data[:split_idx, :, :], data[split_idx:, : , :]
    
    return train_data, val_data

def generate_submission_file(prediction_tensors, filepath): 
    """
    Recommended file path is 'submission_files/xxxxxx.csv'
    """
    vectorizer = MatrixVectorizer()
    all_vectorized_arr = np.concatenate([vectorizer.vectorize(matrix) for matrix in prediction_tensors])
    df = pd.DataFrame({'ID': list(range(1, len(all_vectorized_arr)+1)), 'Predicted': all_vectorized_arr})
    df.to_csv(filepath, index=False)