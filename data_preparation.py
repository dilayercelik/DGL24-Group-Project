# import relevant packages
import numpy as np
import pandas as pd
from torch import Tensor
from MatrixVectorizer import MatrixVectorizer
import matplotlib.pyplot as plt

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
    return df

def generate_histogram(test_pred_tensor, target=None):
    flattened_tensor = test_pred_tensor.flatten() 
    # Create the histogram for predictions in purple
    plt.hist(flattened_tensor, alpha=0.5, bins=50, density=True, color='purple', label='Prediction')  
    if target is not None:
        flattened_target = target.flatten()
        # Create the histogram for ground truth in yellow
        plt.hist(flattened_target, bins=50, alpha=0.5, color='orange', label='Ground Truth', density=True)
    plt.title('Histogram of Tensor Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend() # This will show the labels in the plot

    # Show the plot
    plt.show()

def generate_heatmap(single_tensor, save_path=None):
    """
    Produces heatmap of single tensor

    input:
    - sinlge_tensor: torch.tensor (single adjacency matrix)
    - save_path: string (optional) (string with file name)
    """
    assert len(single_tensor.shape) == 2, 'tensor dimensionality is greater than 2 - only pass one matrix'
    assert single_tensor.shape[0] == single_tensor.shape[1], 'tensor not square'
    plt.imshow(single_tensor, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def new_generate_submission_file(prediction_tensors, filepath): 
    """
    Recommended file path is 'submission_files/xxxxxx.csv'

    input:
    - prediction_tensor: torch.tensor should be of shape [N,268,268] in same order as lr_test.csv
    """
    vectorizer = MatrixVectorizer()
    vectorized = np.array([vectorizer.vectorize(prediction_tensors[i].numpy()) for i in range(prediction_tensors.shape[0])])
    print(vectorized.shape)
    meltedDF = vectorized.flatten()
    df = pd.DataFrame({'ID': list(range(1, len(meltedDF)+1)), 'Predicted': meltedDF})
    df.to_csv(filepath, index=False)
    return df

def alt_generate_submission_file(prediction_tensors, filepath): 
    """
    Recommended file path is 'submission_files/xxxxxx.csv'

    input:
    - prediction_tensor: torch.tensor should be of shape [N,268,268] in same order as lr_test.csv
    """

    data = {
        'ID': [],
        'Predicted': []
    }
    vectorizer = MatrixVectorizer()
    vectorized = np.array([vectorizer.vectorize(prediction_tensors[i]) for i in range(prediction_tensors.shape[0])])
    id = 1
    for flattend in vectorized:
        for value in flattend:
            data['ID'].append(id)
            data['Predicted'].append(value)
            id = id + 1
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return df
