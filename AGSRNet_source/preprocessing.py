import torch
import numpy as np
import os
import scipy.io
import torch.nn.functional as F

path = 'drive/My Drive/BRAIN_DATASET'
roi_str = 'ROI_FC.mat'


def pad_HR_adj(label, split):
    # Convert input to tensor if it's not already
    label = torch.tensor(label, dtype=torch.float32)
    # Apply padding
    label = F.pad(label, (split, split, split, split), "constant", 0)
    # Fill diagonal with 1
    label.fill_diagonal_(1)
    return label


def normalize_adj_torch(mx):
    device = torch.device('cuda')
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = mx.to(device)
    return mx


def unpad(data, split):

    idx_0 = data.shape[0]-split
    idx_1 = data.shape[1]-split
    train = data[split:idx_0, split:idx_1]
    return train


def extract_data(subject, session_str, parcellation_str, subjects_roi):
    folder_path = os.path.join(
        path, str(subject), session_str, parcellation_str)
    roi_data = scipy.io.loadmat(os.path.join(folder_path, roi_str))
    roi = torch.tensor(roi_data['r'], dtype=torch.float32)
    
    # Replacing NaN values
    roi[torch.isnan(roi)] = torch.tensor(1.0)
    
    # Taking the absolute values of the matrix
    roi = torch.abs(roi)
    
    if parcellation_str == 'shen_268':
        roi = roi.reshape(1, 268, 268)
    else:
        roi = roi.reshape(1, 160, 160)
    
    if subject == 25629:
        subjects_roi = roi
    else:
        # Concatenate along the first dimension
        subjects_roi = torch.cat((subjects_roi, roi), dim=0)

    return subjects_roi


def load_data(start_value, end_value):

    subjects_label = torch.zeros(1, 268, 268, dtype=torch.float32)
    subjects_adj = torch.zeros(1, 160, 160, dtype=torch.float32)

    for subject in range(start_value, end_value):
        subject_path = os.path.join(path, str(subject))

        if 'session_1' in os.listdir(subject_path):

            subjects_label = extract_data(
                subject, 'session_1', 'shen_268', subjects_label)
            subjects_adj = extract_data(
                subject, 'session_1', 'Dosenbach_160', subjects_adj)

    return subjects_adj, subjects_label


def data():
    subjects_adj, subjects_labels = load_data(25629, 25830)
    test_adj_1, test_labels_1 = load_data(25831, 25863)
    test_adj_2, test_labels_2 = load_data(30701, 30757)
    test_adj = torch.cat((test_adj_1, test_adj_2), dim=0)
    test_labels = torch.cat((test_labels_1, test_labels_2), dim=0)
    return subjects_adj, subjects_labels, test_adj, test_labels
