from dataclasses import replace
import numpy as np

def data_split(dataset, label, percent = 0.2):
    data_num = dataset.shape[0]
    size = int(data_num * percent)
    idx = np.random.choice(data_num, size, replace = False)
    train_idx = list(set(range(data_num)) - set(idx))
    valid_set, valid_label = dataset[idx], label[idx]
    train_set, train_label = dataset[train_idx], label[train_idx]
    return train_set, train_label, valid_set, valid_label

def random_batch(data, label, batch_size):
    train_num = data.shape[0]
    search_index = np.random.choice(train_num, batch_size)
    return data[search_index], label[search_index]