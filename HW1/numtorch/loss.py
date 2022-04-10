import numpy as np

def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵loss
    """
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    gradient = y_probability - y_true
    return loss, gradient