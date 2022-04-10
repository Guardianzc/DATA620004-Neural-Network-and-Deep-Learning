import numpy as np 

class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        """
        前向传播
        """
        self.output = np.maximum(0, input)
        return self.output
    
    def backward(self, next_gd, regular):
        gradient = np.where(np.greater(self.output, 0), next_gd, 0)
        return gradient

    def step(self, lr):
        pass

    def zero_grad(self):
        pass