import numpy as np 

class fc_layer():
    def __init__(self, input_size, output_size, name):
        self.Weight = np.random.randn(input_size, output_size) / 1000
        self.Bias = np.random.randn(output_size)
        self.name = name

    def forward(self, last_input):
        """
        前向传播
        """
        self.last_input = last_input
        output = np.dot(self.last_input, self.Weight) + self.Bias
        return output
    def backward(self, next_gd, regular):
        """
        反向传播
        """
        N = self.last_input.shape[0]
        gradient = np.dot(next_gd, self.Weight.T)  
        dw = np.dot(self.last_input.T, next_gd)  
        db = np.sum(next_gd, axis=0)  
        self.dw = dw / N + regular * self.Weight
        self.db = db / N + regular * self.Bias
        return gradient

    def save(self, path):
        np.save(path + str(self.name) + '_Weight.npy', self.Weight)
        np.save(path + str(self.name) + '_Bias.npy', self.Bias)
    
    def load(self, path):
        self.Weight = np.load(path + str(self.name) + '_Weight.npy')
        self.Bias = np.load(path + str(self.name) + '_Bias.npy')
    
    def step(self, lr):
        self.Weight -= lr * self.dw
        self.Bias -= lr * self.db
    
    def zero_grad(self):
        self.dw = None
        self.db = None