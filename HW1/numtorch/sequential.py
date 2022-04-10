import numpy as np

class optimizer():
    def __init__(self, layers, lr, regular = 0.01):
        self.layers = layers
        self.length = len(layers)
        self.lr = lr
        self.regular = regular
    
    def backward(self, loss):
        for i in range(self.length-1, -1, -1):
            loss = self.layers[i].backward(loss, self.regular)
    
    def step(self):
        for i in range(self.length):
            x = self.layers[i].step(self.lr)
        return x
    
    def zero_grad(self):
        for i in range(self.length):
            self.layers[i].zero_grad()
        

class Sequential():
    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers)

    def forward(self, x):
        for i in range(self.length):
            x = self.layers[i].forward(x)
        return x
    
    def save(self, path):
        for i in range(self.length):
            try:
                self.layers[i].save(path)
            except:
                pass
    
    def load(self, path):        
        for i in range(self.length):
            try:
                self.layers[i].load(path)
            except:
                pass
    