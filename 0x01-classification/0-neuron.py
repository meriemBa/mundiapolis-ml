import numpy as np


class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.b = 0
        self.A = 0
        self.W = np.random.randn(nx).reshape(1, nx)
        
