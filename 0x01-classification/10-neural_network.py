import numpy as np


class NeuralNetwork:
   

    def __init__(self, nx, nodes):
        

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W1 = np.random.randn(nodes, nx)

        self.__b2 = 0
        self.__A2 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)

    @property
    def b1(self):
        
        return self.__b1

    @property
    def A1(self):
       
        return self.__A1

    @property
    def W1(self):
       
        return self.__W1

    @property
    def b2(self):
        
        return self.__b2

    @property
    def A2(self):
       
        return self.__A2

    @property
    def W2(self):
        
        return self.__W2

    def forward_prop(self, X):
        

        XT = np.transpose(X)
        W1T = np.transpose(self.W1)
        A_1 = np.transpose(np.matmul(XT, W1T)) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-1 * A_1))
        
        A1T = np.transpose(self.A1)
        W2T = np.transpose(self.W2)
        A_2 = np.transpose(np.matmul(A1T, W2T)) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-1 * A_2))

        return self.__A1, self.__A2
