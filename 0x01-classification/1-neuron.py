#!/usr/bin/env python3

import numpy as np


class Neuron:
    def __init__(self, nx):
      
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be an integer")

        self.__b = 0
        self.__A = 0
        self.__W = np.random.randn(nx).reshape(1, nx)

    @property
    def b(self):
     
        return self.__b

    @property
    def A(self):
        
        return self.__A

    @property
    def W(self):
       
        return self.__W
