alexa@ubuntu-xenial:$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('12-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)
alexa@ubuntu-xenial:$ ./12-main.py
[[0 0 0 ... 0 0 0]]
0.7917984405648548
alexa@ubuntu-xenial:$
