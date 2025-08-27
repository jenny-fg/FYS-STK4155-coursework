#oppgave 3a)

import numpy as np

n = 20
income = np.array([116., 161., 167., 118., 172., 163., 179., 173., 162., 116., 101., 
    176., 178., 172., 143., 135., 160., 101., 149., 125.])

children = np.array([5, 3, 0, 4, 5, 3, 0, 4, 4, 3, 3, 5, 1, 0, 2, 3, 2, 1, 5, 4])

spending = np.array([152., 141., 102., 136., 161., 129.,  99., 159., 160., 107.,  98., 
    164., 121.,  93., 112., 127., 117.,  69., 156., 131.])

X = np.zeros((n, 3))
X[:, 0] = 1
X[:, 1] = income
X[:, 2] = children

#print(X)


#oppgave 3b

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

beta = OLS_parameters(X, spending)

print(beta)