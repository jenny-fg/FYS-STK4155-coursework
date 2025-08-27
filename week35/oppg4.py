import numpy as np 
from sklearn.model_selection import train_test_split

n = 100
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1)

def polynomial_features(x, p):
    n = len(x)
    X = np.zeros((n, p + 1))
    for i in range(p+1):
        X[:, i] = x**i
    return X

X = polynomial_features(x, 5)

#oppgave 4b

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

beta = OLS_parameters(X, y)

#print(beta)

#oppgave 4c

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=73)

#oppgave 4d

beta_train = OLS_parameters(X_train, y_train)

y_predikasjon_train = X_train @ beta_train
y_predikasjon_test  = X_test @ beta_train

mse_train = np.mean((y_train - y_predikasjon_train)**2)
mse_test  = np.mean((y_test - y_predikasjon_test)**2)

print("MSE train:", mse_train)
print("MSE test:", mse_test)

#oppgave 4e

