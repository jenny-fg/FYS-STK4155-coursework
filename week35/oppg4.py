import numpy as np 

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
    X, y, test_size=0.2, random_state=42
)