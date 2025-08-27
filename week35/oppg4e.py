import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(73)

n = 100
x = np.linspace(-3, 3, n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1)

def polynomial_features(x, p):
    n = len(x)
    X = np.zeros((n, p + 1))
    for i in range(p+1):
        X[:, i] = x**i
    return X

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=73)
    return X_train, X_test, y_train, y_test

def MSE(X_train, X_test, y_train, y_test):
    beta_train = OLS_parameters(X_train, y_train)
    
    y_predikasjon_train = X_train @ beta_train
    y_predikasjon_test  = X_test @ beta_train
    
    mse_train = np.mean((y_train - y_predikasjon_train)**2)
    mse_test  = np.mean((y_test - y_predikasjon_test)**2)

    return mse_train, mse_test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=73)

degrees, mses_train, mses_test = [], [], []

for p in range(2, 11):
    X_train = polynomial_features(x_train, p)
    X_test  = polynomial_features(x_test,  p)

    beta = OLS_parameters(X_train, y_train)

    yhat_tr = X_train @ beta
    yhat_te = X_test  @ beta

    mses_train.append(np.mean((y_train - yhat_tr)**2))
    mses_test.append(np.mean((y_test  - yhat_te)**2))
    degrees.append(p)
    ...
    print(f"Degree {p}: MSE train = {mses_train[-1]:.4f}, MSE test = {mses_test[-1]:.4f}")



plt.figure()
plt.plot(degrees, mses_train, marker='o', label='Train MSE')
plt.plot(degrees, mses_test,  marker='o', label='Test MSE')
plt.xlabel('Polynomial degree')
plt.ylabel('MSE')
plt.xticks(degrees)       # grader som heltall på x-aksen
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# evt. lagre
# plt.savefig('mse_vs_degree.png', dpi=150)