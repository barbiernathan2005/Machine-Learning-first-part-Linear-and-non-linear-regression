import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y=y+abs(y/2)
y = y.reshape(y.shape[0], 1)
m = len(y)

X = np.hstack((x, np.ones(x.shape)))
X=np.hstack((x**2,X))
theta = np.random.randn(3, 1)

def calcul_Fmodel(X, theta):
    return X.dot(theta)

def fonction_cout(X, y, theta):
    return 1/(2*m) * np.sum((calcul_Fmodel(X, theta) - y) ** 2)

def grad(X, y, theta):
    return (1/m) * X.T.dot(calcul_Fmodel(X, theta) - y)

def descente_gradiant(X, y, theta, learning_rate, n):
    cost_history = []  

    for i in range(0, n):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history.append(float(fonction_cout(X, y, theta)))

    return theta, cost_history 

def coef_determination(y,predictions):
    u=((y-predictions)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-(u/v)

theta_final, cost_history = descente_gradiant(X, y, theta, learning_rate=0.01, n=1000)

print("les valeurs de a et b sont", theta_final)

plt.plot(cost_history)
plt.xlabel("itérations")
plt.ylabel("fonction cout J(θ)")
plt.title("Convergence du coût")
plt.show()

predictions = calcul_Fmodel(X, theta_final)

print("le coeff de determination est:",coef_determination(y,predictions))
idx = np.argsort(x[:, 0])
plt.scatter(x, y)
plt.plot(x[idx], predictions[idx], color="r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Régression non linéaire")
plt.legend()
plt.show()