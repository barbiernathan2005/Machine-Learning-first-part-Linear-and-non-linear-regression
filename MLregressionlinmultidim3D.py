import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D

x, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
y = y.reshape(y.shape[0], 1)
m = len(y)

X = np.hstack((x, np.ones((x.shape[0], 1))))
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

def coef_determination(y, predictions):
    u = ((y - predictions) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - (u / v)

theta_final, cost_history = descente_gradiant(X, y, theta, learning_rate=0.01, n=1000)

print("les valeurs de a et b sont", theta_final)

plt.plot(cost_history)
plt.xlabel("itérations")
plt.ylabel("fonction cout J(θ)")
plt.title("Convergence du coût")
plt.show()

predictions = calcul_Fmodel(X, theta_final)

print("le coeff de determination est:", coef_determination(y, predictions))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x1 = x[:, 0]
x2 = x[:, 1]


ax.scatter(x1, x2, y[:, 0], label="Données", alpha=0.7)


x1_grid = np.linspace(x1.min(), x1.max(), 30)
x2_grid = np.linspace(x2.min(), x2.max(), 30)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)

X_grid = np.hstack((
    X1g.reshape(-1, 1),
    X2g.reshape(-1, 1),
    np.ones((X1g.size, 1))
))
Y_grid = calcul_Fmodel(X_grid, theta_final).reshape(X1g.shape)

ax.plot_surface(X1g, X2g, Y_grid, alpha=0.4, color='r')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Régression linéaire (2 features) - vue 3D")
ax.legend()
plt.show()