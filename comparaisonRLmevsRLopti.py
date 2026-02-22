import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from time import perf_counter


x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y = y.reshape(y.shape[0], 1)
m = len(y)


X = np.hstack((x, np.ones(x.shape)))

def calcul_Fmodel(X, theta):
    return X.dot(theta)

def fonction_cout(X, y, theta):
    return 1/(2*m) * np.sum((calcul_Fmodel(X, theta) - y) ** 2)

def grad(X, y, theta):
    return (1/m) * X.T.dot(calcul_Fmodel(X, theta) - y)


def coef_determination(y, predictions):
    u = ((y - predictions) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - (u / v)


def descente_gradiant(X, y, theta, learning_rate, n, tol=1e-12):
    couts = []
    t0 = perf_counter()
    cout_precedent = np.inf
    for i in range(n):
        theta = theta - learning_rate * grad(X, y, theta)
        cout = fonction_cout(X, y, theta)
        couts.append(cout)
        if abs(cout_precedent - cout) < tol:
            break
        cout_precedent = cout
        if not np.isfinite(cout) or cout > 1e50:
            break

    t1 = perf_counter()
    return theta, np.array(couts), (t1 - t0)


learning_rates = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
n = 1000

resultats = []  
for lr in learning_rates:
    theta = np.random.randn(2, 1)  
    theta_final, couts, temps = descente_gradiant(X, y, theta, learning_rate=lr, n=n)
    y_pred_gd = calcul_Fmodel(X, theta_final)
    resultats.append((lr, theta_final, y_pred_gd, couts, temps))


t0 = perf_counter()
lr_model = LinearRegression()
lr_model.fit(x, y.ravel())
y_pred_skl = lr_model.predict(x).reshape(-1, 1)
t1 = perf_counter()
temps_sklearn = t1 - t0


print("=== SKLEARN ===")
print(f"temps={temps_sklearn:.6f}s | pente={lr_model.coef_[0]:.6f} | biais={lr_model.intercept_:.6f}")
print(f"MSE={mean_squared_error(y, y_pred_skl):.6f} | R2={r2_score(y, y_pred_skl):.6f}")
print(f"R2 (maison)={coef_determination(y, y_pred_skl):.6f}")

print("\n=== TES ESSAIS (GD) ===")
for (lr, theta_final, y_pred_gd, couts, temps) in resultats:
    ok = np.isfinite(couts[-1]) and couts[-1] < 1e50
    status = "OK" if ok else "DIVERGE"
    mse = mean_squared_error(y, y_pred_gd) if ok else float("nan")
    r2 = r2_score(y, y_pred_gd) if ok else float("nan")
    r2_maison = float(coef_determination(y, y_pred_gd)) if ok else float("nan")

    print(f"lr={lr:<6} | {status:<7} | itérations={len(couts):<5} | temps={temps:.6f}s | "
          f"J_fin={couts[-1]:.6e} | MSE={mse:.6f} | R2={r2:.6f} | R2(maison)={r2_maison:.6f}")

plt.figure()
for (lr, theta_final, y_pred_gd, couts, temps) in resultats:
    if np.isfinite(couts).all() and couts[-1] < 1e50:
        plt.plot(couts, label=f"lr={lr}")
    else:
        plt.plot(couts, label=f"lr={lr} (div)")

plt.xlabel("itérations")
plt.ylabel("fonction_cout J(θ)")
plt.title("Convergence du coût pour différents learning rates")
plt.legend()
plt.yscale("log")  
plt.show()


meilleurs = [(lr, theta_final, y_pred_gd, couts, temps) for (lr, theta_final, y_pred_gd, couts, temps) in resultats
            if np.isfinite(couts[-1]) and couts[-1] < 1e50]
meilleur_lr, meilleur_theta, meilleur_pred, meilleur_couts, meilleur_temps = min(meilleurs, key=lambda t: t[3][-1])

idx = np.argsort(x[:, 0])
x_sorted = x[idx]
X_sorted = X[idx]

plt.scatter(x, y, label="Données", alpha=0.7)
plt.plot(x_sorted, calcul_Fmodel(X_sorted, meilleur_theta), label=f"Mon modèle (GD) lr={meilleur_lr}")
plt.plot(x_sorted, y_pred_skl[idx], linestyle="--", label="Sklearn LinearRegression")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparaison des prédictions")
plt.legend()
plt.show()