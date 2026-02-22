#  Machine Learning Level 1 – Linear Regression (Matrix Implementation)

##  Project Objective

This project implements **Linear Regression from scratch using a fully vectorized matrix formulation in NumPy**.

The objective is to understand:

- Matrix-based model formulation
- Gradient Descent optimization
- Cost function minimization
- Coefficient of determination (R²)
- Feature engineering (polynomial features)
- Comparison with Scikit-Learn

All models are implemented manually before comparing with optimized libraries. The programs are launched to retrieve all the information, and the datasets are randomly generated using a library. You can use and implement your own datasets if you wish.

---

##  1️ Mathematical Formulation

### Model Hypothesis

$$
\hat{y} = X\theta
$$

Where:

- $X \in \mathbb{R}^{m \times n}$ is the design matrix  
- $\theta \in \mathbb{R}^{n \times 1}$ is the parameter vector  
- $m$ is the number of samples  

Bias is handled by adding a column of ones to $X$.

### Example (Univariate Case)

$$
X =
\begin{pmatrix}
x_1 & 1 \\
x_2 & 1 \\
\vdots & \vdots \\
x_m & 1
\end{pmatrix}
$$

### Cost Function

$$
J(\theta) = \frac{1}{2m}(X\theta - y)^T (X\theta - y)
$$

### Gradient

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

### Update Rule

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

---

##  2️ Univariate Linear Regression

**File:** `MLregressionlinunidim.py`

### Model

$$
\hat{y} = \theta_1 x + \theta_0
$$

### Coefficient of Determination

$$
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
$$

---

##  3️ Multivariate Linear Regression (2 Features)

**File:** `MLregressionlinmultidim3D.py`

### Model

$$
\hat{y} = \theta_1 x_1 + \theta_2 x_2 + \theta_0
$$

### Design Matrix

$$
X =
\begin{pmatrix}
x_{11} & x_{12} & 1 \\
x_{21} & x_{22} & 1 \\
\vdots & \vdots & \vdots \\
x_{m1} & x_{m2} & 1
\end{pmatrix}
$$

Fully vectorized gradient computation with 3D visualization.

---

##  4️ Polynomial Regression (Feature Engineering)

**File:** `MLregressionnonlin.py`

### Feature Transformation

$$
x' = x^2
$$

### New Design Matrix

$$
X =
\begin{pmatrix}
x_1^2 & x_1 & 1 \\
x_2^2 & x_2 & 1 \\
\vdots & \vdots & \vdots \\
x_m^2 & x_m & 1
\end{pmatrix}
$$

### Model

$$
\hat{y} = \theta_2 x^2 + \theta_1 x + \theta_0
$$

---

##  5️ Comparison with Scikit-Learn

**File:** `comparaisonRLvsRLopti.py`

Comparison between:

- Custom Gradient Descent
- `sklearn.linear_model.LinearRegression`

Evaluated on:

- Execution time
- Final cost
- MSE
- R²
- Convergence stability for different learning rates

---

## Technologies

- Python
- NumPy
- Matplotlib
- Scikit-Learn

---


