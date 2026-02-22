# Machine Learning Level 1 – Linear Regression From Python

## <span style="color:#2E86C1;">Project Overview</span>

This project represents my first structured step into Machine Learning.

Everything has been implemented entirely in **Python**, using NumPy for matrix computations and Matplotlib for visualization.  
The objective was not simply to use machine learning libraries, but to deeply understand how linear regression works mathematically and computationally.

Through this project, I focused on:

- Matrix-based model formulation
- Manual implementation of Gradient Descent
- Cost function minimization
- Manual computation of R²
- Polynomial feature engineering
- Comparison with Scikit-Learn

All models were first implemented from scratch before being compared with optimized tools. To get all the information, you need to compile the programs. I used a library that generates my datasets randomly, but you can use your own if you prefer.

---

## <span style="color:#2E86C1;">Mathematical Formulation</span>

### Model Hypothesis

The regression model is expressed in matrix form:

$$
\hat{y} = X\theta
$$

Where:

- $X \in \mathbb{R}^{m \times n}$ is the design matrix  
- $\theta \in \mathbb{R}^{n \times 1}$ is the parameter vector  
- $m$ is the number of samples  

The bias term is handled by adding a column of ones to $X$.

---

### Cost Function

The cost function implemented in Python is:

$$
J(\theta) = \frac{1}{2m}(X\theta - y)^T (X\theta - y)
$$

This formulation allows a fully vectorized implementation without looping over samples.

---

### Gradient

The gradient used in Gradient Descent is:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

---

### Update Rule

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

Where $\alpha$ is the learning rate.

All computations are performed using NumPy matrix operations.

---

## <span style="color:#28B463;">1. Univariate Linear Regression</span>

**File:** [MLregressionlinunidim.py](MLregressionlinunidim.py)

Model:

$$
\hat{y} = \theta_1 x + \theta_0
$$

In this implementation:

- Parameters are randomly initialized
- The cost is tracked at each iteration
- Convergence is visualized
- R² is computed manually:

$$
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
$$

### Cost Convergence

<img src="Figure_1.png" width="600">

### Regression Result

<img src="RL1.png" width="600">

---

## <span style="color:#28B463;">2. Multivariate Linear Regression (2 Features)</span>

**File:** [MLregressionlinmultidim3D.py](MLregressionlinmultidim3D.py)

Model:

$$
\hat{y} = \theta_1 x_1 + \theta_2 x_2 + \theta_0
$$

The design matrix is constructed as:

$$
X =
\begin{pmatrix}
x_{11} & x_{12} & 1 \\
x_{21} & x_{22} & 1 \\
\vdots & \vdots & \vdots
\end{pmatrix}
$$

This demonstrates:

- Fully vectorized gradient computation
- 3D visualization of the regression plane
- Manual performance evaluation

### 3D Regression Plane

<img src="RL2.png" width="600">

---

## <span style="color:#28B463;">3. Polynomial Regression (Feature Engineering)</span>

**File:** [MLregressionnonlin.py](MLregressionnonlin.py)

To model non-linear relationships, a polynomial feature is introduced:

$$
x' = x^2
$$

The new design matrix becomes:

$$
X =
\begin{pmatrix}
x_1^2 & x_1 & 1 \\
x_2^2 & x_2 & 1 \\
\vdots & \vdots & \vdots
\end{pmatrix}
$$

The model remains linear in parameters:

$$
\hat{y} = \theta_2 x^2 + \theta_1 x + \theta_0
$$

### Non-Linear Regression Result

<img src="RL3.png" width="600">

---

## <span style="color:#E67E22;">4. Comparison: Custom Gradient Descent vs Scikit-Learn</span>

**File:** [comparaisonRLvsRLopti.py](comparaisonRLvsRLopti.py)

In this part, I compared:

- My own Gradient Descent implementation
- `sklearn.linear_model.LinearRegression`

The comparison includes:

- Execution time
- Final cost
- Mean Squared Error
- R² score
- Stability across different learning rates

### Convergence for Different Learning Rates

<img src="comp2.png" width="600">

This clearly shows how sensitive Gradient Descent is to the choice of learning rate.

### Prediction Comparison

<img src="comparaison.png" width="600">

When properly tuned, both approaches converge to nearly identical predictions.

---

## <span style="color:#2E86C1;">Technologies Used</span>

- Python
- NumPy
- Matplotlib
- Scikit-Learn (used only for comparison)

---



