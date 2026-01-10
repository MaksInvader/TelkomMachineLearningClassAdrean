# Chapter 4: Training Models - Summary

## Overview

This chapter explores how machine learning models actually work under the hood, focusing on training algorithms and optimization techniques. Understanding these fundamentals helps you select appropriate models, tune hyperparameters, debug issues, and build better systems.

## Topics Covered
- Linear Regression (closed-form and iterative solutions)
- Gradient Descent variants (Batch, Stochastic, Mini-batch)
- Polynomial Regression
- Learning curves and model evaluation
- Regularization techniques (Ridge, Lasso, Elastic Net)
- Logistic Regression
- Softmax Regression

---

## 1. Linear Regression

### Model Fundamentals

Linear Regression makes predictions by computing a weighted sum of input features plus a bias term:

**Equation 4-1: Linear Regression prediction**

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

Where:
- $\hat{y}$ = predicted value
- $n$ = number of features
- $x_i$ = $i$-th feature value
- $\theta_j$ = $j$-th model parameter ($\theta_0$ is bias, $\theta_1, ..., \theta_n$ are feature weights)

**Vectorized form (Equation 4-2):**

$$\hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta} \cdot \mathbf{x}$$

Where:
- $\boldsymbol{\theta}$ = parameter vector $[\theta_0, \theta_1, ..., \theta_n]$
- $\mathbf{x}$ = feature vector $[x_0, x_1, ..., x_n]$ where $x_0 = 1$
- $\boldsymbol{\theta} \cdot \mathbf{x}$ = dot product

### Training Objective

Training finds the $\boldsymbol{\theta}$ that minimizes the Mean Squared Error (MSE):

**Equation 4-3: MSE cost function**

$$\text{MSE}(\mathbf{X}, h_{\boldsymbol{\theta}}) = \frac{1}{m} \sum_{i=1}^{m} \left(\boldsymbol{\theta}^T \mathbf{x}^{(i)} - y^{(i)}\right)^2$$

---

## 2. The Normal Equation

### Closed-Form Solution

The Normal Equation directly computes the optimal $\boldsymbol{\theta}$:

**Equation 4-4:**

$$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Where:
- $\hat{\boldsymbol{\theta}}$ = optimal parameter vector
- $\mathbf{X}$ = training data matrix
- $\mathbf{y}$ = target values vector

### Implementation

```python
import numpy as np

# Generate sample data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add x₀ = 1 to each instance
X_b = np.c_[np.ones((100, 1)), X]

# Compute θ using Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
```

### Using Scikit-Learn

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
```

**Note:** LinearRegression uses SVD (Singular Value Decomposition) via `scipy.linalg.lstsq()`, which computes the pseudoinverse for better numerical stability.

### Computational Complexity

- **Normal Equation:** O(n²·⁴) to O(n³) where n = number of features
- **SVD approach:** O(n²)
- Both are **linear with respect to training instances** O(m)
- Becomes slow with many features (e.g., 100,000+)
- **Predictions:** Very fast, linear in instances and features

---

## 3. Gradient Descent

### Core Concept

Gradient Descent iteratively tweaks parameters to minimize the cost function by following the direction of steepest descent (negative gradient).

**Process:**
1. Start with random parameter values (random initialization)
2. Compute gradient of cost function
3. Take a step in direction of descending gradient
4. Repeat until convergence (gradient ≈ 0)

### Learning Rate (η)

The learning rate controls step size:

- **Too small:** Slow convergence, many iterations needed
- **Too large:** Algorithm may diverge, jumping across valleys
- **Good value:** Converges efficiently without oscillating

**Important:** Always scale features (e.g., using `StandardScaler`) before using Gradient Descent, as different feature scales create elongated cost function bowls that slow convergence.

### Cost Function Properties

For Linear Regression, MSE is:
- **Convex:** No local minima, only one global minimum
- **Continuous:** Slope never changes abruptly (Lipschitz continuous)
- **Result:** Gradient Descent guaranteed to approach global minimum

---

## 4. Batch Gradient Descent

### Gradient Computation

**Equation 4-5: Partial derivatives**

$$\frac{\partial}{\partial \theta_j} \text{MSE}(\boldsymbol{\theta}) = \frac{2}{m} \sum_{i=1}^{m} \left(\boldsymbol{\theta}^T \mathbf{x}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

**Equation 4-6: Gradient vector**

$$\nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

**Equation 4-7: Gradient Descent step**

$$\boldsymbol{\theta}^{(\text{next step})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})$$

### Implementation

```python
eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```

### Characteristics

- **Name:** Uses whole "batch" of training data at each step
- **Slow on large datasets:** Must process all instances each iteration
- **Scales well with features:** Much faster than Normal Equation for many features
- **Convergence:** O(1/ε) iterations to reach optimum within range ε

---

## 5. Stochastic Gradient Descent (SGD)

### Key Differences

- **Picks one random instance per step** instead of using entire dataset
- **Much faster:** Less data to process per iteration
- **Enables online learning:** Can train on huge datasets that don't fit in memory
- **Irregular path:** Bounces around due to randomness
- **Never settles:** Continues bouncing at minimum

### Learning Schedule

To converge to minimum, gradually reduce learning rate:

```python
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

**Critical:** Training instances must be **independent and identically distributed (IID)**. Shuffle the dataset to ensure this.

### Using Scikit-Learn

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_
```

---

## 6. Mini-batch Gradient Descent

### Concept

Computes gradients on small random sets of instances (mini-batches):

- **Middle ground** between Batch GD and Stochastic GD
- **Main advantage:** Performance boost from hardware optimization (especially GPUs)
- **Less erratic** than Stochastic GD
- **Harder to escape local minima** than Stochastic GD (not an issue for Linear Regression)

### Comparison Table

| Algorithm | Large m | Out-of-core | Large n | Hyperparams | Scaling Required | Scikit-Learn |
|-----------|---------|-------------|---------|-------------|------------------|--------------|
| Normal Equation | Fast | No | Slow | 0 | No | N/A |
| SVD | Fast | No | Slow | 0 | No | LinearRegression |
| Batch GD | Slow | No | Fast | 2 | Yes | SGDRegressor |
| Stochastic GD | Fast | Yes | Fast | ≥2 | Yes | SGDRegressor |
| Mini-batch GD | Fast | Yes | Fast | ≥2 | Yes | SGDRegressor |

**Note:** After training, all algorithms make predictions the same way with similar performance.

---

## 7. Polynomial Regression

### Concept

Fit nonlinear data by adding powers of features as new features.

### Example

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate nonlinear data
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Transform features to add polynomial terms
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# X[0] = [-0.75275929]
# X_poly[0] = [-0.75275929, 0.56664654]  # [x, x²]

# Fit linear model to polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# Output: (array([1.78134581]), array([[0.93366893, 0.56456263]]))
# Model: ŷ = 0.56x₁² + 0.93x₁ + 1.78
```

### Feature Combinations

For multiple features (e.g., a and b), `PolynomialFeatures(degree=3)` adds:
- Powers: a², a³, b², b³
- Combinations: ab, a²b, ab²

**Warning:** Features explode combinatorially: (n+d)! / (d!n!) features where n = original features, d = degree.

---

## 8. Learning Curves

### Purpose

Learning curves plot model performance on training and validation sets as a function of training set size, helping detect:
- **Underfitting:** Both curves plateau at high error, close together
- **Overfitting:** Large gap between curves

### Generating Learning Curves

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

# Example: Underfitting model
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
```

### Interpreting Curves

**Underfitting (high bias):**
- Both curves reach plateau
- Curves are close together
- Both have high error
- **Solution:** Use more complex model or better features

**Overfitting (high variance):**
- Large gap between curves
- Low training error
- High validation error
- **Solution:** Get more training data, reduce model complexity, or regularize

---

## 9. Bias/Variance Trade-off

### Three Error Components

**Generalization error = Bias + Variance + Irreducible error**

1. **Bias:** Error from wrong assumptions (e.g., assuming linear when data is quadratic)
   - High bias → underfitting
   
2. **Variance:** Error from sensitivity to small variations in training data
   - High variance → overfitting
   - High-degree polynomial models have high variance

3. **Irreducible error:** From data noisiness
   - Reduce by cleaning data, removing outliers

**Trade-off:** Increasing model complexity typically increases variance and reduces bias. Decreasing complexity does the opposite.

---

## 10. Regularized Linear Models

Regularization constrains model to reduce overfitting by keeping weights small.

### Ridge Regression (ℓ2 Regularization)

**Equation 4-8: Cost function**

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2$$

**Equation 4-9: Closed-form solution**

$$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{A})^{-1} \mathbf{X}^T \mathbf{y}$$

Where $\mathbf{A}$ is identity matrix with 0 in top-left (bias not regularized).

**Hyperparameter $\alpha$:**
- $\alpha = 0$: Regular Linear Regression
- Large $\alpha$: Weights → 0, flat line through data mean
- Controls bias/variance trade-off

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

**Using SGD:**
```python
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

**Important:** Always scale data before Ridge Regression!

### Lasso Regression (ℓ1 Regularization)

**Equation 4-10: Cost function**

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|$$

**Key feature:** Tends to eliminate weights of least important features (sets to zero), performing automatic **feature selection** and producing **sparse models**.

**Equation 4-11: Subgradient vector (for Gradient Descent)**

$$\mathbf{g}(\boldsymbol{\theta}, J) = \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) + \alpha \begin{pmatrix} \text{sign}(\theta_1) \\ \text{sign}(\theta_2) \\ \vdots \\ \text{sign}(\theta_n) \end{pmatrix}$$

where 

$$\text{sign}(\theta_i) = \begin{cases} -1 & \text{if } \theta_i < 0 \\ 0 & \text{if } \theta_i = 0 \\ +1 & \text{if } \theta_i > 0 \end{cases}$$

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```

**Note:** Gradually reduce learning rate to avoid bouncing around optimum.

### Elastic Net

**Equation 4-12: Cost function**

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r\alpha \sum_{i=1}^{n} |\theta_i| + \frac{(1-r)}{2}\alpha \sum_{i=1}^{n} \theta_i^2$$

Mix of Ridge and Lasso controlled by mix ratio $r$:
- $r = 0$: Ridge Regression
- $r = 1$: Lasso Regression

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

### When to Use Which

- **Ridge:** Good default, use when you suspect most features are useful
- **Lasso:** Use when you suspect only few features are useful (feature selection)
- **Elastic Net:** Preferred over Lasso generally (Lasso can behave erratically with many features or correlated features)

---

## 11. Early Stopping

### Concept

Stop training when validation error reaches minimum, preventing overfitting.

**Geoffrey Hinton:** Called it a "beautiful free lunch"

### Implementation

```python
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Prepare data
poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
```

**Key:** `warm_start=True` makes `fit()` continue training instead of restarting.

**For noisy curves (SGD/Mini-batch):** Stop after validation error stays above minimum for some time, then roll back to minimum.

---

## 12. Logistic Regression

### Purpose

Estimates probability that an instance belongs to a class (binary classifier).

### Estimating Probabilities

**Equation 4-13: Model probability**

$$\hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\mathbf{x}^T\boldsymbol{\theta})$$

**Equation 4-14: Logistic (sigmoid) function**

$$\sigma(t) = \frac{1}{1 + \exp(-t)}$$

Properties:
- Outputs values between 0 and 1
- S-shaped curve
- $\sigma(t) < 0.5$ when $t < 0$
- $\sigma(t) \geq 0.5$ when $t \geq 0$

**Equation 4-15: Prediction**

$$\hat{y} = \begin{cases} 0 & \text{if } \hat{p} < 0.5 \\ 1 & \text{if } \hat{p} \geq 0.5 \end{cases}$$

Decision boundary is where $\mathbf{x}^T\boldsymbol{\theta} = 0$.

### Training and Cost Function

**Objective:** High probability for positive instances ($y=1$), low for negative ($y=0$).

**Equation 4-16: Cost for single instance**

$$c(\boldsymbol{\theta}) = \begin{cases} -\log(\hat{p}) & \text{if } y = 1 \\ -\log(1 - \hat{p}) & \text{if } y = 0 \end{cases}$$

**Equation 4-17: Log loss (cross entropy)**

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})\right]$$

**Equation 4-18: Partial derivatives**

$$\frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left(\sigma(\boldsymbol{\theta}^T \mathbf{x}^{(i)}) - y^{(i)}\right) x_j^{(i)}$$

Properties:
- No closed-form solution
- **Convex:** Gradient Descent guaranteed to find global minimum
- Similar gradient form to Linear Regression MSE

### Decision Boundaries

**Example: Iris Virginica Detection**

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

log_reg = LogisticRegression()
log_reg.fit(X, y)

# Predict probabilities
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

# Make predictions
log_reg.predict([[1.7], [1.5]])
# array([1, 0])
```

**Multi-feature example (petal length and width):**

```python
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(X, y)
```

Decision boundary is linear. Points on boundary have 50% probability for each class.

### Regularization

**Important:** Scikit-Learn's `LogisticRegression` adds ℓ2 penalty by default.

**Hyperparameter:** `C` (inverse of α)
- High C → less regularization
- Low C → more regularization

---

## 13. Softmax Regression (Multinomial Logistic Regression)

### Purpose

Generalization of Logistic Regression for multiple classes (multiclass classification).

**Constraint:** Mutually exclusive classes only (not multioutput).

### Computing Probabilities

**Equation 4-19: Score for class k**

$$s_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\theta}^{(k)}$$

Each class has its own parameter vector $\boldsymbol{\theta}^{(k)}$.

**Equation 4-20: Softmax function**

$$\hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}$$

Where:
- $K$ = number of classes
- $\mathbf{s}(\mathbf{x})$ = vector of all class scores

**Equation 4-21: Prediction**

$$\hat{y} = \underset{k}{\text{argmax}} \, \sigma(\mathbf{s}(\mathbf{x}))_k = \underset{k}{\text{argmax}} \, s_k(\mathbf{x}) = \underset{k}{\text{argmax}} \, (\boldsymbol{\theta}^{(k)})^T \mathbf{x}$$

Predicts class with highest score.

### Training: Cross Entropy

**Equation 4-22: Cost function**

$$J(\boldsymbol{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})$$

Where $y_k^{(i)}$ is target probability (usually 0 or 1).

**Note:** When $K=2$, equivalent to Logistic Regression log loss.

**Equation 4-23: Gradient for class k**

$$\nabla_{\boldsymbol{\theta}^{(k)}} J(\boldsymbol{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{p}_k^{(i)} - y_k^{(i)}\right) \mathbf{x}^{(i)}$$

### Implementation

```python
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

softmax_reg.predict([[5, 2]])
# array([2])

softmax_reg.predict_proba([[5, 2]])
# array([[6.38e-07, 5.75e-02, 9.43e-01]])
# Probabilities: [Iris-setosa, Iris-versicolor, Iris-virginica]
```

**Decision boundaries:** Linear between any two classes.

---

## Key Takeaways

### Model Selection Guidelines

1. **Linear Regression:**
   - Few features: Normal Equation or SVD
   - Many features or instances: Gradient Descent
   - Very large datasets: Stochastic or Mini-batch GD

2. **Regularization:**
   - Almost always use some regularization
   - Ridge: good default
   - Lasso/Elastic Net: when few features are useful
   - Elastic Net preferred over Lasso

3. **Polynomial Regression:**
   - Use for nonlinear data
   - Watch for overfitting with high degrees
   - Use learning curves to detect over/underfitting

4. **Classification:**
   - Binary: Logistic Regression
   - Multiclass (mutually exclusive): Softmax Regression
   - Scale features before training

### Critical Practices

- **Always scale features** when using Gradient Descent or regularization
- **Use learning curves** to diagnose bias/variance issues
- **Apply early stopping** to prevent overfitting
- **Shuffle data** for Stochastic GD to ensure IID instances
- **Reduce learning rate gradually** for SGD/Mini-batch GD convergence

### Computational Complexity Summary

- Normal Equation: O(n²·⁴) to O(n³), linear in m
- SVD: O(n²), linear in m
- Gradient Descent: Fast with many features, scales well with m
- Predictions: Always fast, O(n × m)

---

## Exercises

1. Which Linear Regression training algorithm can you use if you have a training set with millions of features?

2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?

3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?

4. Do all Gradient Descent algorithms lead to the same model, provided you let them run long enough?

5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?

6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?

7. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?

9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?

10. Why would you want to use:
    a. Ridge Regression instead of plain Linear Regression?
    b. Lasso instead of Ridge Regression?
    c. Elastic Net instead of Lasso?

11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?

12. Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).

---

## Mathematical Notation Quick Reference

- $\hat{y}$: predicted value
- $\boldsymbol{\theta}$: parameter vector or matrix
- $\alpha$: regularization strength (or learning rate $\eta$)
- $\eta$ (eta): learning rate
- $\sigma$: sigmoid/softmax function
- $\nabla$: gradient (vector of partial derivatives)
- $^T$: transpose
- $m$: number of training instances
- $n$: number of features
- $K$: number of classes
- $\epsilon$ (epsilon): tolerance for convergence
- $\mathbf{X}$: feature matrix (bold uppercase = matrix)
- $\mathbf{x}$: feature vector (bold lowercase = vector)
- $\boldsymbol{\Theta}$: parameter matrix for Softmax Regression
