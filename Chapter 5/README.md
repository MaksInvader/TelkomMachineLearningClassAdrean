# Support Vector Machines (SVM) - Chapter Summary

## Overview

Support Vector Machines are powerful and versatile machine learning models capable of performing:
- Linear or nonlinear classification
- Regression
- Outlier detection

SVMs are particularly well-suited for classification of complex small- or medium-sized datasets.

## Linear SVM Classification

### Large Margin Classification

The fundamental concept behind SVMs is **large margin classification**. An SVM classifier not only separates classes but stays as far away from the closest training instances as possible, fitting the widest possible "street" between classes.

**Key Points:**
- The decision boundary is fully determined by instances on the edge of the street
- These instances are called **support vectors**
- Adding more training instances "off the street" doesn't affect the decision boundary

### Feature Scaling Importance

⚠️ **SVMs are sensitive to feature scales.** Always scale your features (e.g., using `StandardScaler`) before training an SVM model.

### Soft Margin Classification

**Hard Margin Classification:**
- Strictly requires all instances to be off the street and on the correct side
- Only works if data is linearly separable
- Sensitive to outliers

**Soft Margin Classification:**
- More flexible approach
- Balances between keeping the street as large as possible and limiting margin violations
- Controlled by the **C hyperparameter**:
  - **Low C**: More margin violations, larger margin (may generalize better)
  - **High C**: Fewer margin violations, smaller margin
  - If overfitting: reduce C

### Training a Linear SVM

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])

svm_clf.fit(X, y)
```

**Making Predictions:**

```python
>>> svm_clf.predict([[5.5, 1.7]])
array([1.])
```

**Important Note:** Unlike Logistic Regression, SVM classifiers do not output probabilities for each class.

### Alternative Approaches

**Using SVC with linear kernel:**
```python
SVC(kernel="linear", C=1)
```

**Using SGDClassifier:**
```python
SGDClassifier(loss="hinge", alpha=1/(m*C))
```
- Applies Stochastic Gradient Descent
- Doesn't converge as fast as LinearSVC
- Useful for online classification or huge datasets (out-of-core training)

### LinearSVC Best Practices

⚠️ **Important Configuration:**
- Set `loss="hinge"` (not the default)
- Set `dual=False` for better performance (unless more features than training instances)
- The class regularizes the bias term, so center the training set first (automatic with `StandardScaler`)

## Nonlinear SVM Classification

### Adding Polynomial Features

For nonlinearly separable datasets, you can add polynomial features to make them linearly separable.

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15)

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(X, y)
```

**Limitations:**
- Low polynomial degree: cannot deal with very complex datasets
- High polynomial degree: creates huge number of features, making model too slow

### Polynomial Kernel

The **kernel trick** makes it possible to get the same result as adding many polynomial features without actually adding them, avoiding combinatorial explosion.

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X, y)
```

**Hyperparameters:**
- **degree**: Polynomial degree
  - Overfitting: reduce degree
  - Underfitting: increase degree
- **coef0**: Controls influence of high-degree vs. low-degree polynomials

### Similarity Features

Another approach uses a **similarity function** to measure how much each instance resembles particular landmarks.

**Gaussian Radial Basis Function (RBF):**

$$\phi_\gamma(\mathbf{x}, \ell) = \exp(-\gamma \|\mathbf{x} - \ell\|^2)$$

- Bell-shaped function varying from 0 (far from landmark) to 1 (at landmark)
- Simple approach: create a landmark at every instance location
- Transforms m instances with n features into m instances with m features

### Gaussian RBF Kernel

The kernel trick enables using similarity features efficiently:

```python
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])

rbf_kernel_svm_clf.fit(X, y)
```

**Hyperparameters:**
- **gamma (γ)**: Acts like a regularization hyperparameter
  - **Increasing gamma**: Narrower bell curve, smaller range of influence, more irregular decision boundary
  - **Decreasing gamma**: Wider bell curve, larger range of influence, smoother decision boundary
  - Overfitting: reduce gamma
  - Underfitting: increase gamma
- **C**: Same as before (regularization)

### Other Kernels

- **String kernels**: For text documents or DNA sequences
- **Specialized kernels**: For specific data structures

### Kernel Selection Strategy

💡 **Rule of thumb:**
1. Always try **linear kernel** first (especially if training set is very large or has plenty of features)
   - `LinearSVC` is much faster than `SVC(kernel="linear")`
2. If training set not too large, try **Gaussian RBF kernel** (works well in most cases)
3. If spare time/computing power, experiment with other kernels using cross-validation and grid search
4. Use specialized kernels if available for your data structure

### Finding Hyperparameters

Use **grid search**:
1. Start with coarse grid search
2. Fine grid search around best values found
3. Understanding what each hyperparameter does helps search in the right space

## Computational Complexity

### LinearSVC
- Based on liblinear library
- Does **not** support kernel trick
- Scales almost linearly with training instances and features
- **Time complexity:** O(m × n)
- Controlled by tolerance hyperparameter ε (`tol` in Scikit-Learn)

### SVC
- Based on libsvm library
- **Supports kernel trick**
- **Time complexity:** O(m² × n) to O(m³ × n)
- Gets slow with large training sets (e.g., hundreds of thousands of instances)
- Perfect for complex small/medium training sets
- Scales well with number of features, especially sparse features

### Comparison Table

| Class | Time Complexity | Out-of-Core Support | Scaling Required | Kernel Trick |
|-------|----------------|---------------------|------------------|--------------|
| LinearSVC | O(m × n) | No | Yes | No |
| SGDClassifier | O(m × n) | Yes | Yes | No |
| SVC | O(m² × n) to O(m³ × n) | No | Yes | Yes |

## SVM Regression

SVMs also support regression by **reversing the objective**: instead of fitting the largest street between classes while limiting margin violations, SVM Regression fits as many instances as possible **on the street** while limiting margin violations (instances off the street).

**Width of street controlled by ε (epsilon) hyperparameter**

### Linear SVM Regression

```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
```

**Note:** The model is **ε-insensitive** – adding training instances within the margin doesn't affect predictions.

### Nonlinear SVM Regression

For nonlinear regression tasks, use a kernelized SVM model:

```python
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```

**Regularization:**
- Large C: Little regularization
- Small C: More regularization

**Class Equivalents:**
- `SVR` is regression equivalent of `SVC`
- `LinearSVR` is regression equivalent of `LinearSVC`
- Scaling characteristics are the same as classification versions

**Additional Capability:** SVMs can also be used for outlier detection.

## Under the Hood

### Notation Conventions

- **w**: Feature weights vector
- **b**: Bias term
- **No bias feature** added to input vectors (different from Chapter 4 convention)

### Decision Function and Predictions

Linear SVM classifier predicts class by computing:

$$\mathbf{w}^\top \mathbf{x} + b = w_1x_1 + \cdots + w_nx_n + b$$

**Prediction Rule:**

$$\hat{y} = \begin{cases} 0 & \text{if } \mathbf{w}^\top\mathbf{x} + b < 0 \\ 1 & \text{if } \mathbf{w}^\top\mathbf{x} + b \geq 0 \end{cases}$$

- Decision boundary: where decision function equals 0
- Parallel dashed lines (where function = ±1) form margin around decision boundary

### Training Objective

**Goal:** Minimize ∥w∥ to get large margin

**Why minimize ½w⊺w instead of ∥w∥?**
- ½∥w∥² has simple derivative (just w)
- ∥w∥ is not differentiable at w = 0
- Optimization algorithms work better on differentiable functions

**Hard Margin Objective:**

$$\begin{aligned}
\text{minimize}_{w,b} & \quad \frac{1}{2}\mathbf{w}^\top\mathbf{w} \\
\text{subject to} & \quad t^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1 \text{ for } i = 1, 2, \ldots, m
\end{aligned}$$

where t^(i) = -1 for negative instances, t^(i) = 1 for positive instances

**Soft Margin Objective:**

Introduces **slack variables** ζ^(i) ≥ 0 (zeta) measuring how much each instance violates the margin.

$$\begin{aligned}
\text{minimize}_{w,b,\zeta} & \quad \frac{1}{2}\mathbf{w}^\top\mathbf{w} + C\sum_{i=1}^m \zeta^{(i)} \\
\text{subject to} & \quad t^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1 - \zeta^{(i)} \text{ and } \zeta^{(i)} \geq 0
\end{aligned}$$

**C hyperparameter** defines trade-off between:
- Making slack variables small (reducing margin violations)
- Making ½w⊺w small (increasing margin)

### Quadratic Programming

Hard and soft margin problems are **Quadratic Programming (QP)** problems: convex quadratic optimization with linear constraints.

**General QP Problem:**

$$\begin{aligned}
\text{Minimize}_\mathbf{p} & \quad \frac{1}{2}\mathbf{p}^\top\mathbf{H}\mathbf{p} + \mathbf{f}^\top\mathbf{p} \\
\text{subject to} & \quad \mathbf{A}\mathbf{p} \leq \mathbf{b}
\end{aligned}$$

Many off-the-shelf QP solvers can solve these problems.

### The Dual Problem

Given a **primal problem**, you can express a related **dual problem**. Under certain conditions (met by SVMs), both have the same solution.

**Dual Form of Linear SVM Objective:**

$$\begin{aligned}
\text{minimize}_\alpha & \quad \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha^{(i)}\alpha^{(j)}t^{(i)}t^{(j)}{\mathbf{x}^{(i)}}^\top\mathbf{x}^{(j)} - \sum_{i=1}^m\alpha^{(i)} \\
\text{subject to} & \quad \alpha^{(i)} \geq 0 \text{ for } i = 1, 2, \ldots, m
\end{aligned}$$

**From Dual to Primal:**

$$\mathbf{w} = \sum_{i=1}^m \alpha^{(i)}t^{(i)}\mathbf{x}^{(i)}$$

$$b = \frac{1}{n_s}\sum_{i=1, \alpha^{(i)}>0}^m \left(t^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)}\right)$$

**Advantages:**
- Faster when number of instances < number of features
- Makes kernel trick possible (primal does not)

### Kernelized SVMs

#### The Kernel Trick

Consider second-degree polynomial transformation:

$$\phi\left(\mathbf{x}\right) = \phi\left(\begin{bmatrix}x_1\\x_2\end{bmatrix}\right) = \begin{bmatrix}x_1^2\\\sqrt{2}x_1x_2\\x_2^2\end{bmatrix}$$

**Key insight:** Dot product of transformed vectors equals square of dot product of original vectors:

$$\phi(\mathbf{a})^\top\phi(\mathbf{b}) = (\mathbf{a}^\top\mathbf{b})^2$$

So instead of transforming all instances and computing dot products, just replace dot product with its square!

**Kernel Definition:** Function K(a, b) capable of computing dot product φ(a)⊺φ(b) based only on original vectors a and b, without computing transformation φ.

#### Common Kernels

**Linear:**
$$K(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top\mathbf{b}$$

**Polynomial:**
$$K(\mathbf{a}, \mathbf{b}) = (\gamma\mathbf{a}^\top\mathbf{b} + r)^d$$

**Gaussian RBF:**
$$K(\mathbf{a}, \mathbf{b}) = \exp(-\gamma\|\mathbf{a} - \mathbf{b}\|^2)$$

**Sigmoid:**
$$K(\mathbf{a}, \mathbf{b}) = \tanh(\gamma\mathbf{a}^\top\mathbf{b} + r)$$

#### Mercer's Theorem

If function K(a,b) respects **Mercer's conditions** (continuous, symmetric, etc.), then there exists a function φ that maps a and b to another space such that K(a,b) = φ(a)⊺φ(b).

**Important:** Gaussian RBF kernel maps to infinite-dimensional space!

#### Making Predictions with Kernelized SVM

Since w would have same dimensions as φ(x^(i)) (possibly infinite), we can't compute it directly. Instead, substitute w formula into decision function:

$$h_{\mathbf{w},b}(\phi(\mathbf{x}^{(n)})) = \sum_{i=1, \alpha^{(i)}>0}^m \alpha^{(i)}t^{(i)}K(\mathbf{x}^{(i)}, \mathbf{x}^{(n)}) + b$$

**Key advantage:** Only compute with support vectors (where α^(i) ≠ 0), not all training instances.

**Bias term computation:**

$$b = \frac{1}{n_s}\sum_{i=1, \alpha^{(i)}>0}^m \left(t^{(i)} - \sum_{j=1, \alpha^{(j)}>0}^m \alpha^{(j)}t^{(j)}K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})\right)$$

## Online SVMs

**Online learning:** Learning incrementally as new instances arrive

### Linear SVM with Gradient Descent

Use Gradient Descent (e.g., `SGDClassifier`) to minimize cost function:

$$J(\mathbf{w}, b) = \frac{1}{2}\mathbf{w}^\top\mathbf{w} + C\sum_{i=1}^m \max(0, 1 - t^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b))$$

**Cost Function Components:**
1. **First term (½w⊺w)**: Pushes model toward small weight vector → larger margin
2. **Second term**: Total of all margin violations
   - Margin violation = 0 if instance off street and on correct side
   - Otherwise proportional to distance to correct side

**Limitation:** Gradient Descent converges much slower than QP-based methods

### Hinge Loss Function

The function max(0, 1 - t) is the **hinge loss function**:
- Equal to 0 when t ≥ 1
- Derivative = -1 if t < 1
- Derivative = 0 if t > 1
- Not differentiable at t = 1 (use any subderivative between -1 and 0)

### Kernelized Online SVMs

Possible to implement online kernelized SVMs (see research papers mentioned in text). For large-scale nonlinear problems, consider neural networks instead.

## Key Takeaways

1. **SVMs fit the widest possible street** between classes (large margin classification)
2. **Always scale features** before training SVMs
3. **C hyperparameter** controls regularization (lower C = more regularization)
4. **Linear kernel** should be tried first, especially for large datasets
5. **RBF kernel** works well in most cases for smaller datasets
6. **Gamma hyperparameter** in RBF kernel controls regularization
7. **Kernel trick** allows computing in high/infinite dimensions without explicit transformation
8. **SVMs work for regression** too (ε-insensitive regression)
9. Use **LinearSVC/LinearSVR** for linear problems (faster)
10. Use **SVC/SVR** for nonlinear problems (slower but supports kernels)

## Practice Exercises

1. What is the fundamental idea behind Support Vector Machines?
2. What is a support vector?
3. Why is it important to scale the inputs when using SVMs?
4. Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?
5. Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?
6. Say you've trained an SVM classifier with an RBF kernel, but it seems to underfit the training set. Should you increase or decrease γ (gamma)? What about C?
7. How should you set the QP parameters (H, f, A, and b) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?
8. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a SGDClassifier on the same dataset. See if you can get them to produce roughly the same model.
9. Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary classifiers, you will need to use one-versus-the-rest to classify all 10 digits. You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?
10. Train an SVM regressor on the California housing dataset.
