# Chapter 8: Dimensionality Reduction

## Overview

Machine Learning problems often involve thousands or millions of features per training instance, leading to slow training and difficulty finding good solutions - a problem known as the **curse of dimensionality**. Dimensionality reduction addresses this by reducing features while preserving essential information.

## The Curse of Dimensionality

### Key Challenges

High-dimensional spaces behave counterintuitively:

- **Sparse Data**: In a 10,000-dimensional unit hypercube, over 99.999999% of points are very close to the border
- **Distance Paradox**: Average distance between random points grows dramatically with dimensions
  - 2D unit square: ~0.52
  - 3D unit cube: ~0.66
  - 1,000,000D hypercube: ~408.25 (roughly √1,000,000/6)
- **Overfitting Risk**: Training instances become far apart, making predictions less reliable

### Training Data Requirements

To reach sufficient density, training instances needed grow exponentially with dimensions. With 100 features, you'd need more instances than atoms in the observable universe for instances to be within 0.1 of each other on average.

## Benefits of Dimensionality Reduction

1. **Speed up training** - Smaller datasets train faster
2. **Data visualization** - Reduce to 2D or 3D for plotting and pattern detection
3. **Noise filtering** - May remove unnecessary details (though generally won't improve performance, just speed)

**Important**: Always try training with original data first before applying dimensionality reduction, as it adds pipeline complexity and may slightly reduce performance.

## Main Approaches

### 1. Projection

Training instances often lie within a much lower-dimensional subspace. Projection maps instances perpendicularly onto this subspace.

**Example**: A 3D dataset lying close to a 2D plane can be projected onto that plane, reducing from 3D to 2D.

**Limitation**: Doesn't work well for twisted manifolds (like the Swiss roll dataset), where simple projection squashes different layers together.

### 2. Manifold Learning

A **d-dimensional manifold** is a part of an n-dimensional space (where d < n) that locally resembles a d-dimensional hyperplane.

**Manifold Assumption**: Most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This is often empirically observed (e.g., MNIST handwritten digits have constrained features compared to random images).

**Implicit Assumption**: Tasks become simpler when expressed in the manifold's lower-dimensional space. However, this doesn't always hold - sometimes decision boundaries are simpler in the original space.

## Principal Component Analysis (PCA)

PCA is the most popular dimensionality reduction algorithm. It identifies the hyperplane closest to the data and projects onto it.

### How PCA Works

1. **Identify principal components**: Axes that account for the largest amounts of variance
   - 1st PC: Axis with maximum variance
   - 2nd PC: Orthogonal axis with maximum remaining variance
   - Continue for all dimensions

2. **Project data**: Map to lower dimensions using the first d principal components

### Preserving Variance

PCA selects the axis that preserves maximum variance, which:
- Loses less information than other projections
- Minimizes mean squared distance between original data and projection

### Implementation with NumPy

PCA uses **Singular Value Decomposition (SVD)** to decompose the training set matrix **X** into three matrices **U Σ V**^⊺.

**Equation 8-1: Principal components matrix**

$$\mathbf{V} = \begin{pmatrix}
| & | & & | \\
\mathbf{c}_1 & \mathbf{c}_2 & \cdots & \mathbf{c}_n \\
| & | & & |
\end{pmatrix}$$

Where **V** contains the unit vectors that define all principal components.

```python
# Center the data
X_centered = X - X.mean(axis=0)

# Perform SVD to get principal components
U, s, Vt = np.linalg.svd(X_centered)

# Extract first two principal components
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
```

**Projecting down to d dimensions**

**Equation 8-2: Projecting the training set down to d dimensions**

$$\mathbf{X}_{d\text{-proj}} = \mathbf{X}\mathbf{W}_d$$

Where **W**_d is the matrix containing the first d columns of **V**.

```python
# Project down to 2D
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
```

**Note**: PCA assumes data is centered around the origin. Scikit-Learn handles this automatically.

### Using Scikit-Learn

```python
from sklearn.decomposition import PCA

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

# Access principal components
# First PC: pca.components_.T[:, 0]
```

### Explained Variance Ratio

Shows proportion of variance along each principal component:

```python
>>> pca.explained_variance_ratio_
array([0.84248607, 0.14631839])
```

This means 84.2% of variance lies along the 1st PC, 14.6% along the 2nd PC, leaving <1.2% for the 3rd PC.

### Choosing Number of Dimensions

**Method 1: Compute cumulative variance**

```python
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```

**Method 2: Set target variance directly**

```python
pca = PCA(n_components=0.95)  # Preserve 95% variance
X_reduced = pca.fit_transform(X_train)
```

**Method 3: Plot explained variance** and look for an "elbow" where variance stops growing fast.

### PCA for Compression

Applying PCA to MNIST with 95% variance preservation:
- Original: 784 features
- Reduced: ~154 features (~20% of original size)
- Significantly speeds up classification algorithms

**Reconstruction**:

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)
```

**Equation 8-3: PCA inverse transformation, back to the original number of dimensions**

$$\mathbf{X}_{\text{recovered}} = \mathbf{X}_{d\text{-proj}}\mathbf{W}_d^{\top}$$

**Reconstruction error**: Mean squared distance between original and reconstructed data (some information is lost).

### Randomized PCA

Uses a stochastic algorithm for faster approximation:

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)
```

**Complexity**: O(m × d²) + O(d³) instead of O(m × n²) + O(n³)

**Default behavior**: Scikit-Learn automatically uses randomized PCA if:
- m or n > 500
- d < 80% of m or n

### Incremental PCA

For datasets too large to fit in memory, IPCA processes mini-batches:

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

**Using memmap for large files**:

```python
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```

## Kernel PCA

Applies the kernel trick to PCA, enabling complex nonlinear projections. Good at:
- Preserving clusters after projection
- Unrolling twisted manifolds

```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

### Selecting Kernel and Hyperparameters

**Method 1: Grid search with supervised task**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

>>> print(grid_search.best_params_)
{'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}
```

**Method 2: Minimize reconstruction pre-image error (unsupervised)**

```python
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# Compute reconstruction error
>>> from sklearn.metrics import mean_squared_error
>>> mean_squared_error(X, X_preimage)
32.786308795766132
```

The reconstruction involves finding a point in original space that maps close to the reconstructed point (the "pre-image").

## Locally Linear Embedding (LLE)

A Manifold Learning technique that doesn't rely on projections. Works by:
1. Measuring how each instance linearly relates to its closest neighbors
2. Finding a low-dimensional representation preserving these local relationships

Particularly good at unrolling twisted manifolds with limited noise.

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```

### LLE Algorithm

**Step 1: Model local relationships**

For each instance **x**^(i), find weights w_i,j that reconstruct it from k nearest neighbors:

**Equation 8-4: LLE step one: linearly modeling local relationships**

$$\hat{\mathbf{W}} = \underset{\mathbf{W}}{\arg\min} \sum_{i=1}^{m} \left(\mathbf{x}^{(i)} - \sum_{j=1}^{m} w_{i,j}\mathbf{x}^{(j)}\right)^2$$

**Subject to**:
- $w_{i,j} = 0$ if **x**^(j) is not one of the k nearest neighbors of **x**^(i)
- $\sum_{j=1}^{m} w_{i,j} = 1$ for $i = 1, 2, \cdots, m$

After this step, the weight matrix **W** (containing the weights $\hat{w}_{i,j}$) encodes the local linear relationships between training instances.

**Step 2: Reduce dimensionality**

Map instances to d-dimensional space while preserving relationships:

**Equation 8-5: LLE step two: reducing dimensionality while preserving relationships**

$$\mathbf{Z} = \underset{\mathbf{Z}}{\arg\min} \sum_{i=1}^{m} \left(\mathbf{z}^{(i)} - \sum_{j=1}^{m} w_{i,j}\mathbf{z}^{(j)}\right)^2$$

Where **z**^(i) is the image of **x**^(i) in d-dimensional space, and **Z** is the matrix containing all **z**^(i).

**Complexity**: O(m log(m)n log(k)) + O(mnk³) + O(dm²)

The m² term makes it scale poorly to very large datasets.

## Other Dimensionality Reduction Techniques

### Random Projections
- Projects data using random linear projection
- Surprisingly likely to preserve distances well (Johnson-Lindenstrauss lemma)
- Quality depends on number of instances and target dimensionality, not initial dimensionality

### Multidimensional Scaling (MDS)
- Reduces dimensionality while preserving distances between instances

### Isomap
- Creates a graph connecting each instance to nearest neighbors
- Preserves geodesic distances (shortest path in graph)

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Keeps similar instances close and dissimilar instances apart
- Mostly used for visualization of clusters in high-dimensional space

### Linear Discriminant Analysis (LDA)
- Classification algorithm that learns most discriminative axes between classes
- Projects onto hyperplane keeping classes far apart
- Good preprocessing before other classifiers (e.g., SVM)

## Key Takeaways

1. **Curse of dimensionality** makes high-dimensional problems computationally expensive and prone to overfitting
2. **Always try original data first** before applying dimensionality reduction
3. **PCA** is the most popular method - fast, simple, and effective for linear relationships
4. **Kernel PCA** handles nonlinear relationships using the kernel trick
5. **LLE** excels at unrolling twisted manifolds by preserving local relationships
6. **Choose dimensions** based on explained variance ratio or task performance
7. **Visualization** benefits from reducing to 2D or 3D using t-SNE or other methods
8. Dimensionality reduction is often a **preprocessing step** for supervised learning tasks

## Warning About Instability

PCA finds zero-centered unit vectors pointing in PC directions. Since opposing unit vectors lie on the same axis, the direction is not stable - small perturbations may flip directions. However, they generally remain on the same axes, and the planes they define stay consistent.

---

## Exercises

1. **What are the main motivations for reducing a dataset's dimensionality? What are the main drawbacks?**

2. **What is the curse of dimensionality?**

3. **Once a dataset's dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?**

4. **Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?**

5. **Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?**

6. **In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?**

7. **How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?**

8. **Does it make any sense to chain two different dimensionality reduction algorithms?**

9. **Load the MNIST dataset (introduced in Chapter 3) and split it into a training set and a test set (take the first 60,000 instances for training, and the remaining 10,000 for testing). Train a Random Forest classifier on the dataset and time how long it takes, then evaluate the resulting model on the test set. Next, use PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%. Train a new Random Forest classifier on the reduced dataset and see how long it takes. Was training much faster? Next, evaluate the classifier on the test set. How does it compare to the previous classifier?**

10. **Use t-SNE to reduce the MNIST dataset down to two dimensions and plot the result using Matplotlib. You can use a scatterplot using 10 different colors to represent each image's target class. Alternatively, you can replace each dot in the scatterplot with the corresponding instance's class (a digit from 0 to 9), or even plot scaled-down versions of the digit images themselves (if you plot all digits, the visualization will be too cluttered, so you should either draw a random sample or plot an instance only if no other instance has already been plotted at a close distance). You should get a nice visualization with well-separated clusters of digits. Try using other dimensionality reduction algorithms such as PCA, LLE, or MDS and compare the resulting visualizations.**
