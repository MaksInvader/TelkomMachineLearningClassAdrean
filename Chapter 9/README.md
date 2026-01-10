# Chapter 9: Unsupervised Learning Techniques

## Overview

Unsupervised learning works with unlabeled data (input features X without labels y). According to Yann LeCun: "if intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing, and reinforcement learning would be the cherry."

This chapter covers three main unsupervised learning tasks:

1. **Clustering** - Grouping similar instances together
2. **Anomaly Detection** - Identifying abnormal instances
3. **Density Estimation** - Estimating probability density functions

## Clustering

Clustering identifies similar instances and assigns them to clusters (groups of similar instances). Unlike classification, clustering is unsupervised—no labels are provided during training.

### Applications of Clustering

- **Customer Segmentation** - Group customers by behavior for targeted marketing
- **Data Analysis** - Analyze each cluster separately for insights
- **Dimensionality Reduction** - Replace features with cluster affinities
- **Anomaly Detection** - Instances with low affinity to all clusters are anomalies
- **Semi-Supervised Learning** - Propagate labels within clusters
- **Search Engines** - Find similar images by clustering
- **Image Segmentation** - Group pixels by color or features

## K-Means

K-Means is a simple, fast algorithm for clustering blob-shaped datasets.

### Basic Usage

```python
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

# Access cluster labels
print(y_pred is kmeans.labels_)  # True

# View centroids
print(kmeans.cluster_centers_)

# Predict new instances
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(kmeans.predict(X_new))
```

### Hard vs Soft Clustering

- **Hard clustering**: Each instance assigned to one cluster (via `predict()`)
- **Soft clustering**: Each instance gets scores for all clusters (via `transform()`)

```python
# Get distances to all centroids (soft clustering)
distances = kmeans.transform(X_new)
```

### The K-Means Algorithm

1. Randomly initialize k centroids
2. **Label step**: Assign each instance to nearest centroid
3. **Update step**: Compute new centroids as mean of assigned instances
4. Repeat steps 2-3 until convergence

The algorithm is guaranteed to converge but may reach local optima depending on initialization.

**Complexity**: O(kmn) - linear with instances (m), clusters (k), and dimensions (n)

### Centroid Initialization

**K-Means++** (default method):
1. Choose first centroid uniformly at random
2. For each subsequent centroid, choose instance x^(i) with probability:
   - D(x^(i))² / Σⱼ D(x^(j))²
   - where D(x^(i)) is distance to nearest chosen centroid
3. Repeat until k centroids selected

```python
# Custom initialization
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)

# Force original random initialization
kmeans = KMeans(n_clusters=5, init="random")
```

### Performance Metrics

**Inertia**: Mean squared distance between instances and closest centroids

```python
print(kmeans.inertia_)
print(kmeans.score(X))  # Returns negative inertia
```

The model runs `n_init` times (default=10) and keeps the best solution (lowest inertia).

### Mini-Batch K-Means

For large datasets, use mini-batches for faster training:

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
```

Trade-off: 3-4x faster but slightly higher inertia, especially as k increases.

### Finding Optimal Number of Clusters

#### Elbow Method

Plot inertia vs k and look for the "elbow" where inertia decrease slows:

```python
# Inertia decreases as k increases, look for inflection point
```

#### Silhouette Score

Better metric than inertia. For each instance:
- a = mean distance to other instances in same cluster
- b = mean distance to instances in next closest cluster
- Silhouette coefficient = (b - a) / max(a, b)

Range: [-1, +1]
- +1: Well inside own cluster, far from others
- 0: Close to cluster boundary
- -1: Likely assigned to wrong cluster

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, kmeans.labels_)
print(score)
```

Silhouette diagrams show coefficients for all instances, sorted by cluster.

### Limitations of K-Means

- Requires specifying k in advance
- Needs multiple runs to avoid suboptimal solutions
- Struggles with:
  - Varying cluster sizes
  - Different densities
  - Non-spherical shapes (ellipsoidal clusters)

**Important**: Always scale features before K-Means to prevent stretched clusters.

## Applications of K-Means

### Image Segmentation

Cluster pixels by color to segment images:

```python
from matplotlib.image import imread

image = imread("ladybug.png")  # Shape: (533, 800, 3)

# Reshape to list of RGB colors
X = image.reshape(-1, 3)

# Cluster colors
kmeans = KMeans(n_clusters=8).fit(X)

# Replace each color with cluster mean
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

### Preprocessing for Classification

Use K-Means as dimensionality reduction before classification:

```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X_digits, y_digits = load_digits(return_X_y=True)

# Baseline: 96.9% accuracy
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# With K-Means preprocessing: 97.8% accuracy
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression()),
])
pipeline.fit(X_train, y_train)

# Find optimal k with GridSearch
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
# Best: k=99, accuracy=98.22%
```

### Semi-Supervised Learning

When labels are scarce, use clustering to maximize label utility:

```python
# 1. Cluster training data
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)

# 2. Find representative images (closest to centroids)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# 3. Manually label only these 50 images
y_representative_digits = np.array([4, 8, 0, 6, 8, 3, ...])

# Result: 92.2% accuracy with only 50 labels (vs 83.3% with random 50)

# 4. Label Propagation: spread labels to entire cluster
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

# Result: 93.3% accuracy

# 5. Partial Propagation: only label instances closest to centroids
percentile_closest = 20
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

# Result: 94.0% accuracy (99% label accuracy)
```

### Active Learning

Iterative process to improve models with expert labeling:

1. Train model on labeled data
2. Use model to predict on unlabeled instances
3. Select most uncertain predictions for expert to label
4. Repeat until performance plateaus

Strategies for selection:
- Lowest estimated probability (uncertainty sampling)
- Largest model change
- Largest validation error drop
- Instances where different models disagree

## DBSCAN

Density-Based Spatial Clustering identifies clusters of arbitrary shapes based on local density.

### Algorithm

- **ε (epsilon)**: Distance defining instance's neighborhood
- **min_samples**: Minimum instances in ε-neighborhood for core instance
- **Core instance**: Has ≥ min_samples neighbors (including itself)
- **Cluster**: All instances in neighborhood of core instances
- **Anomaly**: Non-core instance without core neighbors (label = -1)

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# Access results
print(dbscan.labels_)  # -1 indicates anomalies
print(len(dbscan.core_sample_indices_))  # Number of core instances
print(dbscan.components_)  # Core instances themselves
```

### Making Predictions

DBSCAN has no `predict()` method. Use a classifier on core instances:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

# Predict new instances
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
predictions = knn.predict(X_new)
probabilities = knn.predict_proba(X_new)

# Add maximum distance threshold for anomaly detection
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1  # Mark distant instances as anomalies
```

### Characteristics

- **Strengths**: Identifies any number of clusters with any shape, robust to outliers
- **Limitations**: Struggles when density varies significantly across clusters
- **Complexity**: ~O(m log m), but can require O(m²) memory if eps is large

## Other Clustering Algorithms

- **Agglomerative Clustering**: Bottom-up hierarchical clustering, builds tree of clusters
- **BIRCH**: For very large datasets with limited features (<20)
- **Mean-Shift**: Shifts circles toward density maxima, but O(m²) complexity
- **Affinity Propagation**: Voting system, O(m²) complexity
- **Spectral Clustering**: Uses similarity matrix and dimensionality reduction

## Gaussian Mixtures

Gaussian Mixture Models (GMM) assume data generated from mixture of Gaussian distributions.

### Probabilistic Model

For each instance:
1. Pick cluster j with probability ϕ^(j) (cluster weight)
2. Sample location from Gaussian: x^(i) ~ N(μ^(j), Σ^(j))

Where:
- ϕ^(j) = weight of cluster j
- μ^(j) = mean of cluster j
- Σ^(j) = covariance matrix of cluster j
- z^(i) = cluster assignment (latent variable)

### Basic Usage

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

# View learned parameters
print(gm.weights_)      # Cluster weights ϕ
print(gm.means_)        # Cluster means μ
print(gm.covariances_)  # Covariance matrices Σ

# Check convergence
print(gm.converged_)
print(gm.n_iter_)

# Hard clustering
labels = gm.predict(X)

# Soft clustering (probabilities)
probabilities = gm.predict_proba(X)

# Generate new samples
X_new, y_new = gm.sample(6)

# Density estimation (log of PDF)
log_density = gm.score_samples(X)
```

### Expectation-Maximization Algorithm

Similar to K-Means but with soft assignments:

1. **Initialization**: Randomly initialize cluster parameters
2. **Expectation step**: Compute probability each instance belongs to each cluster (responsibilities)
3. **Maximization step**: Update cluster parameters weighted by responsibilities
4. Repeat until convergence

**Important**: Set `n_init=10` (default is 1) to run multiple times and keep best solution.

### Covariance Type Constraints

Control cluster shapes to reduce parameters:

```python
# "full" (default): Any shape, size, orientation
gm = GaussianMixture(n_components=3, covariance_type="full")

# "spherical": Spherical clusters, different diameters
gm = GaussianMixture(n_components=3, covariance_type="spherical")

# "diag": Ellipsoidal, axes parallel to coordinates
gm = GaussianMixture(n_components=3, covariance_type="diag")

# "tied": All clusters share same shape/size/orientation
gm = GaussianMixture(n_components=3, covariance_type="tied")
```

**Complexity**:
- "spherical" or "diag": O(kmn)
- "tied" or "full": O(kmn² + kn³)

### Anomaly Detection

Instances in low-density regions are anomalies:

```python
# Flag bottom 4% as anomalies
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

**Note**: Too many outliers can bias the model. Consider:
- Iterative outlier removal and refitting
- Robust covariance estimation (EllipticEnvelope)

**Novelty vs Anomaly Detection**:
- Novelty: Training set is clean
- Anomaly: Training set may contain outliers

### Selecting Number of Clusters

Use information criteria that penalize complexity:

**BIC = log(m)p - 2log(L̂)**

**AIC = 2p - 2log(L̂)**

Where:
- m = number of instances
- p = number of parameters
- L̂ = maximized likelihood

```python
# Compute BIC and AIC
print(gm.bic(X))
print(gm.aic(X))

# Lower is better - compare across different k values
```

BIC selects simpler models than AIC, especially on large datasets.

#### Likelihood Function

- **Probability**: Describes how plausible future outcome x is given parameters θ
- **Likelihood**: Describes how plausible parameters θ are given observed x

**Maximum Likelihood Estimation (MLE)**: Find θ that maximizes L(θ|x)

**Maximum A-Posteriori (MAP)**: Maximize L(θ|x)g(θ) using prior g(θ) - regularized MLE

Maximizing log-likelihood is equivalent and simpler (converts products to sums).

### Bayesian Gaussian Mixture

Automatically determines optimal number of clusters:

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)

# Unnecessary clusters get weights near 0
print(np.round(bgm.weights_, 2))
# array([0.4, 0.21, 0.4, 0., 0., 0., 0., 0., 0., 0.])
```

#### How It Works

Cluster parameters treated as latent random variables with prior distributions:

- **Weight concentration prior**: Controls expected number of clusters
  - Low (e.g., 0.01): Expects few clusters
  - High (e.g., 10000): Expects many clusters

#### Variational Inference

Bayes' theorem: **p(z|X) = p(X|z)p(z) / p(X)**

Problem: p(X) is intractable (requires integrating over all z)

Solution: Approximate p(z|X) with simpler distribution q(z; λ):
1. Choose family of distributions q(z; λ)
2. Minimize KL divergence from q to p(z|X)
3. Equivalent to maximizing ELBO (Evidence Lower Bound)

**KL(q||p) = log p(X) - ELBO**

**ELBO = E_q[log p(z,X)] - E_q[log q(z)]**

**Black Box Stochastic Variational Inference (BBSVI)**: Use gradient ascent on ELBO with samples from q - works with any differentiable model (enables Bayesian Deep Learning).

### Limitations

GMMs work well for ellipsoidal clusters but fail on arbitrary shapes (e.g., moons dataset).

## Other Anomaly/Novelty Detection Algorithms

- **PCA**: Compare reconstruction errors (anomalies have higher error)
- **Fast-MCD (EllipticEnvelope)**: Robust covariance estimation for outlier detection
- **Isolation Forest**: Random forest that isolates anomalies in fewer splits
- **Local Outlier Factor (LOF)**: Compares local density to neighbors' densities
- **One-Class SVM**: Separates instances from origin in high-dimensional space

## Key Takeaways

1. **Clustering** groups similar instances without labels
2. **K-Means**: Fast, simple, works for blob-shaped clusters
3. **DBSCAN**: Handles arbitrary shapes based on density
4. **Gaussian Mixtures**: Probabilistic model for ellipsoidal clusters
5. **Semi-supervised learning**: Leverage clustering to maximize label utility
6. **Anomaly detection**: Identify low-density or unusual instances
7. Always **scale features** before clustering
8. Use **silhouette score** or **information criteria** to select cluster count

## Code Summary

```python
# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(X)
labels = kmeans.predict(X_new)

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)

# Gaussian Mixture
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
densities = gm.score_samples(X)

# Bayesian Gaussian Mixture
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)
```
