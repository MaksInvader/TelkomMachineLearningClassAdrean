# Chapter 6: Decision Trees - Summary

## Overview

Decision Trees are versatile Machine Learning algorithms capable of performing classification, regression, and multioutput tasks. They are powerful, intuitive, and serve as fundamental components of Random Forests. This chapter covers training, visualization, predictions, the CART algorithm, regularization, and limitations of Decision Trees.

## Training and Visualizing a Decision Tree

### Basic Training Example

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

### Visualization

```python
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
```

Convert the .dot file to an image:
```bash
$ dot -Tpng iris_tree.dot -o iris_tree.png
```

## Making Predictions

Decision Trees make predictions by traversing from the root node to a leaf node based on feature values:

1. **Root node** (depth 0): Checks if petal length < 2.45 cm
2. If true → left child (leaf node) → predicts Iris setosa
3. If false → right child → checks if petal width < 1.75 cm
   - If true → predicts Iris versicolor
   - If false → predicts Iris virginica

### Key Node Attributes

- **samples**: Number of training instances at the node
- **value**: Distribution of training instances across classes
- **gini**: Impurity measure (0 = pure node, all instances of same class)

### Gini Impurity

**Equation 6-1:**

$$G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2$$

Where $p_{i,k}$ is the ratio of class k instances in the $i^{th}$ node.

**Example calculation:** For depth-2 left node with 0, 49, and 5 instances:
$$G_i = 1 - (0/54)^2 - (49/54)^2 - (5/54)^2 ≈ 0.168$$

### Important Properties

- **No feature scaling required**: Decision Trees don't require feature scaling or centering
- **Binary trees**: Scikit-Learn's CART algorithm produces only binary trees (yes/no questions)
- **White box models**: Decisions are easy to interpret and explain

## Estimating Class Probabilities

Decision Trees estimate probabilities by returning the ratio of training instances of each class in the leaf node:

```python
>>> tree_clf.predict_proba([[5, 1.5]])
array([[0., 0.90740741, 0.09259259]])

>>> tree_clf.predict([[5, 1.5]])
array([1])
```

For a flower with petals 5 cm long and 1.5 cm wide:
- 0% Iris setosa (0/54)
- 90.7% Iris versicolor (49/54)
- 9.3% Iris virginica (5/54)

## The CART Training Algorithm

The **Classification and Regression Tree (CART)** algorithm trains Decision Trees by:

1. Splitting the training set using a feature $k$ and threshold $t_k$
2. Searching for the pair $(k, t_k)$ that produces the purest subsets

### CART Cost Function for Classification

**Equation 6-2:**

$$J(k, t_k) = \frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right}$$

Where:
- $G_{left/right}$ measures impurity of left/right subset
- $m_{left/right}$ is the number of instances in left/right subset

### Algorithm Characteristics

- **Greedy algorithm**: Searches for optimal split at each level without looking ahead
- **Not guaranteed optimal**: Finding the optimal tree is NP-Complete
- **Recursive splitting**: Continues until max_depth is reached or no split reduces impurity

### Computational Complexity

- **Prediction**: $O(\log_2(m))$ - very fast, independent of features
- **Training**: $O(n × m \log_2(m))$ - compares all features on all samples at each node
- **Presort optimization**: Can speed up small datasets (< few thousand instances)

## Gini Impurity vs Entropy

### Entropy Measure

**Equation 6-3:**

$$H_i = -\sum_{k=1, p_{i,k} ≠ 0}^{n} p_{i,k}\log_2(p_{i,k})$$

**Example:** Depth-2 left node entropy:
$$H_i = -(49/54)\log_2(49/54) - (5/54)\log_2(5/54) ≈ 0.445$$

### Comparison

- **Gini impurity**: Slightly faster to compute (good default)
- **Entropy**: Tends to produce slightly more balanced trees
- **Practical difference**: Usually minimal impact on results
- Set `criterion="entropy"` to use entropy instead of Gini

## Regularization Hyperparameters

Decision Trees are **nonparametric models** that can overfit if left unconstrained. Regularization restricts the tree's freedom during training.

### Key Hyperparameters

- **max_depth**: Maximum depth of the tree (default: None/unlimited)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node
- **min_weight_fraction_leaf**: Same as min_samples_leaf but as a fraction
- **max_leaf_nodes**: Maximum number of leaf nodes
- **max_features**: Maximum features to evaluate for splitting at each node

### Regularization Strategy

- **Increase** min_* hyperparameters → more regularization
- **Decrease** max_* hyperparameters → more regularization

### Pruning Alternative

Some algorithms use **pruning**: train without restrictions, then delete unnecessary nodes using statistical tests (e.g., χ² test with p-value threshold, typically 5%).

## Regression with Decision Trees

Decision Trees can perform regression by predicting values instead of classes.

### Training Example

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```

### Key Differences from Classification

- Each node predicts a **value** (average target value) instead of a class
- Uses **MSE** (Mean Squared Error) instead of Gini impurity
- Predictions are constant within each region

### CART Cost Function for Regression

**Equation 6-4:**

$$J(k, t_k) = \frac{m_{left}}{m}\text{MSE}_{left} + \frac{m_{right}}{m}\text{MSE}_{right}$$

Where:

$$\text{MSE}_{node} = \sum_{i \in node}(\hat{y}_{node} - y^{(i)})^2$$

$$\hat{y}_{node} = \frac{1}{m_{node}}\sum_{i \in node} y^{(i)}$$

### Regularization for Regression

Without regularization, regression trees easily overfit. Setting `min_samples_leaf=10` or similar constraints produces more reasonable models.

## Limitations and Instability

### Sensitivity to Data

Decision Trees are highly sensitive to:

1. **Training set rotation**: Prefer orthogonal decision boundaries (perpendicular to axes)
   - Solution: Use Principal Component Analysis (PCA) for better data orientation

2. **Small variations**: Removing a single instance can produce very different trees
   - Solution: Use Random Forests to average predictions over many trees

3. **Stochastic training**: Same data can produce different models
   - Solution: Set `random_state` hyperparameter for reproducibility

### Advantages

- Simple to understand and interpret
- Easy to use
- Versatile (classification and regression)
- Powerful
- No data preparation needed (no feature scaling)
- White box model (explainable predictions)

### Disadvantages

- Prone to overfitting
- Sensitive to training set rotation
- Unstable with small data variations
- Not guaranteed to find optimal solution

## Key Takeaways

1. Decision Trees split data recursively using feature thresholds
2. CART algorithm minimizes impurity (Gini or entropy) for classification, MSE for regression
3. Regularization is essential to prevent overfitting
4. Trees are interpretable but unstable
5. Random Forests (Chapter 7) address instability by ensemble averaging
6. No feature scaling required
7. Predictions are fast: $O(\log_2(m))$

---

**Note**: Decision Trees form the foundation for ensemble methods like Random Forests and Gradient Boosting, which significantly improve performance by combining multiple trees.

## Exercises

1. What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with one million instances?

2. Is a node's Gini impurity generally lower or greater than its parent's? Is it generally lower/greater, or always lower/greater?

3. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing `max_depth`?

4. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

5. If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances?

6. If your training set contains 100,000 instances, will setting `presort=True` speed up training?

7. Train and fine-tune a Decision Tree for the moons dataset by following these steps:
   - a. Use `make_moons(n_samples=10000, noise=0.4)` to generate a moons dataset.
   - b. Use `train_test_split()` to split the dataset into a training set and a test set.
   - c. Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.
   - d. Train it on the full training set using these hyperparameters, and measure your model's performance on the test set. You should get roughly 85% to 87% accuracy.

8. Grow a forest by following these steps:
   - a. Continuing the previous exercise, generate 1,000 subsets of the training set, each containing 100 instances selected randomly. Hint: you can use Scikit-Learn's `ShuffleSplit` class for this.
   - b. Train one Decision Tree on each subset, using the best hyperparameter values found in the previous exercise. Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets, these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy.
   - c. Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees, and keep only the most frequent prediction (you can use SciPy's `mode()` function for this). This approach gives you majority-vote predictions over the test set.
   - d. Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random Forest classifier!
