# Chapter 7: Ensemble Learning and Random Forests

## Overview

Ensemble Learning combines multiple predictors to create a more powerful model than any individual predictor. This approach leverages the "wisdom of the crowd" - aggregating predictions from multiple models often yields better results than using a single expert model.

## Core Concept

An **ensemble** is a group of predictors, and the technique of combining them is called **Ensemble Learning**. The resulting algorithm is an **Ensemble method**.

### Key Example: Random Forests
A Random Forest trains multiple Decision Trees on different random subsets of the training set, then predicts by aggregating their votes. Despite its simplicity, it's one of the most powerful ML algorithms available.

---

## 1. Voting Classifiers

### Hard Voting
Aggregates predictions from multiple classifiers and selects the class with the most votes.

**Why It Works:**
- Even if each classifier is a **weak learner** (slightly better than random guessing), the ensemble can be a **strong learner** (high accuracy)
- Requires sufficient number of diverse weak learners
- Based on the **law of large numbers**: with 1,000 classifiers at 51% accuracy each, the ensemble can achieve ~75% accuracy

**Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)
```

**Performance Example:**
```python
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Output:
# LogisticRegression 0.864
# RandomForestClassifier 0.896
# SVC 0.888
# VotingClassifier 0.904  # Best performance!
```

### Soft Voting
Predicts the class with the highest average probability across all classifiers. Often achieves higher performance than hard voting by giving more weight to confident predictions.

```python
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')  # Changed from 'hard' to 'soft'
```

**Note:** All classifiers must have a `predict_proba()` method. For SVC, set `probability=True`.

### Critical Success Factor
**Ensemble methods work best when predictors are as independent as possible.** One way to achieve this is using very different training algorithms, which increases the chance of making different types of errors.

---

## 2. Bagging and Pasting

Another approach to create diverse classifiers: use the same training algorithm but train on different random subsets of the training set.

### Definitions
- **Bagging** (Bootstrap Aggregating): Sampling **with replacement**
- **Pasting**: Sampling **without replacement**

Both allow training instances to be sampled multiple times across different predictors, but only bagging allows the same instance to be sampled multiple times for the same predictor.

### Benefits
- **Reduces variance** while maintaining similar bias
- **Highly parallelizable** - predictors can be trained simultaneously on different CPU cores or servers
- Predictions can also be made in parallel

### Implementation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=500,
    max_samples=100, 
    bootstrap=True,  # Set to False for pasting
    n_jobs=-1)       # Use all available CPU cores

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

**Note:** `BaggingClassifier` automatically performs soft voting if the base classifier has a `predict_proba()` method.

### Bagging vs. Pasting
- **Bagging** introduces more diversity (higher bias but lower variance)
- Generally results in better models, which is why it's preferred
- Use cross-validation to determine which works best for your specific case

### Out-of-Bag (OOB) Evaluation

With bagging, approximately 63% of training instances are sampled for each predictor. The remaining 37% are called **out-of-bag instances**.

**Key Advantage:** These instances can be used for evaluation without needing a separate validation set.

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=500,
    bootstrap=True, 
    n_jobs=-1, 
    oob_score=True)  # Enable OOB evaluation

bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)  # e.g., 0.901

# Verify on test set
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))  # e.g., 0.912
```

Access OOB decision function (class probabilities):
```python
print(bag_clf.oob_decision_function_)
# Output: array of class probabilities for each training instance
```

---

## 3. Random Forests

A Random Forest is an ensemble of Decision Trees trained via bagging (typically with `max_samples` equal to training set size).

### Key Innovation
Instead of searching for the best feature when splitting nodes, Random Forests search for the best feature among a **random subset of features**. This introduces extra randomness, trading higher bias for lower variance.

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(
    n_estimators=500, 
    max_leaf_nodes=16, 
    n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

**Equivalent BaggingClassifier:**
```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=500, 
    max_samples=1.0, 
    bootstrap=True, 
    n_jobs=-1)
```

### Extra-Trees (Extremely Randomized Trees)

Makes trees even more random by using **random thresholds** for each feature rather than searching for the best thresholds.

**Advantages:**
- Much faster to train than regular Random Forests
- Trades more bias for lower variance

```python
from sklearn.ensemble import ExtraTreesClassifier

extra_trees_clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
extra_trees_clf.fit(X_train, y_train)
```

**Note:** Use cross-validation to determine whether RandomForestClassifier or ExtraTreesClassifier performs better.

### Feature Importance

Random Forests measure feature importance by looking at how much tree nodes using that feature reduce impurity on average across all trees.

```python
from sklearn.datasets import load_iris

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

# Output:
# sepal length (cm) 0.112
# sepal width (cm) 0.023
# petal length (cm) 0.441  # Most important
# petal width (cm) 0.423   # Second most important
```

This is invaluable for **feature selection** and understanding what features actually matter.

### Random Patches and Random Subspaces

The `BaggingClassifier` also supports sampling features:
- `max_features`: number of features to sample
- `bootstrap_features`: whether to sample with replacement

**Random Patches method:** Sampling both training instances AND features
**Random Subspaces method:** Keeping all instances but sampling features (set `bootstrap=False` and `max_samples=1.0`)

---

## 4. Boosting

**Boosting** combines several weak learners into a strong learner by training predictors sequentially, each trying to correct its predecessor.

### AdaBoost (Adaptive Boosting)

The algorithm pays more attention to training instances that predecessors underfitted by increasing their relative weights.

**How It Works:**
1. Train a base classifier and make predictions on training set
2. Increase weights of misclassified instances
3. Train second classifier using updated weights
4. Repeat: make predictions, update weights, train next classifier
5. Stop when desired number of predictors reached or perfect predictor found

**Mathematical Details:**

**Weighted Error Rate of the jth predictor:**

$$r_j = \frac{\sum_{i=1, \hat{y}_j^{(i)} \neq y^{(i)}}^{m} w^{(i)}}{\sum_{i=1}^{m} w^{(i)}}$$

where $\hat{y}_j^{(i)}$ is the jth predictor's prediction for the ith instance.

**Predictor Weight:**

$$\alpha_j = \eta \log \frac{1 - r_j}{r_j}$$

where η is the learning rate hyperparameter (defaults to 1). The more accurate the predictor, the higher its weight. If it's just guessing randomly, weight is close to zero. If it's mostly wrong (less accurate than random guessing), weight is negative.

**Weight Update Rule:**

For i = 1, 2, ..., m:

$$w^{(i)} \leftarrow \begin{cases} 
w^{(i)} & \text{if } \hat{y}_j^{(i)} = y^{(i)} \\
w^{(i)} \exp(\alpha_j) & \text{if } \hat{y}_j^{(i)} \neq y^{(i)}
\end{cases}$$

Then all instance weights are normalized (divided by $\sum_{i=1}^{m} w^{(i)}$).

**AdaBoost Predictions:**

$$\hat{y}(x) = \underset{k}{\operatorname{argmax}} \sum_{j=1, \hat{y}_j(x) = k}^{N} \alpha_j$$

where N is the number of predictors.

**Implementation:**

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # Decision Stump
    n_estimators=200,
    algorithm="SAMME.R",  # Uses class probabilities
    learning_rate=0.5)
ada_clf.fit(X_train, y_train)
```

**Note:** A Decision Stump is a Decision Tree with `max_depth=1` (one decision node + two leaf nodes).

**Regularization:** If overfitting, reduce `n_estimators` or regularize the base estimator more strongly.

**Important Limitation:** Cannot be parallelized (each predictor depends on the previous one).

### Gradient Boosting

Instead of tweaking instance weights, Gradient Boosting fits new predictors to the **residual errors** made by the previous predictor.

**Manual Implementation:**

```python
from sklearn.tree import DecisionTreeRegressor

# Train first predictor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)

# Train second predictor on residual errors
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)

# Train third predictor on residual errors of second
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

# Make predictions by summing all trees
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

**Using GradientBoostingRegressor:**

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(
    max_depth=2, 
    n_estimators=3, 
    learning_rate=1.0)
gbrt.fit(X, y)
```

### Learning Rate and Shrinkage

Lower learning rates require more trees but generally generalize better. This regularization technique is called **shrinkage**.

**Finding Optimal Number of Trees (Early Stopping):**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
```

**Early Stopping with warm_start:**

```python
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0

for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping
```

### Stochastic Gradient Boosting

The `subsample` hyperparameter specifies the fraction of training instances for each tree:

```python
gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=100,
    subsample=0.25)  # Each tree trained on 25% of instances
```

Trades higher bias for lower variance and speeds up training considerably.

### XGBoost (Extreme Gradient Boosting)

An optimized implementation that's extremely fast, scalable, and portable. Often a key component of winning ML competition entries.

```python
import xgboost

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)

# With automatic early stopping
xgb_reg.fit(X_train, y_train,
            eval_set=[(X_val, y_val)], 
            early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
```

---

## 5. Stacking (Stacked Generalization)

Instead of using simple functions to aggregate predictions, **train a model to perform the aggregation**. This final predictor is called a **blender** or **meta learner**.

### Training Process

**Step 1:** Split training set into two subsets
- Subset 1: Train first layer of predictors
- Subset 2: Hold-out set for training the blender

**Step 2:** First layer makes predictions on hold-out set

**Step 3:** Create new training set using first layer's predictions as features

**Step 4:** Train blender on this new training set

### Multi-Layer Stacking

For multiple blender layers, split training set into three subsets:
- Subset 1: Train first layer
- Subset 2: Create training set for second layer (using first layer's predictions)
- Subset 3: Create training set for third layer (using second layer's predictions)

### Important Note

Scikit-Learn doesn't support stacking directly. You can:
- Implement your own
- Use open-source implementations like **DESlib**

---

## Summary Comparison

| Method | Parallelizable | Key Strength | Common Use Case |
|--------|----------------|--------------|-----------------|
| **Voting** | Yes | Simple, effective | Combining diverse models |
| **Bagging** | Yes | Reduces variance | Random Forests |
| **Pasting** | Yes | Faster than bagging | Large datasets |
| **Boosting** | No/Partial | Corrects predecessors | Gradient Boosting, AdaBoost |
| **Stacking** | Partially | Learns to combine | Competition winning solutions |

---

## Key Takeaways

1. **Ensemble methods often outperform individual predictors** by combining multiple models
2. **Diversity is crucial** - predictors should be as independent as possible
3. **Random Forests** are simple yet powerful, excellent for feature importance
4. **Bagging reduces variance** through parallel training on random subsets
5. **Boosting corrects errors sequentially**, trading some parallelization for accuracy
6. **Early stopping** is essential for Gradient Boosting to prevent overfitting
7. **Stacking** can achieve excellent results but requires careful implementation
8. **XGBoost** is the go-to implementation for Gradient Boosting in practice

---

## Practical Tips

### For Overfitting:
- **AdaBoost**: Reduce `n_estimators` or regularize base estimator
- **Gradient Boosting**: Decrease `learning_rate` or use early stopping

### For Speed:
- Use **bagging/pasting** (parallelizable)
- Use **Extra-Trees** (faster than Random Forests)
- Use **subsample < 1.0** in Gradient Boosting

### For Best Results:
- Combine diverse algorithms in voting ensemble
- Use soft voting when classifiers provide probability estimates
- Consider stacking for competition-level performance
- Always validate with cross-validation

---

## Exercises

1. **If you have trained five different models on the exact same training data, and they all achieve 95% precision, is there any chance that you can combine these models to get better results? If so, how? If not, why?**

2. **What is the difference between hard and soft voting classifiers?**

3. **Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, Random Forests, or stacking ensembles?**

4. **What is the benefit of out-of-bag evaluation?**

5. **What makes Extra-Trees more random than regular Random Forests? How can this extra randomness help? Are Extra-Trees slower or faster than regular Random Forests?**

6. **If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak and how?**

7. **If your Gradient Boosting ensemble overfits the training set, should you increase or decrease the learning rate?**

8. **Load the MNIST data (introduced in Chapter 3), and split it into a training set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing). Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM classifier. Next, try to combine them into an ensemble that outperforms each individual classifier on the validation set, using soft or hard voting. Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?**

9. **Run the individual classifiers from the previous exercise to make predictions on the validation set, and create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image's class. Train a classifier on this new training set. Congratulations, you have just trained a blender, and together with the classifiers it forms a stacking ensemble! Now evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble's predictions. How does it compare to the voting classifier you trained earlier?**

---

**Note:** Solutions to these exercises are available in Appendix A of the original text.