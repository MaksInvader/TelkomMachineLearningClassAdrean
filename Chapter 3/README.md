# Chapter 3: Classification

## Introduction

This chapter covers classification systems in Machine Learning, focusing on the MNIST dataset of handwritten digits. While regression predicts values, classification predicts classes.

## MNIST Dataset

The MNIST dataset contains 70,000 images of handwritten digits (28×28 pixels = 784 features per image). Each pixel's intensity ranges from 0 (white) to 255 (black).

### Loading the Data

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'details', 'categories', 'url'])

X, y = mnist["data"], mnist["target"]
X.shape  # (70000, 784)
y.shape  # (70000,)
```

### Visualizing a Digit

```python
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

y[0]  # '5'
y = y.astype(np.uint8)  # Convert labels to integers
```

### Train/Test Split

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

The training set is already shuffled, which ensures similar cross-validation folds and prevents poor performance from similar consecutive instances.

## Binary Classification

We'll start with a simple binary classifier to detect the digit 5.

### Creating Target Vectors

```python
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)
```

### Training an SGD Classifier

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])  # array([True])
```

**Note:** SGDClassifier handles large datasets efficiently by processing instances independently. Set `random_state` for reproducible results.

## Performance Measures

### Cross-Validation

#### Manual Implementation

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))  # prints 0.9502, 0.96565, and 0.96495
```

**StratifiedKFold** performs stratified sampling to maintain representative class ratios in each fold.

#### Using cross_val_score

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.96355, 0.93795, 0.95615])
```

#### Why Accuracy Can Be Misleading

```python
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.91125, 0.90855, 0.90915])
```

This "never-5" classifier achieves 90% accuracy simply because only 10% of images are 5s. **Accuracy is not preferred for skewed datasets.**

### Confusion Matrix

A confusion matrix counts how often class A instances are classified as class B.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# array([[53057,  1522],
#        [ 1325,  4096]])
```

**Structure:**
- Rows: Actual classes
- Columns: Predicted classes
- **True Negatives (TN):** 53,057 non-5s correctly classified
- **False Positives (FP):** 1,522 non-5s wrongly classified as 5s
- **False Negatives (FN):** 1,325 5s wrongly classified as non-5s
- **True Positives (TP):** 4,096 5s correctly classified

Perfect classifier:
```python
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)
# array([[54579,     0],
#        [    0,  5421]])
```

### Precision and Recall

**Precision:** Accuracy of positive predictions

$$\text{precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity/True Positive Rate):** Ratio of positive instances correctly detected

$$\text{recall} = \frac{TP}{TP + FN}$$

```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)  # 0.729 (72.9%)
recall_score(y_train_5, y_train_pred)     # 0.756 (75.6%)
```

### F₁ Score

The **F₁ score** is the harmonic mean of precision and recall:

$$F_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}} = \frac{TP}{TP + \frac{FN + FP}{2}}$$

```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)  # 0.742
```

The F₁ score favors classifiers with similar precision and recall. However, depending on context:
- **High precision needed:** Safe video classifier (reject many good videos but keep only safe ones)
- **High recall needed:** Shoplifter detector (30% precision acceptable if 99% recall)

**Precision/Recall Trade-off:** Increasing precision reduces recall, and vice versa.

### Precision/Recall Trade-off

The SGDClassifier computes a score for each instance. If the score exceeds a threshold, it assigns the instance to the positive class.

#### Getting Decision Scores

```python
y_scores = sgd_clf.decision_function([some_digit])
# array([2412.53175101])

threshold = 0
y_some_digit_pred = (y_scores > threshold)
# array([True])

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
# array([False])
```

#### Computing Precision and Recall for All Thresholds

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

#### Plotting Precision/Recall vs Threshold

```python
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    # Add legend, axis labels, and grid

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```

**Note:** Precision can decrease when raising the threshold, but recall only decreases.

#### Selecting a Threshold for 90% Precision

```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]  # ~7816

y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y_train_5, y_train_pred_90)  # 0.900
recall_score(y_train_5, y_train_pred_90)     # 0.437
```

High precision is achieved, but recall is low (only 43.7% of 5s detected).

### ROC Curve

The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (recall) against the False Positive Rate.

**False Positive Rate (FPR):**

$$\text{FPR} = 1 - \text{TNR} = \frac{FP}{FP + TN}$$

Where **TNR (True Negative Rate/Specificity)**:

$$\text{TNR} = \frac{TN}{TN + FP}$$

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    # Add axis labels and grid

plot_roc_curve(fpr, tpr)
plt.show()
```

A good classifier stays far from the diagonal (toward the top-left corner).

#### ROC AUC Score

**Area Under the Curve (AUC):** Perfect classifier has ROC AUC = 1; random classifier has ROC AUC = 0.5.

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)  # 0.961
```

**When to use PR vs ROC curve:**
- **Use PR curve:** When positive class is rare or false positives matter more
- **Use ROC curve:** Otherwise

### Comparing Classifiers: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]  # Probability of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)  # 0.998
```

The RandomForestClassifier significantly outperforms SGDClassifier (99% precision, 86.6% recall).

## Multiclass Classification

**Multiclass classifiers** distinguish between more than two classes.

### Strategies

1. **One-vs-Rest (OvR):** Train N binary classifiers (one per class). Select the class with the highest score.

2. **One-vs-One (OvO):** Train N × (N-1) / 2 classifiers (one for each pair). Select the class that wins the most duels.

**OvO** is preferred for algorithms that scale poorly (e.g., SVM).
**OvR** is preferred for most algorithms.

### Training a Multiclass Classifier

```python
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)  # y_train, not y_train_5
svm_clf.predict([some_digit])  # array([5], dtype=uint8)
```

Scikit-Learn automatically uses OvO for SVC (trains 45 binary classifiers).

```python
some_digit_scores = svm_clf.decision_function([some_digit])
# Returns 10 scores (one per class)

np.argmax(some_digit_scores)  # 5
svm_clf.classes_  # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### Forcing OvR or OvO

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])  # array([5], dtype=uint8)
len(ovr_clf.estimators_)  # 10
```

### SGD Classifier (Native Multiclass)

```python
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])  # array([5], dtype=uint8)

sgd_clf.decision_function([some_digit])
# array([[-15955.23, -38080.96, -13326.67, 573.53, -17680.68,
#         2412.53, -25526.86, -12290.16, -7946.05, -10631.36]])
```

SGD classifiers handle multiple classes natively.

### Evaluating Multiclass Classifiers

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.849, 0.871, 0.870])
```

#### Improving with Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# array([0.897, 0.896, 0.907])
```

Scaling improves accuracy from ~85% to ~90%.

## Error Analysis

### Confusion Matrix Analysis

```python
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```

### Analyzing Error Rates

```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)  # Focus on errors
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
```

**Key insights:**
- Rows represent actual classes; columns represent predicted classes
- Bright columns indicate many images misclassified as that class
- 3s and 5s are often confused

### Analyzing Individual Errors

```python
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
```

**Linear model limitations:** SGDClassifier assigns weights per pixel per class. It's sensitive to image shifting and rotation. The difference between 3s and 5s is mainly the position of the joining line.

**Improvement strategies:**
- Preprocess images (centering, rotation correction)
- Engineer features (count closed loops)
- Use image processing libraries (Scikit-Image, Pillow, OpenCV)

## Multilabel Classification

**Multilabel classification** outputs multiple binary labels per instance.

### Example

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])  # array([[False, True]])
```

The digit 5 is not large (False) and is odd (True).

### Evaluation

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")  # 0.976
```

**Options:**
- `average="macro"`: Equal weight for all labels
- `average="weighted"`: Weight by support (number of instances per label)

## Multioutput Classification

**Multioutput-multiclass classification:** Each label can be multiclass (more than two values).

### Example: Noise Removal

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```

The classifier outputs pixel intensities (0-255) for each pixel, making it a multioutput system.

## Exercises

1. **Build a 97%+ accuracy classifier** for MNIST using KNeighborsClassifier with grid search on `weights` and `n_neighbors`.

2. **Data augmentation:** Write a function to shift MNIST images by one pixel in any direction. Create four shifted copies per training image, add to training set, and retrain.

3. **Titanic dataset:** Build a survival classifier (available on Kaggle).

4. **Spam classifier:**
   - Download spam/ham from Apache SpamAssassin
   - Split into train/test sets
   - Build pipeline to convert emails to feature vectors (word presence/absence)
   - Add hyperparameters: strip headers, lowercase, remove punctuation, URL/number replacement, stemming
   - Train multiple classifiers for high precision and recall

## Key Takeaways

- **Accuracy** can be misleading with skewed datasets
- **Precision/Recall trade-off** depends on application requirements
- **Confusion matrix** provides detailed error analysis
- **ROC curves** and **AUC** help compare classifiers
- **Multiclass strategies:** OvR for most algorithms, OvO for SVMs
- **Error analysis** guides improvement strategies
- **Data preprocessing** (scaling, augmentation) significantly improves performance
- **Multilabel** and **multioutput** classification extend binary classification concepts
