# Chapter 2: End-to-End Machine Learning Project

## Overview

This chapter walks through a complete machine learning project from start to finish using the California Housing Prices dataset. The goal is to build a model that predicts median housing prices in California districts based on census data.

## Main Steps in an ML Project

1. Look at the big picture
2. Get the data
3. Discover and visualize the data to gain insights
4. Prepare the data for Machine Learning algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor, and maintain your system

---

## 1. Look at the Big Picture

### Frame the Problem

**Business Objective**: Build a model to predict median housing prices in districts. The predictions will feed into a downstream investment analysis system.

**Problem Type**:
- **Supervised learning** (labeled training examples with expected outputs)
- **Regression task** (predicting a value)
- **Multiple regression** (using multiple features)
- **Univariate regression** (predicting single value per district)
- **Batch learning** (no continuous data flow, data fits in memory)

### Select a Performance Measure

**Root Mean Square Error (RMSE)** - typical for regression problems:

$$\text{RMSE}(\mathbf{X}, h) = \sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(h(\mathbf{x}^{(i)}) - y^{(i)}\right)^2}$$

Where:
- $m$ = number of instances in the dataset
- $\mathbf{x}^{(i)}$ = feature vector of the $i$-th instance
- $y^{(i)}$ = label (desired output) for the $i$-th instance
- $h$ = prediction function (hypothesis)
- $\mathbf{X}$ = matrix containing all feature values (excluding labels)

**Alternative**: Mean Absolute Error (MAE) for datasets with many outliers:

$$\text{MAE}(\mathbf{X}, h) = \frac{1}{m}\sum_{i=1}^{m}\left|h(\mathbf{x}^{(i)}) - y^{(i)}\right|$$

---

## 2. Get the Data

### Download the Data

```python
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```

### Load the Data

```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

### Quick Look at Data Structure

```python
housing = load_housing_data()
housing.head()  # View first 5 rows
housing.info()  # Get overview of data types and null values
housing["ocean_proximity"].value_counts()  # Check categorical values
housing.describe()  # Statistical summary
```

**Dataset contains 10 attributes**:
- longitude, latitude
- housing_median_age
- total_rooms, total_bedrooms
- population, households
- median_income
- median_house_value (target)
- ocean_proximity (categorical)

### Create a Test Set

**Important**: Create test set early to avoid data snooping bias.

**Simple random sampling**:

```python
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

**Using Scikit-Learn**:

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

### Stratified Sampling

Ensures test set is representative of important attribute categories (e.g., income categories):

```python
# Create income categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Stratified split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove income_cat attribute
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

---

## 3. Discover and Visualize the Data

### Visualizing Geographical Data

```python
housing = strat_train_set.copy()

# Basic scatter plot
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# With population and price information
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", 
            figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), 
            colorbar=True)
plt.legend()
```

**Key insight**: Housing prices strongly related to location and population density.

### Looking for Correlations

```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```

**Output**:
```
median_house_value    1.000000
median_income         0.687170
total_rooms           0.135231
housing_median_age    0.114220
...
```

**Scatter matrix for promising attributes**:

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", 
              "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

**Correlation coefficient** ranges from -1 to 1:
- Close to 1: strong positive correlation
- Close to -1: strong negative correlation
- Close to 0: no linear correlation

⚠️ **Note**: Correlation coefficient only measures *linear* relationships.

### Experimenting with Attribute Combinations

Create new features that might be more useful:

```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# Check new correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```

---

## 4. Prepare the Data for Machine Learning

### Separate Predictors and Labels

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### Data Cleaning

**Handle missing values** (total_bedrooms has 207 missing values):

Three options:
1. Drop districts with missing values: `housing.dropna(subset=["total_bedrooms"])`
2. Drop the whole attribute: `housing.drop("total_bedrooms", axis=1)`
3. Fill with value (median): `housing["total_bedrooms"].fillna(median, inplace=True)`

**Using Scikit-Learn's SimpleImputer**:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# Remove categorical attribute (median only works on numerical)
housing_num = housing.drop("ocean_proximity", axis=1)

# Fit and transform
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# Convert back to DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, 
                         index=housing_num.index)
```

### Scikit-Learn Design Principles

**Consistency**: All objects share simple interface:

1. **Estimators**: Estimate parameters from dataset via `fit()` method
   - Hyperparameters set as instance variables
   
2. **Transformers**: Transform datasets via `transform()` method
   - Convenience method: `fit_transform()`
   
3. **Predictors**: Make predictions via `predict()` method
   - Quality measured via `score()` method

**Other principles**:
- **Inspection**: Hyperparameters accessible via public instance variables
- **Nonproliferation**: Uses NumPy arrays/SciPy sparse matrices
- **Composition**: Easy to combine building blocks
- **Sensible defaults**: Reasonable default values provided

### Handling Text and Categorical Attributes

**Convert categories to numbers**:

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat = housing[["ocean_proximity"]]
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
```

⚠️ **Problem**: ML algorithms may assume nearby values are similar (0 and 4 more similar than 0 and 1).

**One-hot encoding** - better solution:

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# Returns sparse matrix to save memory
```

### Custom Transformers

Create custom transformers by implementing `fit()`, `transform()`, and `fit_transform()`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

### Feature Scaling

**Why**: ML algorithms don't perform well when features have very different scales.

**Two common methods**:

1. **Min-max scaling (normalization)**: Scale to 0-1 range
   - Scikit-Learn: `MinMaxScaler`

2. **Standardization**: Zero mean, unit variance
   - Less affected by outliers
   - Doesn't bound to specific range
   - Scikit-Learn: `StandardScaler`

⚠️ **Important**: Fit scalers on training data only, then use to transform training and test sets.

### Transformation Pipelines

**Numerical pipeline**:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

**Full pipeline with ColumnTransformer** (handles numerical and categorical):

```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
```

---

## 5. Select and Train a Model

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Test on a few instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# Measure RMSE
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)  # => 68,628
```

**Result**: Model underfitting (RMSE of $68,628 too high).

### Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)  # => 0.0 (overfitting!)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
# Mean: 71,407 (worse than Linear Regression!)
```

⚠️ **Note**: Scikit-Learn uses utility function (greater is better), so MSE is negative.

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_rmse_scores = # ... cross-validation
# Mean: 50,182 (much better!)
```

**Save models for later**:

```python
import joblib

joblib.dump(my_model, "my_model.pkl")
# Later...
my_model_loaded = joblib.load("my_model.pkl")
```

---

## 6. Fine-Tune Your Model

### Grid Search

Systematically tries all combinations of hyperparameter values:

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Get best parameters
print(grid_search.best_params_)
# {'max_features': 8, 'n_estimators': 30}

# Get best estimator
print(grid_search.best_estimator_)

# View all scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

**Result**: 18 combinations × 5 folds = 90 training rounds.

### Randomized Search

Better when hyperparameter space is large:

```python
from sklearn.model_selection import RandomizedSearchCV

# Evaluates given number of random combinations
# More control over computing budget
```

### Ensemble Methods

Combine models that perform best - often better than individual models (covered in Chapter 7).

### Analyze Best Models

```python
feature_importances = grid_search.best_estimator_.feature_importances_

# Display with attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
```

**Top features**:
1. median_income (0.366)
2. INLAND (0.165)
3. pop_per_hhold (0.109)
4. longitude (0.073)
5. latitude (0.063)

### Evaluate on Test Set

```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)  # => 47,730
```

**Compute 95% confidence interval**:

```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))
# array([45685, 49691])
```

---

## 7. Launch, Monitor, and Maintain

### Deployment Options

1. **Save and load model**:
   ```python
   joblib.dump(my_model, "my_model.pkl")
   my_model_loaded = joblib.load("my_model.pkl")
   ```

2. **Deploy as web service** (REST API)
   - Makes upgrading easier
   - Simplifies scaling
   - Language-agnostic

3. **Cloud deployment** (e.g., Google Cloud AI Platform)
   - Upload model to cloud storage
   - Create model version
   - Get automatic load balancing and scaling

### Monitoring

**Why models "rot"**:
- World changes over time
- Data distribution shifts
- New categories emerge
- Camera/sensor specifications change

**Monitoring strategies**:

1. **Downstream metrics**: Track business KPIs affected by model
2. **Human evaluation**: Sample predictions for expert review
3. **Input quality**: Monitor for data drift, missing features, new categories

**Automated retraining**:
- Collect and label fresh data regularly
- Automate training and hyperparameter tuning
- Evaluate new model vs. previous on updated test set
- Deploy if performance hasn't decreased

**Best practices**:
- Keep backups of all model versions
- Keep backups of all dataset versions
- Have rollback procedures ready
- Monitor input data quality
- Create specialized test subsets for different scenarios

---

## Key Takeaways

1. **Data preparation is crucial** - often more work than model training
2. **Avoid data snooping bias** - create test set early and don't look at it
3. **Use stratified sampling** for important categorical attributes
4. **Visualize data** to gain insights and identify patterns
5. **Create transformation pipelines** for reproducibility and automation
6. **Start simple** - try basic models before complex ones
7. **Use cross-validation** to evaluate model performance
8. **Fine-tune systematically** with grid/randomized search
9. **Analyze feature importance** to understand model
10. **Infrastructure matters** - monitoring and maintenance are essential

---

## Exercises

1. Try Support Vector Machine regressor with various hyperparameters
2. Replace GridSearchCV with RandomizedSearchCV
3. Add transformer to select only most important attributes
4. Create single pipeline for full data preparation + prediction
5. Explore preparation options using GridSearchCV

---

## Additional Resources

- Full code available at: https://github.com/ageron/handson-ml2
- Kaggle competitions for practice: http://kaggle.com/
- California Housing dataset from StatLib repository
