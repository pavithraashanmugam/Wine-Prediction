# Wine Classification with Logistic Regression and Hyperparameter Tuning

This project demonstrates how to classify wine types using the popular Wine dataset, leveraging machine learning models like Logistic Regression. It involves using techniques such as GridSearchCV and RandomizedSearchCV for hyperparameter optimization to improve the performance of the model.

## Project Overview

The Wine dataset contains 178 samples of wines, with 13 numerical features related to chemical properties, such as alcohol content, malic acid, and proline, among others. The goal is to predict the wine's class based on these features. This repository showcases the process of data preprocessing, training machine learning models, and tuning hyperparameters for improved classification accuracy.

## Key Features
- **Data Loading**: The Wine dataset is loaded from `sklearn.datasets.load_wine()` and converted into a pandas DataFrame for easy analysis and manipulation.
- **Preprocessing**: The features are standardized using `StandardScaler` to ensure that the model's performance is not biased due to differing feature scales.
- **Modeling**: Logistic Regression is used as the primary classifier. We will explore hyperparameter tuning using both Grid Search (`GridSearchCV`) and Randomized Search (`RandomizedSearchCV`) to optimize the model.
- **Evaluation**: The model's performance is evaluated using cross-validation (5-fold cross-validation), and the best hyperparameters are selected based on the highest mean score.
  
## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

### Install Dependencies

To install the necessary libraries, run:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Code Walkthrough

### 1. Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
```
Here, we import the necessary libraries for data manipulation (`numpy`, `pandas`), machine learning tasks (`scikit-learn`), and visualization (`matplotlib`).

### 2. Loading and Preparing Data
We load the Wine dataset and convert it into a pandas DataFrame for better exploration.
```python
wine_dataset = load_wine()
df = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
df['target'] = wine_dataset.target
```

### 3. Splitting Features and Target Variable
We separate the features (X) from the target variable (Y):
```python
X = df.drop(columns='target', axis=1)
Y = df['target']
```

### 4. Data Standardization
Standardize the features using `StandardScaler` to bring all features to a comparable scale, which is important for models like Logistic Regression.
```python
scalar = StandardScaler()
X = scalar.fit_transform(X)
```

### 5. Hyperparameter Tuning: GridSearchCV
We perform hyperparameter optimization using `GridSearchCV` to find the best parameters for Logistic Regression:
```python
parameters = { 'penalty': ['l2'],
               'C': [1, 5, 10, 20],
               'solver': ['liblinear', 'lbfgs'] }
classifier = GridSearchCV(LogisticRegression(), parameters, cv=5)
classifier.fit(X, Y)
```
This grid search explores the different combinations of hyperparameters (`penalty`, `C`, and `solver`) and uses cross-validation to select the best-performing model.

### 6. Hyperparameter Tuning: RandomizedSearchCV
We also use `RandomizedSearchCV`, which is a more computationally efficient method for hyperparameter optimization, especially when the search space is large:
```python
classifier = RandomizedSearchCV(LogisticRegression(), parameters, cv=5)
classifier.fit(X, Y)
```

### 7. Results and Evaluation
Both GridSearchCV and RandomizedSearchCV provide results on the best hyperparameters and the corresponding performance:
```python
classifier.best_params_  # Best hyperparameters
classifier.best_score_   # Best score achieved
```
You can view detailed results from the search using `cv_results_` to inspect the performance of different hyperparameter combinations.

### 8. Visualizing Results (Optional)
You can visualize the results of your models using `matplotlib` to create performance plots.

### 9. Summary of Results
After running GridSearchCV and RandomizedSearchCV, you’ll get an output detailing the best combination of hyperparameters, such as:
- **Best Hyperparameters**: `C=1`, `solver=lbfgs`, `penalty=l2`
- **Best Accuracy Score**: 98.89%

### Example Output

After running the above code, you will get detailed information on the results of the hyperparameter search:

```python
{
    'mean_test_score': array([0.97777778, 0.98888889, 0.98333333, 0.97761905, 0.98333333]),
    'std_test_score': array([0.02078699, 0.01360828, 0.01360828, 0.0111947]),
    'rank_test_score': array([5, 1, 2, 6, 2], dtype=int32)
}
```

This provides information on the mean accuracy scores for each set of hyperparameters and ranks them by performance.

