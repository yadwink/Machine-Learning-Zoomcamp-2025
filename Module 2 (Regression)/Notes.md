# Machine Learning Regression Module - Standard ML Project Steps

This module covers Regression and the standard steps needed to implement an ML project. Below are the key steps we follow:

## 1. Data Preparation/Cleaning
This step includes handling missing values and outliers.

**Imputation Methods:**
- **Mean/Median Imputation:** Filling in missing values with either the average (mean) or middle value (median) of that variable
  - Use mean for normally distributed data
  - Use median for skewed data or when outliers are present

- **K-Nearest Neighbors (KNN) Imputation:** A more sophisticated approach that finds similar data points (neighbors) and uses their values to estimate missing values
  - Preserves relationships between variables
  - More accurate than simple mean/median
  - Computationally more expensive

## 2. Data Integration
If multiple related datasets exist, merge them into a single dataset based on common identifiers (keys). This creates a unified dataset for analysis.

## 3. Data Transformation
Converting variables into suitable formats for analysis through feature engineering.

**Common transformations:**
- **One-Hot Encoding:** Converting categorical variables into binary columns (for nominal categories)
- **Label Encoding:** Converting categorical variables into numerical labels (for ordinal categories)
- **Feature Engineering:** Creating new variables that might have better relationships with target variables

## 4. Feature Scaling
Standardizing or normalizing the range of features so they're on a similar scale.

**Why it matters:**
Many algorithms use distance calculations or gradient descent, which are sensitive to feature magnitude. Without scaling, features with larger ranges dominate the model.

**Example:**
- Feature 1: Age (20-80)
- Feature 2: Income (20,000-200,000)
- Income would disproportionately influence the model due to its larger scale

**Common Scaling Techniques:**

### 4.1 Normalization (Min-Max Scaling)
- Scales features to range [0, 1]
- Formula: (x - min) / (max - min)
- **Use when:** You need bounded values and don't have outliers

### 4.2 Standardization (Z-score Normalization)
- Transforms data to mean=0, standard deviation=1
- Formula: (x - mean) / standard deviation
- **Use when:** Data is normally distributed or has outliers (more robust)

### 4.3 Robust Scaling
- Uses median and interquartile range
- Less affected by outliers

**When scaling is needed:**
- Distance-based algorithms: KNN, K-means, SVM
- Gradient descent algorithms: Neural networks, linear/logistic regression
- Algorithms with regularization

**When scaling is NOT needed:**
- Tree-based models: Decision Trees, Random Forests, XGBoost

## 5. Train-Validation-Test Split
Split the dataset into three parts for model building and evaluation:

- **Training set:** 60% - used to train the model
- **Validation set:** 20% - used to tune hyperparameters and prevent overfitting
- **Test set:** 20% - used for final model evaluation (untouched until the end)



