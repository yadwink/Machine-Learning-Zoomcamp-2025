# Decision Trees & Ensemble Learning

- **Example**: Bank loan application for mobile phone
- Bank needs to predict: Will customer default on loan?
- Model outputs: Risk score (probability of default)
- **Binary classification problem**: 
  - y = 0 (OK - customer repays)
  - y = 1 (DEFAULT - customer doesn't repay)

## **What is a Decision Tree?**
- Tree-like structure for making predictions
- **Components**:
  - Internal nodes = conditions/features
  - Branches = decision outcomes (true/false)
  - Leaf nodes = final predictions
- Benefits: Simple, easy to interpret
- Used for classification and regression

## **How Decision Trees Work**
- Each node contains a condition
- Left arrow = condition false
- Right arrow = condition true
- Process continues until final decision reached

## **The Overfitting Problem**
- Tree memorizes training data too specifically
- Creates unique rule for each example
- Works on training data but fails on new data
- **Cause**: Tree grows too deep
- **Solution**: Restrict tree depth

## **Decision Stump**
- Decision tree with depth = 1
- Just one condition node
- Validation performance: 74% (vs 65% overfitted)
- Simpler but nearly as effective

## **Decision Tree Learning Algorithm**
1. **Find Best Split**: Test all features/thresholds, choose lowest impurity
2. **Check Max Depth**: Stop if depth limit reached
3. **Check LEFT subset**: If large enough and not pure, repeat splitting
4. **Check RIGHT subset**: If large enough and not pure, repeat splitting

## **Key Parameters to Tune**
- **criterion**: Impurity measure ('gini' or 'entropy')
- **max_depth**: Maximum tree depth (prevents overfitting)
- **min_samples_leaf**: Minimum samples per leaf node

## **Parameter Tuning Process**
- Goal: Maximize AUC on validation set
- Start with max_depth tuning first
- Test multiple values including 'None' (unrestricted)
- Balance between model simplicity and complexity

# Gradient Boosting & XGBoost

## **What is Boosting?**
- Different approach from random forest for combining decision trees
- Models trained **sequentially** (one after another)
- Each new model corrects errors of previous model
- Method called "boosting"

## **Gradient Boosting vs Random Forest**

### **Random Forest**:
- Multiple **independent** trees trained on same dataset
- Final prediction = average of all trees: (1/n) * Σ(pi)

### **Boosting**:
- **Sequential training** process:
  1. Train first model → makes predictions → evaluate errors
  2. Train second model based on first model's errors → makes predictions → introduces new errors
  3. Train third model to correct second model's errors
  4. Repeat for multiple iterations
  5. Combine all predictions into final result
- Core idea: Each model learns from previous model's mistakes

## **XGBoost Library**
- Highly effective implementation of gradient boosting
- Installation: `!pip install xgboost`
- Import: `import xgboost as xgb`

## **Training Process**
- Structure data into **DMatrix** format (optimized for faster training)
- Use `xgb.train()` function

## **Key XGBoost Parameters**

- **eta**: Learning rate (controls how quickly model learns)
- **max_depth**: Controls tree size (like in decision trees)
- **min_child_weight**: Minimum observations in leaf node (similar to min_samples_leaf)
- **objective**: Specify problem type (e.g., 'binary:logistic' for binary classification)
- **nthread**: Number of threads for parallel training
- **seed**: Controls randomization
- **verbosity**: Controls detail level of warnings/messages
- **num_boost_round**: Number of trees to train

## **Model Performance**
- Make predictions: `model.predict(dval)`
- Returns one-dimensional array of predictions
- **Example result**: AUC ≈ 81% with default settings
- Good performance without parameter tuning
- Performance with 10 trees comparable to 200 trees

## **Important Considerations**
- Watch out for **overfitting**
- Be cautious about:
  - Number of trees trained
  - Tree sizes

# Notes: Selecting the Final Model

## **Model Selection Process**
- Compare best models from each algorithm type
- Evaluate performance on validation data
- Select overall winner
- Retrain on complete dataset
- Final evaluation on test set

## **Model Comparison Results**

### **Simple Model** (Decision Tree):
- Validation performance: ~78-79%
- Simplest approach

### **Medium Complexity** (Random Forest):
- Validation performance: ~82-83%
- Balanced complexity

### **Advanced Model** (XGBoost):
- Validation performance: ~83%+
- Highest performance
- Selected as winner

## **Final Model Training Process**

### **Data Preparation**:
1. Combine all available training data
2. Clean and organize dataset (reset indices)
3. Separate target variable from features
4. Remove target column to prevent data leakage
5. Transform features into proper format
6. Prepare separate test dataset
7. Verify data integrity

### **Model-Specific Requirements**:
- Convert data into optimized format if needed
- Include feature names for tracking
- Keep test and train data consistent

## **Final Performance Assessment**
- Test set performance: Similar to validation (~83%)
- Small performance difference acceptable (<1%)
- Indicates good generalization
- No significant overfitting detected

## **Advanced Model Trade-offs**

### **Strengths**:
- Best performance for structured/tabular data
- Superior accuracy over simpler models
- Handles complex patterns well

### **Challenges**:
- Higher complexity to implement
- More parameters to configure
- Requires careful tuning
- Greater risk of overfitting
- Needs more expertise

### **Key Takeaway**:
- More complexity = Better results BUT harder to manage
- Choose based on: available expertise, time, and required accuracy  