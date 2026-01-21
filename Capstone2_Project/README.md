# Thyroid Disease Detector

This project focuses on the prediction of thyroid disease using machine learning techniques. The model analyzes biological factors and hormone measurements commonly used in medicine to assess thyroid function and identify potential thyroid disorders.

## Problem Description

Thyroid disease is a common endocrine disorder that affects the normal functioning of the thyroid gland. Based on specific biological factors and hormone levels, the goal is to determine whether a person has a thyroid disorder or not.
There are two primary types of thyroid disorders:
Hypothyroidism – a condition in which the thyroid gland is underactive and does not produce sufficient thyroid hormones.
Hyperthyroidism – a condition in which the thyroid gland is overactive and produces excessive thyroid hormones.

Among these, hypothyroidism is the most prevalent. In this project, I used machine learning algorithms to analyze patient data and predict whether an individual is affected by hypothyroidism based on relevant biological and hormonal features.

## Dataset Description

The data can be downloaded from here https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set/code
The dataset is collected from 3772 patients which contains following variables:

| Category | Variables |
|--------|-----------|
| Demographic | age, sex |
| Medication & Treatment | on thyroxine, query on thyroxine, on antithyroid medication, I131 treatment |
| Medical Conditions | sick, pregnant, thyroid surgery, lithium, goitre, tumor, hypopituitary, psych |
| Thyroid Status Queries | query hypothyroid, query hyperthyroid |
| Hormone Measurements (Flags) | TSH measured, T3 measured, TT4 measured, T4U measured, FTI measured, TBG measured |
| Hormone Levels | TSH, T3, TT4, T4U, FTI, TBG |
| Referral Information | referral source |
| Target Variable | binaryClass (thyroid disease indicator) |


## Project Structure and Workflow
### Jupyter Notebook

The Jupyter Notebook (notebook.ipynb) walks through the complete machine learning pipeline, including:

* Data preparation
    * Loading the thyroid dataset
    * Cleaning missing and inconsistent values
    * Encoding categorical variables
    * Feature engineering

* Exploratory Data Analysis (EDA)
    * Descriptive statistics
    * Histograms of numerical variables
    * Correlation matrix and target correlation analysis

* Model development and comparison
    * Linear Regression (baseline)
    * Random Forest Regressor
    * XGBoost Regressor

* Hyperparameter tuning using validation sets and manual loops
    * Evaluation and model selection
    * RMSE calculation
    * Comparison of baseline and tuned models
    * Final model selection based on test RMSE

## Installation
### Clone the repository and install dependencies:

```
git clone https://github.com/yadwink/Machine-Learning-Zoomcamp-2025/tree/main/Capstone2_Project
cd Capstone2_Project
pip install -r requirements.txt

```
**Make sure you are using Python 3.12 and an activated virtual environment.** 

## Training the Model
### Train the final model and save it to a file:

```
python train.py --data data/thyroid_clean.csv --model-out model.pkl

```
This script:

Trains the final Random Forest model using optimal parameters

Evaluates it on the test set

Saves the trained model as model.pkl

## Running the Prediction Service
Start the API manually

You can start the prediction service locally using:

Running the Prediction Service
Start the API manually

```
python predict.py
```

## Test the API with curl

Example request:

Test the API with curl

```
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"TSH":0.4,"TT4":88,"T4U":0.89,"FTI":99,"age":16,"sex":0}'

  {"prediction": 0.82}
```

## Deployment Proof

The model was deployed locally using Docker.  
The container runs a Flask-based REST API that serves predictions on port 8080.

Proof of deployment includes:
- A running Docker container (`docker ps`)
- A successful prediction request using `curl`

Screenshots is added as Deployment-Proof.png

The prediction service was deployed locally using Docker.  
The container exposes port 8080 and serves predictions via a REST API.

Deployment steps:
1. Build Docker image using `docker build`
2. Run container using `docker run`
3. Send POST requests to `/predict` endpoint
4. Receive JSON predictions successfully

This confirms the model is deployed and operational.

