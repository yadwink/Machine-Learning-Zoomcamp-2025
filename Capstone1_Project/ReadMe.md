# Stress & Exercise Classification using Wearable Sensor Data

## Project Overview

### See the complete project description here
https://physionet.org/content/wearable-device-dataset/1.0.1/

This project analyzes physiological signals collected from Empatica E4 wristbands during **stress**, **aerobic**, and **anaerobic** conditions.  
We extract statistical features from cleaned signals (EDA, HR, TEMP, BVP, ACC) and train machine learning models to classify the type of activity or stress condition.



##  Dataset
Dataset: [Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (PhysioNet)](https://physionet.org/content/wearable-stress-exercise/1.0.1/)

### Folder structure
data\wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1\Wearable_Dataset
├── STRESS/
├── AEROBIC/
└── ANAEROBIC/


Each participant folder contains Empatica E4 sensor data (EDA, HR, BVP, TEMP, ACC, IBI, tags).



##  Running the Project


### 1. Create environment

python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

### 2. Data preparation

Clean and resample the raw signals:

python -m src.data.clean_signals 

or just use the following .csv: It was generated from the raw PhysioNet dataset by cleaning, resampling,
and aggregating Empatica E4 signals (EDA, HR, TEMP, BVP, ACC).

data/processed/features_per_session.csv


### 3. Model training

Train model and export it:

python train.py

### 4. Serve the model

Launch FastAPI web service:

python predict.py

Libraries used:

Python, Pandas, NumPy, Scikit-learn

### 5. FastAPI for deployment

Matplotlib & Seaborn for visualization

📈 Deliverables

EDA notebook: notebooks/notebook.ipynb

Model training: train.py

Web API service: predict.py

Docker deployment file



### 6. notebook.ipynb contains

**EDA + model experiments + feature insights**.  

### 7. API Prediction Example:


I tested the deployed FastAPI service by sending a JSON payload with the 10 engineered physiological features. The `/predict` endpoint returned:


{
  "prediction": "STRESS",
  "prediction_id": 0,
  "probabilities": {
    "STRESS": 0.82,
    "AEROBIC": 0.13,
    "ANAEROBIC": 0.05
  }
}


The model classifies this sample as STRESS with high confidence (~82%).

The probabilities for AEROBIC and ANAEROBIC are much lower, which indicates the model clearly distinguishes stress-like physiology (elevated EDA, specific HR patterns, lower movement) from exercise-like physiology (high HR + ACC).

This confirms that the end-to-end system, from data preprocessing and feature engineering, through model training, to deployment with FastAPI—works as expected.

### 8. Docker containerisation

docker build -t stress-exercise-api .
docker run --rm -p 8000:8000 stress-exercise-api

### 9. Proof of deployment (local Docker execution) ( see deployment_proof_1.png & deployment_proof_2.png)

## Using the API 

The FastAPI service is exposed locally after running the Docker container.

#### 1. Start the service

docker run --rm -p 8000:8000 stress-exercise-api

##### 2. Open API documentation

Once the container is running, open the interactive API docs in your browser:

http://127.0.0.1:8000/docs

This Swagger UI allows you to send test requests to the /predict endpoint.

##### 3. Example prediction request

You can also send a request from the command line:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "EDA_mean": 0.12,
    "EDA_std": 0.03,
    "TEMP_mean": 36.5,
    "TEMP_std": 0.2,
    "HR_mean": 85.4,
    "HR_std": 5.1,
    "BVP_mean": 0.45,
    "BVP_std": 0.1,
    "ACC_mag_mean": 0.9,
    "ACC_mag_std": 0.05
  }'



