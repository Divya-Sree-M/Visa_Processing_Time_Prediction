🛂 AI Enabled Visa Processing Time Prediction

📌 Project Overview

Visa applicants often face uncertainty due to long and unpredictable processing times.
This project builds a machine learning–based predictive system that estimates visa processing time using historical application data, enabling better transparency and planning for applicants.

The system analyzes factors such as applicant country, job role, wage level, industry, and seasonal trends to predict the expected number of processing days.

🎯 Objectives

Predict visa processing time using historical data

Analyze seasonal and regional patterns affecting processing

Compare baseline and advanced regression models

Build a deployment-ready prediction pipeline

🧠 Approach & Methodology

1️⃣ Data Preprocessing

Removed irrelevant and post-decision fields

Handled missing values and inconsistent formats

Converted categorical features into numerical representations using:

Ordinal encoding (education, wage level)

Binary encoding (experience required)

Grouping + target mean encoding for high-cardinality features

Extracted seasonal features from application dates

Removed raw string and datetime columns after encoding

2️⃣ Feature Engineering

Key engineered features include:

wage_level_encoded

education_encoded

experience_required_encoded

Grouped & mean-encoded features:

industry_grouped_enc

city_grouped_enc

job_grouped_enc

country_grouped_enc

Seasonal indicators:

app_month

app_quarter

seasonal_index

Aggregated historical averages:

country_avg_days

industry_avg

job_avg

visa_avg

🤖 Models Implemented (Milestone 3)
Baseline Model

Linear Regression

Used to establish a baseline

Performs reasonably but struggles with non-linear patterns

Final Model

Random Forest Regressor

Captures non-linear relationships

More robust to outliers

Selected as the final model

📊 Model Performance
Model	MAE (days)	RMSE (days)	R² Score
Linear Regression	79.20	144.02	0.46
Random Forest Regressor	57.53	123.13	0.61

✅ Random Forest Regressor outperforms Linear Regression and is chosen as the final model.
The random forest regressor model couldn't be uploaded because of it's size.
🏆 Key Results

Reduced average prediction error by ~22 days

Explained 61% variance in visa processing time

Successfully handled complex, non-linear data patterns

🚀Milestone 4: Deployment, Testing & Documentation

Objective:
Deploy the visa processing time prediction system and ensure reliability through testing and structured documentation.

Key Activities:

Developed a web-based prediction interface using Streamlit for user interaction

Integrated the trained Random Forest prediction engine with the frontend

Implemented input validation and preprocessing pipelines for real-time predictions

Performed unit testing on core components such as preprocessing, feature encoding, and model inference

Maintained a defect tracker to log, analyze, and resolve issues during development

Prepared Agile documentation, including sprint planning and task tracking

Structured project documentation for maintainability and future scalability

Outcome:
A deployment-ready, well-documented AI system capable of providing real-time visa processing time estimates, aligned with industry-grade development practices.

☁️ Deployment Status & Limitations

The machine learning pipeline and Streamlit application were developed to be deployment-ready.

Cloud deployment could not be completed due to resource and environment constraints within the internship timeline.

The project structure and application code are prepared for seamless deployment on platforms such as Streamlit Community Cloud or cloud services.

📁 Project Structure
Visa_Processing_Time_Prediction/
│
├── data/
│   └── preprocessed.csv
│
├── docs/
│   ├── Divya_Agile.pdf
│   ├── Divya_UnitTesting.xlsx
│   └── Divya_DefectTracker.xlsx
│
├── models/
│   └── linear_regression_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── predict.py
│
├── app.py                  # Streamlit web application
├── run_processing.py       # Data preprocessing pipeline
├── run_training.py         # Model training script
├── visa_prediction.ipynb   # Experimentation notebook
├── requirements.txt
├── README.md
└── LICENSE


🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Joblib

Streamlit


📌 Conclusion

This project demonstrates a complete machine learning pipeline—from data preprocessing and feature engineering to model evaluation and selection.
The Random Forest Regressor proved to be an effective solution for predicting visa processing times and is suitable for real-world deployment.

👩‍💻 Author

Divya Sree Machani
AI Intern | Machine Learning Enthusiast
