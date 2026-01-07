Project Overview
This project focuses on predicting the number of calories burnt during physical activity using machine learning regression techniques. The model leverages physiological and activity-related features to estimate calorie expenditure accurately.
The project is implemented using Python, Pandas, Scikit-learn, and XGBoost, with exploratory data analysis and multiple regression models for comparison.

📂 Dataset

Source: Kaggle – Calories Burnt Prediction
Total Records: 15,000
Features include:
Gender
Age
Height
Weight
Duration of exercise
Heart Rate
Body Temperature
Target Variable:
Calories

🛠️ Technologies & Libraries Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
KaggleHub

⚙️ Project Workflo
Data Loading & Preprocessing
Dataset downloaded using KaggleHub
Handling categorical data (Gender encoding)
Feature scaling using StandardScaler
Exploratory Data Analysis
Scatter plots for feature relationships
Distribution plots
Correlation heatmap
Feature Selection
Removed highly correlated features to avoid multicollinearity

Model Training
Models used:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
XGBoost Regressor
Model Evaluation
Metric used: Mean Absolute Error (MAE)
Comparison of training and validation errors

 Results Summary
XGBoost Regressor and Random Forest Regressor performed best

Achieved low validation error compared to linear models

Ensemble models showed better generalization
