# 🔥 Calories Burnt Prediction

> A machine learning project that predicts calories burned during physical activity using physiological and demographic features.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Models](https://img.shields.io/badge/Models-5-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Records-15%2C000-purple?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-MAE-red?style=flat-square)

---

## Overview

This project predicts the number of calories burnt during physical activity using supervised machine learning regression techniques. It combines physiological features (heart rate, body temperature, weight) with activity data (duration) to estimate calorie expenditure accurately.

Five regression models are trained and compared — from simple linear baselines to ensemble methods — with **XGBoost** and **Random Forest** emerging as the top performers.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Calories Burnt Prediction](https://www.kaggle.com/) |
| Total Records | 15,000 |
| Target Variable | `Calories` |

### Features

| Feature | Type |
|---|---|
| Gender | Categorical (encoded) |
| Age | Numerical |
| Height | Numerical |
| Weight | Numerical |
| Duration | Numerical (minutes) |
| Heart Rate | Numerical (bpm) |
| Body Temperature | Numerical (°C) |

---

## Project Workflow

```
Data Loading → Preprocessing → EDA → Feature Selection → Model Training → Evaluation
```

### 1. Data Loading & Preprocessing
- Dataset downloaded via **KaggleHub**
- Gender encoded from categorical to numerical
- Feature scaling applied with **StandardScaler**

### 2. Exploratory Data Analysis
- Scatter plots to examine feature-target relationships
- Distribution plots for each feature
- Correlation heatmap to detect multicollinearity

### 3. Feature Selection
- Highly correlated features removed to reduce multicollinearity and improve model generalisation

### 4. Model Training
Five regression models trained and compared:

| Model | Type |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | Regularised linear (L2) |
| Lasso Regression | Regularised linear (L1) |
| Random Forest Regressor | Ensemble — bagging |
| XGBoost Regressor | Ensemble — boosting |

### 5. Evaluation
- Metric: **Mean Absolute Error (MAE)**
- Both training and validation errors compared across all models

---

## Results

| Model | Performance |
|---|---|
| 🥇 XGBoost Regressor | Best — lowest validation MAE |
| 🥈 Random Forest Regressor | Close second — strong generalisation |
| Linear / Ridge / Lasso | Higher error — limited by linearity assumption |

Ensemble models significantly outperformed linear baselines, demonstrating better capture of non-linear relationships between physiological features and calorie expenditure.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting and visualisation |
| `seaborn` | Statistical plots — heatmaps, distributions |
| `scikit-learn` | Preprocessing, linear models, Random Forest, MAE evaluation |
| `xgboost` | Gradient boosting regressor |
| `kagglehub` | Programmatic dataset download from Kaggle |

---

## Project Structure

```
Calories-Burnt-Prediction/
│
├── calorie.py                        ← Main script (preprocessing, training, evaluation)
├── Calories Burnt Prediction.pdf     ← Project report
├── output/                           ← Saved plots and results
├── LICENSE                           ← MIT License
└── README.md                         ← Project documentation
```

---

## Installation

### Prerequisites

- Python 3.x
- A Kaggle account (for dataset download via KaggleHub)

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost kagglehub
```

### Kaggle API Setup

To use KaggleHub for dataset download, place your `kaggle.json` credentials file at:

```
~/.kaggle/kaggle.json        # macOS / Linux
C:\Users\<user>\.kaggle\kaggle.json   # Windows
```

---

## Usage

```bash
python calorie.py
```

The script will:
1. Download and load the dataset
2. Preprocess and scale features
3. Run EDA and generate visualisations into `output/`
4. Train all five models
5. Print MAE comparison across models

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

