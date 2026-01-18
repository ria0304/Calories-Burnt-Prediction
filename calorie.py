import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

\
path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")
csv_path = os.path.join(path, "calories.csv")

df = pd.read_csv(csv_path)

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())



sb.scatterplot(x='Height', y='Weight', data=df)
plt.show()

features = ['Age', 'Height', 'Weight', 'Duration']
plt.figure(figsize=(15, 10))

for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sample_df = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=sample_df)

plt.tight_layout()
plt.show()



float_cols = df.select_dtypes(include='float').columns
plt.figure(figsize=(15, 10))

for i, col in enumerate(float_cols):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)

plt.tight_layout()
plt.show()



df.replace({'male': 0, 'female': 1}, inplace=True)

plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()



df.drop(['Weight', 'Duration'], axis=1, inplace=True)



X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=22
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



models = [
    LinearRegression(),
    XGBRegressor(),
    Lasso(),
    RandomForestRegressor(),
    Ridge()
]

for model in models:
    model.fit(X_train, y_train)

    print(model)
    train_preds = model.predict(X_train)
    print("Training Error :", mae(y_train, train_preds))

    val_preds = model.predict(X_val)
    print("Validation Error :", mae(y_val, val_preds))
    print("-" * 40)
