# ===============================================================
# SCRIPT 1: TRAIN AND SAVE MODELS
# ===============================================================
# Purpose: To train all three weather models from the historical
#          dataset and save them to the 'models' folder.
# Run this script only once or when you need to retrain.
# ===============================================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- Step 1: Load and Prepare Data ---
print("--- Loading and preparing Kolkata dataset ---")
df = pd.read_csv('data/Kolkata_weather_data(2017-2022).csv')
df.columns = df.columns.str.strip()
df.rename(columns={
    'Date time': 'datetime',
    'Maximum Temperature': 'max_temp',
    'Minimum Temperature': 'min_temp',
    'Relative Humidity': 'humidity'
}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df = df[['max_temp', 'min_temp', 'humidity']]
df.dropna(inplace=True)
print("Dataset prepared successfully.")

# --- Step 2: Feature Engineering ---
print("\n--- Engineering features ---")
X = df.copy()
X['day_of_year'] = X.index.dayofyear
y_max = df['max_temp'].shift(-1)
y_min = df['min_temp'].shift(-1)
y_hum = df['humidity'].shift(-1)
X = X.iloc[:-1]
y_max = y_max.iloc[:-1]
y_min = y_min.iloc[:-1]
y_hum = y_hum.iloc[:-1]
print("Features and targets are ready.")

# --- Step 3: Train and Save Models ---
print("\n--- Training and saving all models ---")
os.makedirs('models', exist_ok=True)
common_model_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}

# Model 1: Max Temperature
model_max = RandomForestRegressor(**common_model_params)
model_max.fit(X, y_max)
joblib.dump(model_max, 'models/model_max_temp.joblib')
print("Max Temperature model trained and saved.")

# Model 2: Min Temperature
model_min = RandomForestRegressor(**common_model_params)
model_min.fit(X, y_min)
joblib.dump(model_min, 'models/model_min_temp.joblib')
print("Min Temperature model trained and saved.")

# Model 3: Humidity
model_hum = RandomForestRegressor(**common_model_params)
model_hum.fit(X, y_hum)
joblib.dump(model_hum, 'models/model_humidity.joblib')
print("Humidity model trained and saved.")

print("\n--- Model training complete! ---")
