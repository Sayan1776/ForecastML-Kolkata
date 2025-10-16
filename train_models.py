# ===============================================================
# MODEL TRAINING SCRIPT FOR KOLKATA WEATHER
# ===============================================================

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
# ADDED: Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Step 1: Load and Prepare the Kolkata Dataset
# ===============================================================
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


# ===============================================================
# Step 1.5: Exploratory Data Analysis (EDA) with Plots
# ===============================================================
print("\n--- Generating plots for the Kolkata dataset ---")
sns.set_style('whitegrid')

# Plot 1: High and Low Temperatures Over Time
plt.figure(figsize=(16, 6))
plt.title('High & Low Temperatures in Kolkata Over Time')
sns.lineplot(data=df['max_temp'], label='Max Temp (°C)', color='orange')
sns.lineplot(data=df['min_temp'], label='Min Temp (°C)', color='lightblue')
plt.xlabel('Date') # Explicitly label the X-axis
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Plot 2: Correlation Matrix
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix of Weather Variables')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Plot 3: Distribution of Each Variable with Clear Axis Labels
# df.hist() returns a NumPy array of axes, one for each subplot.
axes = df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Weather Variables in Kolkata')

# Iterate through each subplot (ax) and set the axis labels
for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel("Value of the Variable")
        ax.set_ylabel("Frequency (Number of Days)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ===============================================================
# Step 2: Feature Engineering (with Day of Year)
# ===============================================================
print("\n--- Engineering features with seasonality ---")

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


# ===============================================================
# Step 3: Train and Save All Three Models
# ===============================================================
print("\n--- Training and saving all three RandomForest models ---")

os.makedirs('models', exist_ok=True)
common_model_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}

# --- Model 1: Max Temperature ---
model_max = RandomForestRegressor(**common_model_params)
model_max.fit(X, y_max)
print("Max Temperature model trained.")
joblib.dump(model_max, 'models/model_max_temp.joblib')
print("Max Temperature model saved.")

# --- Model 2: Min Temperature ---
model_min = RandomForestRegressor(**common_model_params)
model_min.fit(X, y_min)
print("Min Temperature model trained.")
joblib.dump(model_min, 'models/model_min_temp.joblib')
print("Min Temperature model saved.")

# --- Model 3: Humidity ---
model_hum = RandomForestRegressor(**common_model_params)
model_hum.fit(X, y_hum)
print("Humidity model trained.")
joblib.dump(model_hum, 'models/model_humidity.joblib')
print("Humidity model saved.")

print("\n--- Model training complete. ---")

