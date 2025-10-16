# ===============================================================
# SCRIPT 2: LOAD MODELS AND CREATE FORECAST
# ===============================================================
# Purpose: To load the pre-trained models, fetch live weather
#          data, and generate a 7-day forecast.
# Run this script anytime you want a new forecast.
# ===============================================================
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ===============================================================
# Step 1: Load the Pre-Trained Models
# ===============================================================
print("--- Loading pre-trained models ---")
try:
    model_max = joblib.load('models/model_max_temp.joblib')
    model_min = joblib.load('models/model_min_temp.joblib')
    model_hum = joblib.load('models/model_humidity.joblib')
    print("Models loaded successfully.")
except FileNotFoundError:
    print("\nError: Model files not found!")
    print("Please run the 'train_models.py' script first to train and save the models.")
    exit()

# ===============================================================
# Step 2: Live Forecasting Logic
# ===============================================================
print("\n--- Preparing for live forecast ---")

def get_current_weather(api_key, city):
    """Fetches the 24-hour high, low, and current humidity."""
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = f"{base_url}appid={api_key}&q={city}&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    
    if data["cod"] == "200":
        forecast_list = data["list"]
        next_24h = forecast_list[:8]
        max_temp = max(item['main']['temp_max'] for item in next_24h)
        min_temp = min(item['main']['temp_min'] for item in next_24h)
        current_humidity = forecast_list[0]['main']['humidity']
        today_date = pd.to_datetime('today').normalize()
        
        print(f"\nLive data fetched for {city}: High={max_temp:.2f}°C, Low={min_temp:.2f}°C, Humidity={current_humidity}%")
        
        day_of_year = today_date.dayofyear
        
        # NOTE: The columns list must match what the model was trained on
        live_data = pd.DataFrame(
            [[max_temp, min_temp, current_humidity, day_of_year]],
            columns=['max_temp', 'min_temp', 'humidity', 'day_of_year'],
            index=[today_date]
        )
        return live_data
    else:
        print(f"City '{city}' not found or API error! Response: {data}")
        return None

# --- Main Forecasting Execution ---
API_KEY = os.getenv("API_KEY")
CITY_NAME = "Kolkata"

start_data = get_current_weather(API_KEY, CITY_NAME)

if start_data is not None:
    print(f"\n--- Generating 7-Day Detailed Forecast for {CITY_NAME} ---")
    
    future_max_temps, future_min_temps, future_humidities, future_dates = [], [], [], []
    last_known_data = start_data.copy()

    for _ in range(7):
        pred_max = model_max.predict(last_known_data)[0]
        pred_min = model_min.predict(last_known_data)[0]
        pred_hum = model_hum.predict(last_known_data)[0]

        future_max_temps.append(pred_max)
        future_min_temps.append(pred_min)
        future_humidities.append(pred_hum)
        
        last_date = last_known_data.index[0]
        next_date = last_date + pd.Timedelta(days=1)
        future_dates.append(next_date)
        
        next_day_of_year = next_date.dayofyear
        last_known_data = pd.DataFrame(
            [[pred_max, pred_min, pred_hum, next_day_of_year]],
            columns=['max_temp', 'min_temp', 'humidity', 'day_of_year'],
            index=[next_date]
        )
    
    # --- Step 3: Visualize the Detailed Forecast ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=future_dates, y=future_max_temps, name='Predicted High (°C)', mode='lines+markers', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_min_temps, name='Predicted Low (°C)', mode='lines+markers', line=dict(color='lightblue')))
    fig.add_trace(go.Bar(x=future_dates, y=future_humidities, name='Predicted Humidity (%)', opacity=0.5, marker_color='lightgreen'), secondary_y=True)
    fig.update_layout(title_text=f'7-Day Detailed Forecast for {CITY_NAME}', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Humidity (%)", secondary_y=True, showgrid=False)
    
    print("\nDisplaying forecast plot...")
    fig.show()

else:
    print("\nCould not retrieve live weather data. Halting forecast.")
