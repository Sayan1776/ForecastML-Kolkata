Detailed Weather Forecast Predictor for Kolkata

    This project uses machine learning to provide a detailed 7-day weather forecast for Kolkata, predicting the maximum temperature, minimum temperature, and relative humidity. It leverages historical weather data and a live API to generate real-time predictions.

    The system is built on a professional two-script workflow: one script for training the models and another for generating live forecasts.

Key Features

    Multi-Output Forecast: Predicts three key weather variables: high temp, low temp, and humidity.

    Advanced Model: Uses a RandomForestRegressor for each variable to capture complex weather patterns more accurately than simple linear models.

    Seasonal Awareness: Incorporates the day of the year as a feature to better understand and predict seasonal changes.

    Live Data Integration: Fetches real-time weather data from the OpenWeatherMap API to provide an up-to-date starting point for its forecast.

    Interactive Visualization: Displays the final 7-day forecast in an interactive Plotly graph.

    Secure API Key Handling: Uses a .env file to keep API keys safe and out of version control.

Final Output

The script generates an interactive plot showing the 7-day forecast.

(To add your own image, upload a screenshot to a service like Imgur and paste the direct link below.)

Project Structure

WeatherPredictor/
├── data/
│   └── Kolkata_weather_history.csv
├── models/
│   ├── model_humidity.joblib
│   ├── model_max_temp.joblib
│   └── model_min_temp.joblib
├── train_models.py
├── forecast.py
├── requirements.txt
├── README.md
└── .gitignore


Setup and Installation

1. Clone the Repository

git clone [https://github.com/your-username/Weather-Forecast-Predictor.git](https://github.com/your-username/Weather-Forecast-Predictor.git)
cd Weather-Forecast-Predictor


2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt


4. Get the Dataset

Download the historical weather data from Kaggle.

Place the kolkata_weather_history.csv file inside the data/ folder.

5. Set Up Your API Key (Crucial)

Create a free account at OpenWeatherMap to get an API key.

Create a file named .env in the root of the project folder.

Add your API key to the .env file like this:

API_KEY="your_actual_api_key_here"


The .gitignore file is already configured to keep this file private.

How to Use the Project

The project is split into two main scripts for a professional workflow.

Step 1: Train the Models (Run Once)

First, you need to run the training script. This will process the historical data, train the three machine learning models, and save them into the models/ folder.

python train_models.py


This step can be re-run whenever you want to retrain the models on new or different historical data.

Step 2: Get a Live Forecast (Run Anytime)

Once the models are trained and saved, you can run the forecast script anytime you want an up-to-date 7-day forecast. This script is fast because it loads the pre-trained models.

python forecast.py


This will fetch live data for Kolkata and generate the interactive forecast plot.