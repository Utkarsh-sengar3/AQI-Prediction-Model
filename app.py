from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests as http_requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

API_KEY = "c858e10f19d8ddef51725ae43cb42dd9"


def get_aqi_category(aqi):
    """Return AQI category string based on value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    city = data.get("city", "").strip()

    if not city:
        return jsonify({"error": "City name is required."}), 400

    # Step 1: Geocode the city
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_response = http_requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data:
            return jsonify({"error": f"City '{city}' not found. Please try again."}), 404

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0].get("name", city)
        country = geo_data[0].get("country", "")
    except Exception as e:
        return jsonify({"error": f"Failed to geocode city: {str(e)}"}), 500

    # Step 2: Fetch historical AQI data (365 days / 1 year) in 30-day batches
    HISTORY_DAYS = 365
    BATCH_DAYS = 30

    daily_aqi = {}  # date_str -> list of aqi values for that day

    now = datetime.now()
    start_date = now - timedelta(days=HISTORY_DAYS)

    # Fetch in 30-day windows to minimize API calls (~12 calls instead of 365)
    current_start = start_date
    while current_start < now:
        current_end = min(current_start + timedelta(days=BATCH_DAYS), now)
        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())

        try:
            url = (
                f"http://api.openweathermap.org/data/2.5/air_pollution/history"
                f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={API_KEY}"
            )
            response = http_requests.get(url, timeout=30)
            resp_data = response.json()

            if "list" in resp_data:
                for item in resp_data["list"]:
                    item_dt = datetime.fromtimestamp(item["dt"])
                    day_key = item_dt.strftime("%Y-%m-%d")
                    aqi_val = item["main"]["aqi"]
                    if day_key not in daily_aqi:
                        daily_aqi[day_key] = []
                    daily_aqi[day_key].append(aqi_val)
        except Exception:
            pass

        current_start = current_end

    # Build sorted daily averages
    aqi_history = []
    dates = []

    for day_offset in range(HISTORY_DAYS, -1, -1):
        day = now - timedelta(days=day_offset)
        day_key = day.strftime("%Y-%m-%d")
        if day_key in daily_aqi:
            avg_aqi = np.mean(daily_aqi[day_key])
            aqi_scaled = (avg_aqi - 1) * 125
            aqi_history.append(float(aqi_scaled))
            dates.append(day_key)

    if len(aqi_history) < 30:
        return jsonify({"error": "Not enough historical data available for this city. Need at least 30 days."}), 500

    # Step 3: Prepare training data with larger window
    window_size = 14
    X_train = []
    y_train = []

    for i in range(len(aqi_history) - window_size):
        X_train.append(aqi_history[i : i + window_size])
        y_train.append(aqi_history[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Step 4: Train model (more estimators & depth for richer dataset)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_tr, y_tr)

    y_pred_test = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, y_pred_test))
    r2 = float(r2_score(y_te, y_pred_test))

    # Step 5: Predict next 5 days
    future_predictions = []
    current_window = list(aqi_history[-window_size:])

    for day_num in range(1, 6):
        pred = float(model.predict([current_window])[0])
        pred = max(0, pred)  # AQI can't be negative
        future_date = (datetime.now() + timedelta(days=day_num)).strftime("%Y-%m-%d")

        future_predictions.append(
            {
                "day": day_num,
                "date": future_date,
                "aqi": round(pred, 2),
                "category": get_aqi_category(pred),
            }
        )

        current_window.pop(0)
        current_window.append(pred)

    # Build full history (all 365 days for chart & download)
    full_history = []
    for d, a in zip(dates, aqi_history):
        full_history.append(
            {"date": d, "aqi": round(a, 2), "category": get_aqi_category(a)}
        )

    # Build display history (last 11 days only for dashboard widgets)
    DISPLAY_DAYS = 11
    history = full_history[-DISPLAY_DAYS:]

    return jsonify(
        {
            "city": city_name,
            "country": country,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "history": history,
            "full_history": full_history,
            "predictions": future_predictions,
            "model_metrics": {"r2": round(r2, 4), "mae": round(mae, 4)},
        }
    )


# ===== INDIA METRO CITIES ENDPOINT =====

METRO_CITIES = [
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
]


@app.route("/api/metro-cities")
def metro_cities():
    """Fetch current AQI, temperature, and humidity for Indian metro cities."""
    results = []

    for city in METRO_CITIES:
        city_data = {
            "name": city["name"],
            "lat": city["lat"],
            "lon": city["lon"],
            "aqi": None,
            "category": None,
            "temp": None,
            "humidity": None,
        }

        try:
            # Fetch current air pollution
            aqi_url = (
                f"http://api.openweathermap.org/data/2.5/air_pollution"
                f"?lat={city['lat']}&lon={city['lon']}&appid={API_KEY}"
            )
            aqi_resp = http_requests.get(aqi_url, timeout=10)
            aqi_data = aqi_resp.json()

            if "list" in aqi_data and aqi_data["list"]:
                raw_aqi = aqi_data["list"][0]["main"]["aqi"]
                # Scale from 1-5 index to US AQI scale
                scaled_aqi = round((raw_aqi - 1) * 125, 1)
                city_data["aqi"] = scaled_aqi
                city_data["category"] = get_aqi_category(scaled_aqi)
        except Exception:
            pass

        try:
            # Fetch current weather (temp + humidity)
            weather_url = (
                f"http://api.openweathermap.org/data/2.5/weather"
                f"?lat={city['lat']}&lon={city['lon']}&appid={API_KEY}&units=metric"
            )
            weather_resp = http_requests.get(weather_url, timeout=10)
            weather_data = weather_resp.json()

            if "main" in weather_data:
                city_data["temp"] = round(weather_data["main"]["temp"])
                city_data["humidity"] = weather_data["main"]["humidity"]
        except Exception:
            pass

        results.append(city_data)

    return jsonify({"cities": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
