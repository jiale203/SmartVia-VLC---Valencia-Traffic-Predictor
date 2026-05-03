from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

app = Flask(__name__)
CORS(app)

# ── 1. Load model ──
print("Loading model...")
model = lgb.Booster(model_file="lightgbm_lag_v2_model.txt")

# ── 2. Load pre-fitted artifacts (from save_preprocessors_lag.py) ──
historical = joblib.load("historical.joblib")
historical_fallback = joblib.load("historical_fallback.joblib")
le_denom = joblib.load("le_denom.joblib")
le_day = joblib.load("le_day.joblib")
scaler = joblib.load("scaler_lag.joblib")
known_roads = set(joblib.load("known_roads.joblib"))

continuous_cols = ["tmed", "prec", "tmin", "tmax", "sol", "minutes",
                   "hour_sin", "hour_cos", "month_sin", "month_cos",
                   "day_sin", "day_cos", "estado_rolling_4", "estado_rolling_8"]

print(f"Model loaded. {len(known_roads)} known roads.")


def get_historical_seed(road, day_of_week, minute_slot):
    match = historical[
        (historical["road"] == road) &
        (historical["day_of_week"] == day_of_week) &
        (historical["minute_slot"] == minute_slot)
    ]
    if len(match) > 0:
        return match["hist_prob"].values[0]
    fallback = historical_fallback[
        (historical_fallback["road"] == road) &
        (historical_fallback["minute_slot"] == minute_slot)
    ]
    if len(fallback) > 0:
        return fallback["hist_prob_fallback"].values[0]
    return 0.02


def build_features(road, dt, weather, lag_1, lag_2, lag_4, rolling_4, rolling_8, count_4):
    minutes = dt.hour * 60 + dt.minute
    month = dt.month
    day = dt.day
    week_of_year = dt.isocalendar()[1]
    is_weekend = 1 if dt.weekday() in [5, 6] else 0
    is_rush_hour = 1 if (7 <= dt.hour <= 9) or (17 <= dt.hour <= 20) else 0
    is_night = 1 if (dt.hour >= 22 or dt.hour <= 6) else 0

    day_names_map = {
        0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves",
        4: "viernes", 5: "sábado", 6: "domingo"
    }
    day_name = day_names_map.get(dt.weekday(), "lunes")

    summer = 1 if month in [6, 7, 8] else 0
    winter = 1 if month in [12, 1, 2] else 0
    autumn = 1 if month in [9, 10, 11] else 0
    spring = 1 if month in [3, 4, 5] else 0

    features = {
        "Denominació_/_Denominación": le_denom.transform([road])[0],
        "tmed": weather.get("tmed", 20.0),
        "prec": weather.get("prec", 0.0),
        "tmin": weather.get("tmin", 15.0),
        "tmax": weather.get("tmax", 25.0),
        "sol": weather.get("sol", 8.0),
        "Day_of_week": le_day.transform([day_name])[0],
        "Public_holiday": weather.get("public_holiday", 0),
        "School_holiday": weather.get("school_holiday", 0),
        "Festival_de_les_arts": 0,
        "Davis_Cup": 0,
        "Elections": 0,
        "Demonstrations": 0,
        "Marathons": weather.get("marathon", 0),
        "Fallas": weather.get("fallas", 0),
        "Mascletá/Crida": weather.get("mascleta", 0),
        "University_Entrance_Exams": 0,
        "BigSound_Concerts": 0,
        "Roig_Arena_Events": 0,
        "Football_Matches": weather.get("football", 0),
        "San_Juan": 0,
        "Summer": summer,
        "Winter": winter,
        "Autumn": autumn,
        "Spring": spring,
        "estado_lag_1": lag_1,
        "estado_lag_2": lag_2,
        "estado_lag_4": lag_4,
        "estado_rolling_4": rolling_4,
        "estado_rolling_8": rolling_8,
        "estado_count_4": count_4,
        "minutes": minutes,
        "hour_sin": np.sin(2 * np.pi * minutes / 1440),
        "hour_cos": np.cos(2 * np.pi * minutes / 1440),
        "month": month,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day": day,
        "day_sin": np.sin(2 * np.pi * day / 31),
        "day_cos": np.cos(2 * np.pi * day / 31),
        "week_of_year": week_of_year,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_night": is_night,
    }
    result = pd.DataFrame([features])
    result[continuous_cols] = scaler.transform(result[continuous_cols])
    return result


def get_historical_lags(road, dt, interval_minutes=15):
    seeds = []
    for i in range(1, 9):
        t = dt - pd.Timedelta(minutes=interval_minutes * i)
        seed = get_historical_seed(road, t.weekday(), t.hour * 60 + t.minute)
        seeds.append(1 if seed > 0.5 else 0)

    lag_1 = seeds[0]
    lag_2 = seeds[1]
    lag_4 = seeds[3]
    rolling_4 = np.mean(seeds[:4])
    rolling_8 = np.mean(seeds[:8])
    count_4 = sum(seeds[:4])
    return lag_1, lag_2, lag_4, rolling_4, rolling_8, count_4


def predict_road(road, start_time, weather, interval_minutes=15, num_steps=8):
    predictions = []
    for step in range(num_steps):
        current_time = start_time + pd.Timedelta(minutes=interval_minutes * step)

        lag_1, lag_2, lag_4, rolling_4, rolling_8, count_4 = get_historical_lags(
            road, current_time, interval_minutes)

        X = build_features(road, current_time, weather,
                          lag_1, lag_2, lag_4, rolling_4, rolling_8, count_4)

        prob = float(model.predict(X)[0])
        state = 1 if prob > 0.8 else 0

        predictions.append({
            "time": current_time.strftime("%H:%M"),
            "probability": round(prob, 4),
            "state": state,
        })

    return predictions


@app.route("/")
def index():
    return send_from_directory(".", "website3.html")

@app.route("/datos_trafico.js")
def serve_js():
    return send_from_directory(".", "datos_trafico.js")

@app.route("/smartvia-logo-light.svg")
def serve_logo():
    return send_from_directory(".", "smartvia-logo-light.svg")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    datetime_str = data.get("datetime", "2025-06-15T08:00")
    weather_condition = data.get("weather", "clear")
    duration_hours = data.get("duration", 2)

    start_time = pd.Timestamp(datetime_str)
    num_steps = int(duration_hours * 60 / 15)

    weather_presets = {
        "clear": {"tmed": 25.0, "prec": 0.0, "tmin": 18.0, "tmax": 32.0, "sol": 10.0},
        "cloudy": {"tmed": 20.0, "prec": 0.0, "tmin": 15.0, "tmax": 25.0, "sol": 4.0},
        "rain":  {"tmed": 15.0, "prec": 5.0, "tmin": 12.0, "tmax": 18.0, "sol": 2.0},
    }
    weather = weather_presets.get(weather_condition, weather_presets["clear"])

    weather["school_holiday"] = data.get("school_holiday", 0)
    weather["football"] = data.get("football", 0)
    weather["marathon"] = data.get("marathon", 0)

    if data.get("fallas", 0):
        weather["fallas"] = 1
        weather["mascleta"] = 1
    else:
        month = start_time.month
        day = start_time.day
        if month == 3 and 15 <= day <= 19:
            weather["fallas"] = 1
            weather["mascleta"] = 1

    results = {}
    for road in known_roads:
        preds = predict_road(road, start_time, weather, num_steps=num_steps)
        avg_prob = np.mean([p["probability"] for p in preds])
        max_prob = max(p["probability"] for p in preds)
        results[road] = {
            "probability": round(avg_prob, 4),
            "max_probability": round(max_prob, 4),
            "state": 1 if avg_prob > 0.8 else 0,
            "timeline": preds,
        }

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=False, port=5002)
