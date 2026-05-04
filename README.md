<img width="1600" height="930" alt="image" src="https://github.com/user-attachments/assets/1ecb4c3d-a073-49f1-9b4a-71fc4882f0b6" /><p align="center">
  <img src="smartvia.png" alt="SmartVia VLC" width="500">
</p>

<p align="center">
  <strong>Real-time traffic congestion predictor for Valencia, Spain</strong><br>
  Powered by LightGBM &middot; Flask &middot; Leaflet.js
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-ML_Model-green?logo=lightgbm" alt="LightGBM">
  <img src="https://img.shields.io/badge/Flask-API-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Leaflet.js-Map-brightgreen?logo=leaflet" alt="Leaflet">
  <img src="https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws" alt="AWS">
</p>

---

## Overview

SmartVia VLC is a machine learning-based web application that predicts traffic congestion across **144 road segments** in Valencia, Spain. Users can select a date, time, weather condition, and special events to visualize predicted traffic states on an interactive map. We have developed a website, www.smartviavlc.com, and it is deployed via AWS.

This project was developed as part of **Proyecto III** in the **Grado en Ciencia de Datos** at the **Universitat Politecnica de Valencia (UPV)**.
<img src="smartvia.png" alt="SmartVia VLC" width="500">
## Features

- **ML-Powered Predictions** -- LightGBM model trained on historical traffic data with lag features and SMOTE balancing
- **Interactive Map** -- Leaflet.js map with color-coded road segments indicating congestion risk
- **Weather Integration** -- Predictions adjust for clear, cloudy, and rainy conditions
- **Event Awareness** -- Accounts for Fallas, school holidays, and marathons
- **Multi-language** -- English, Spanish, Valencian, and Chinese
- **Day/Night Mode** -- Toggle between light and dark map themes
- **Responsive Design** -- Works on desktop and mobile devices

## Architecture

```
User (Browser)
     |
     v
  Nginx (port 80) --> Gunicorn --> Flask API (app3.py)
                                      |
                              LightGBM Model (.txt)
                              Preprocessors (.joblib)
                              Historical Seeds (.joblib)
```

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| ML Model        | LightGBM (gradient boosting)        |
| Data Balancing  | SMOTE                               |
| Backend         | Flask + Gunicorn                    |
| Frontend        | HTML/CSS/JS + Leaflet.js            |
| Map Tiles       | Jawg Maps (Light & Dark)            |
| Deployment      | AWS EC2 + Nginx + Route 53          |

## Project Structure

```
.
├── app3.py                      # Flask API server
├── website3.html                # Frontend (map + controls)
├── datos_trafico.js             # Road segment polyline coordinates
├── smartvia-logo-light.svg      # App logo (SVG)
├── smartvia.png                 # App logo (PNG)
├── train_lightgbm_lag_v2.py     # Model training script
├── save_preprocessors_lag.py    # Generates .joblib preprocessors
├── split_data.py                # Train/test data split
├── lightgbm_lag_v2_model.txt    # Trained LightGBM model
├── historical.joblib            # Historical traffic probabilities
├── historical_fallback.joblib   # Fallback probabilities
├── le_denom.joblib              # Road name label encoder
├── le_day.joblib                # Day of week label encoder
├── scaler_lag.joblib            # Feature scaler
├── known_roads.joblib           # List of 144 known roads
├── DatasetFestivos.csv          # Holidays and events dataset
├── climatologia_2023_2025.csv   # Weather data (AEMET)
└── dataset_final.zip            # Full training dataset (compressed)
```

## Installation

### Local Setup

```bash
git clone https://github.com/jiale203/SmartVia-VLC---Valencia-Traffic-Predictor.git
cd smartvia-vlc

pip install flask flask-cors pandas numpy lightgbm joblib scikit-learn gunicorn

python app3.py
```

Visit `http://localhost:5002` in your browser.

### Production Deployment (AWS EC2)

```bash
gunicorn app3:app --bind 127.0.0.1:5002 --timeout 120 --workers 2
```

Configure Nginx as a reverse proxy on port 80 to forward requests to Gunicorn.

## Model Details

| Parameter          | Value                                     |
|--------------------|-------------------------------------------|
| Algorithm          | LightGBM (gradient boosted trees)         |
| Training Data      | ~400K samples (SMOTE balanced)            |
| Features           | 39 (temporal, weather, lag, events)        |
| Lag Features       | estado_lag_1, lag_2, lag_4, rolling_4/8    |
| Congestion Threshold | 0.8 probability                         |
| Macro F1 Score     | 0.8762                                    |
| Prediction Interval | 15 minutes                               |

### Key Features Used

| Feature              | Importance   |
|----------------------|-------------|
| estado_rolling_8     | Very High   |
| estado_lag_1         | Very High   |
| hour_sin / hour_cos  | High        |
| minutes              | High        |
| Road (encoded)       | Medium      |
| Weather (tmed, sol)  | Medium      |
| Events (Fallas, etc.)| Low-Medium  |

## Data Sources

- **Traffic Data** -- Ajuntament de Valencia open data portal (real-time traffic sensors)
- **Weather Data** -- AEMET (Agencia Estatal de Meteorologia)
- **Holiday/Event Data** -- Manually curated calendar of local events

## License

This project is licensed under the [GNU License](LICENSE).

## Authors
-Jiale Mao
-Vicente Llacer
-Miguel Carrañas
-Ivette 
-Lucia Fuentes
-Maria
