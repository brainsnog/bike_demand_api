# ⬡ BikeFlow — LSTM Bike Demand Forecasting

> Real-time demand prediction for a 15-station synthetic Capital Bikeshare network, powered by a trained LSTM model and deployed as a professional-grade Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

BikeFlow is an end-to-end machine learning project that trains a Long Short-Term Memory (LSTM) neural network on Capital Bikeshare data from Washington DC (2011–2012) and deploys it as an interactive forecasting dashboard. The system predicts hourly bike demand across 15 synthetic stations and provides a 6-hour rolling demand trend — all presented through a dark-mode ops dashboard aesthetic inspired by Uber, Lime, and Google Maps.

The project demonstrates the full ML deployment pipeline: data preprocessing, feature engineering with cyclical encodings, LSTM sequence modelling, model serialisation, and production-grade UI design — in a single deployable application.

---

## Demo

| Feature | Description |
|---|---|
| **Point Forecast** | Predict demand at any station for a chosen date, time, and weather scenario |
| **6-Hour Trend** | Rolling LSTM forecast showing indexed demand trajectory for the next 6 hours |
| **Interactive Map** | Click-to-select stations on a live Washington DC pydeck map |
| **Live Weather** | Real-time DC conditions fetched from Open-Meteo API (no key required) |
| **KPI Summary** | At-a-glance summary row showing time, station, season, and current conditions |

---

## Project Structure

```
bikeflow/
├── app.py                  # Streamlit dashboard — all UI logic lives here
├── model_lstm.h5           # Trained LSTM model (TensorFlow/Keras)
├── scaler.joblib           # Fitted feature scaler (scikit-learn)
├── df_train.csv            # Training dataset (Capital Bikeshare 2011–2012)
├── notebooks/              # Exploration, feature engineering, model training
├── src/                    # Supporting modules and data pipeline utilities
└── requirements.txt        # Python dependencies
```

---

## Model

### Architecture

The model is a sequential LSTM network trained to predict hourly bike demand from a 14-feature input vector. Input sequences are constructed with a 24-step lookback window (representing the prior 24 hours of context) before each prediction.

```
Input shape:  (batch, 24 timesteps, 14 features)
Architecture: LSTM → Dense output
Loss:         Mean Squared Error
Optimiser:    Adam
```

### Feature Engineering

Raw input variables are transformed before being passed to the model:

| Feature | Transformation | Notes |
|---|---|---|
| `hour` | `sin(2π·h/24)`, `cos(2π·h/24)` | Cyclical encoding — preserves 23→00 continuity |
| `month` | `sin(2π·m/12)`, `cos(2π·m/12)` | Cyclical encoding — preserves Dec→Jan continuity |
| `atemp` | `/50.0` | Normalised apparent temperature (°C) |
| `hum` | `/100.0` | Normalised humidity (%) |
| `windspeed` | `/67.0` | Normalised wind speed (km/h) |
| `season` | Integer 1–4 | Spring / Summer / Autumn / Winter |
| `holiday` | Binary 0/1 | Public holiday flag |
| `workingday` | Binary 0/1 | Derived: not weekend and not holiday |
| `weathersit` | Integer 1–4 | Weather condition category |
| `station_id` | Integer 0–14 | Synthetic station identifier |
| `yr` | Fixed at `1` | Year index (0=2011, 1=2012); fixed to 2012 peak year |
| `weekday` | Integer 0–6 | Day of week (Monday=0) |

Cyclical encodings are the critical engineering decision here. A naive numeric hour encoding (e.g. hour=23 followed by hour=0) creates a discontinuity the model has to learn across. Sine/cosine encoding places adjacent hours near each other in 2D feature space — the model learns temporal patterns correctly without any explicit calendar logic.

### Training Data

- **Source:** [Capital Bikeshare System Data](https://capitalbikeshare.com/system-data) (Washington DC)
- **Period:** January 2011 – December 2012
- **Granularity:** Hourly
- **Stations:** The original dataset does not contain station-level data. 15 synthetic stations were constructed from the city-wide hourly counts using a seeded random distribution, producing realistic variance in demand patterns across a simulated network. Station coordinates are placed at authentic Capital Bikeshare service zones across DC.

---

## Dashboard

### Design System

The UI was designed from scratch using CSS injected via `st.markdown()`. The design language is intentionally distinct — cycling brand aesthetics rather than generic data dashboard defaults.

| Token | Value | Usage |
|---|---|---|
| Background base | `#0D0F0D` | Page background |
| Surface | `#131613` | Sidebar |
| Card | `#181C18` | Component containers |
| Accent | `#C8F135` | Electric lime — Lime Bikes brand colour |
| Accent dim | `#8AAD1E` | Section labels, secondary accent |
| Text primary | `#E8EDE8` | Body text |
| Text muted | `#506050` | Labels, hints |
| Font display | Barlow Condensed 800 | Headers, labels |
| Font data | Space Mono | Numbers, values, map tooltips |
| Font body | Barlow | Descriptive copy |

### Key UI Components

**Hero Header** — full-width banner with BikeFlow wordmark, animated pulsing `MODEL ACTIVE` status indicator, and vertical line grid overlay.

**Sidebar** — three named control sections (Date & Time · Live Weather · Station & Trip), each with an icon label and horizontal rule divider. Live weather is fetched from Open-Meteo and displayed as a 3-column grid card with real-time temperature, humidity, and wind speed.

**Prediction Card** — large Space Mono demand number with text glow, 2×2 context grid (station, time, day, season), and a `cardReveal` fade-in animation on render. Displays an idle placeholder state before the first forecast is run.

**Interactive Map** — pydeck `ScatterplotLayer` with three layers: muted teal unselected stations, lime accent selected station, translucent pulse ring. Dark basemap via Carto dark-matter (no API key). Click-to-select updates session state and re-centres the view.

**KPI Row** — four equal-width cards rendered below the map/prediction columns: Forecast Time · Active Station · Season · Conditions.

**Trend Chart** — Altair four-layer composition: dashed zero reference line, translucent area fill, lime line, and interactive circle points. Fully dark-themed via `.configure()`. Shows indexed demand (% change from baseline) across the next 6 hours.

**Footer** — two-column layout: BikeFlow wordmark + technical attribution stack (Model / Data / Weather), and a pill badge with build credit.

---

## Live Weather Integration

Weather inputs (temperature, humidity, wind speed) are fetched live from [Open-Meteo](https://open-meteo.com/) — a free, no-key-required weather API — and passed directly into the model's feature vector. This means every prediction reflects actual current DC conditions rather than manual user estimates.

```python
@st.cache_data(ttl=1800)  # Refreshes every 30 minutes
def fetch_dc_weather():
    # Fetches apparent_temperature, relative_humidity_2m, wind_speed_10m
    # Falls back to DC annual averages on failure
    ...
```

Graceful fallback: if the API is unreachable (offline environment, timeout), the app substitutes DC annual averages and surfaces a red "Offline — using fallback" indicator in the sidebar. The app never crashes.

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bikeflow.git
cd bikeflow

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

---

## Requirements

Key dependencies (see `requirements.txt` for pinned versions):

```
streamlit
tensorflow
keras
numpy
pandas
scikit-learn
joblib
altair
pydeck
requests
```

---

## Usage

1. **Select a date** using the date picker in the sidebar. Season is inferred automatically.
2. **Set the hour** using the time slider (0–23).
3. **Review live weather** — current DC conditions are shown automatically and fed to the model.
4. **Select a station** by clicking any dot on the Washington DC map. The sidebar updates to reflect your selection.
5. **Choose trip context** — set the Holiday flag and Weather Situation from the dropdowns.
6. **Run the forecast** — press the lime **Run Demand Forecast** button to generate a point prediction.
7. **Read the trend** — the 6-hour demand trend chart below updates automatically with every change to the sidebar inputs.

---

## Technical Decisions

### Why LSTM for demand forecasting?

Bike demand is a temporal sequence problem with strong daily and weekly periodicity. LSTM networks excel at capturing long-range dependencies in sequential data — the 24-step lookback window allows the model to condition each prediction on a full day of prior context. A feed-forward baseline would treat each hour independently and miss the autocorrelation structure.

### Why cyclical feature encoding?

Standard numeric encoding of hours (0–23) creates an artificial discontinuity: hour 23 and hour 0 are numerically far apart but temporally adjacent. Projecting onto a unit circle with sine/cosine pairs removes this discontinuity — the model sees adjacent hours as similar without any hand-crafted calendar logic.

### Why pydeck over Altair for the map?

The original dashboard used an Altair `mark_image` chart with a static London map `.png` as a background — a workaround that was broken, geographically incorrect, and aesthetically generic. Pydeck is Uber's open-source WebGL geospatial library, ships with Streamlit, and requires no API key when used with Carto tiles. It renders authentic Washington DC geography with a dark basemap that matches the dashboard aesthetic natively.

### Why replace weather sliders with live data?

The model was trained on real DC weather data. Asking users to manually estimate temperature and humidity introduces error and cognitive friction. Fetching live conditions from Open-Meteo means the model's inputs are grounded in reality — the prediction is for the actual current DC environment, not a hypothetical one.

### Why synthesise stations?

The public Capital Bikeshare dataset aggregates demand to the city level; no station-level data is available for 2011–2012. Synthesising 15 stations from the aggregate allows the model to support a realistic station-selection interface without requiring proprietary data. The station ID feature (`0–14`) gives the model enough signal to learn demand variance across the synthetic network.

---

## Acknowledgements

- **Dataset:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — Fanaee-T, H. & Gama, J. (2014)
- **Weather API:** [Open-Meteo](https://open-meteo.com/) — Free, open-source weather API
- **Basemap:** [Carto Dark Matter](https://carto.com/basemaps/) — Free tile layer, no API key required
- **Map rendering:** [pydeck](https://deckgl.readthedocs.io/) — Uber's WebGL geospatial visualisation library

---

## Licence

MIT — see `LICENSE` for details.
