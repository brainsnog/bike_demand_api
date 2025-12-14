import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import math
from datetime import datetime
import pandas as pd
import altair as alt
import random
import base64

# Load the dataset
df_train = pd.read_csv("df_train.csv")

# --------------------------------------------------
# Fake station map data (static)
# --------------------------------------------------
# --------------------------------------------------
# Fake station map data (static, cached)
# --------------------------------------------------
N_STATIONS = 15

@st.cache_data
def get_station_map():
    np.random.seed(17)
    return pd.DataFrame({
        "station_id": range(N_STATIONS),
        "x": np.random.uniform(0, 100, N_STATIONS),
        "y": np.random.uniform(0, 100, N_STATIONS),
    })

station_map_df = get_station_map()



# --------------------------------------------------
# Load model and scaler (cached)
# --------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("model_lstm.h5", compile=False)
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------
st.set_page_config(page_title="Bike Demand Predictor", layout="wide")
st.title("🚴 Bike Demand Predictor")
st.write("Predict bike demand per station using weather and time features.")

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Input Features")

# --------------------------------------------------
# Calendar input for date
# --------------------------------------------------
selected_date = st.sidebar.date_input(
    "What Day Are You Traveling?",
    value=datetime.today(),
    key="calendar_date"
)

month = selected_date.month
weekday_numeric = selected_date.weekday()

# Automatically set season
if month in [3, 4, 5]:
    season = 1
elif month in [6, 7, 8]:
    season = 2
elif month in [9, 10, 11]:
    season = 3
else:
    season = 4

# --------------------------------------------------
# Hour input
# --------------------------------------------------
hour = st.sidebar.slider("Time of Travel", 0, 23, 12)

# --------------------------------------------------
# Historical averages
# --------------------------------------------------
monthly_stats = df_train.groupby("mnth")[["atemp", "hum", "windspeed"]].mean()

default_atemp = monthly_stats.loc[month, "atemp"] * 50
default_hum = monthly_stats.loc[month, "hum"] * 100
default_windspeed = monthly_stats.loc[month, "windspeed"]

# --------------------------------------------------
# Weather sliders
# --------------------------------------------------
atemp = st.sidebar.slider("Temperature (°C)", -10, 40, int(default_atemp))

hum_input = st.sidebar.slider("Humidity (%)", 0, 100, int(default_hum))
hum = hum_input / 100

windspeed = st.sidebar.slider(
    "Wind Speed (km/h)", 0.0, 50.0, float(default_windspeed)
)

# --------------------------------------------------
# Station ID (synced with map)
# --------------------------------------------------
if "selected_station_id" not in st.session_state:
    st.session_state.selected_station_id = 0

station_id = st.sidebar.number_input(
    "Station ID",
    min_value=0,
    max_value=N_STATIONS - 1,
    value=st.session_state.selected_station_id,
    key="station_id_input"
)

st.session_state.selected_station_id = station_id

# --------------------------------------------------
# Other categorical inputs
# --------------------------------------------------
yr = 1

holiday_title = st.sidebar.selectbox("Holiday", ["No", "Yes"])
holiday = 1 if holiday_title == "Yes" else 0

is_weekend = weekday_numeric >= 5
workingday = 0 if (is_weekend or holiday == 1) else 1

weather_options = {
    "Clear / Few clouds / Partly cloudy": 1,
    "Mist / Cloudy / Few clouds": 2,
    "Light Snow / Light Rain / Scattered clouds": 3,
    "Heavy Rain / Ice Pellets / Thunderstorm / Snow / Fog": 4
}
weathersit_title = st.sidebar.selectbox(
    "Weather Situation", list(weather_options.keys())
)
weathersit = weather_options[weathersit_title]

# --------------------------------------------------
# Cyclical features
# --------------------------------------------------
hour_sin = math.sin(2 * math.pi * hour / 24)
hour_cos = math.cos(2 * math.pi * hour / 24)
month_sin = math.sin(2 * math.pi * month / 12)
month_cos = math.cos(2 * math.pi * month / 12)

# --------------------------------------------------
# Layout
# --------------------------------------------------
left_col, right_col = st.columns([2, 3])


# =========================
# LEFT COLUMN – PREDICTION
# =========================
with left_col:
    if st.button("Predict Bike Demand"):
        try:
            feature_row = [
                season,
                yr,
                holiday,
                weekday_numeric,
                workingday,
                weathersit,
                atemp / 50.0,
                hum,
                windspeed / 67.0,
                month_sin,
                month_cos,
                hour_sin,
                hour_cos,
                station_id
            ]

            feature_array = np.array(feature_row, dtype=float).reshape(1, 14)
            feature_scaled = scaler.transform(feature_array)
            sequence_scaled = np.repeat(
                feature_scaled[np.newaxis, :, :], 24, axis=1
            )

            prediction = model.predict(sequence_scaled, verbose=0)
            predicted_demand = max(0, int(round(prediction[0, 0])))

            st.success(
                f"### 🚲 Predicted Demand: **{predicted_demand} bikes**"
            )

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# =========================
# RIGHT COLUMN – MAP + TREND
# =========================
with right_col:
    st.markdown("### 🗺️ Station Map")

    station_map_df["selected"] = (
        station_map_df["station_id"] == st.session_state.selected_station_id
    )

    # Background image
    # Background as an image stretched over the coordinate system
    with open("map_of_london.png", "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    img_url = f"data:image/png;base64,{img_b64}"

    fixed_scale = alt.Scale(domain=[0, 100], nice=False)  # disables rescaling

    # Full-coordinate background
    background = alt.Chart(pd.DataFrame([{"x": 0, "y": 0, "x2": 100, "y2": 100}])).mark_image(
        url=img_url
    ).encode(
        x=alt.X('x:Q', scale=fixed_scale, axis=None),
        x2='x2:Q',
        y=alt.Y('y:Q', scale=fixed_scale, axis=None),
        y2='y2:Q'
    )


    stations = (
        alt.Chart(station_map_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("x:Q", axis=None, scale=fixed_scale),
            y=alt.Y("y:Q", axis=None, scale=fixed_scale),
            color=alt.condition(
                alt.datum.selected,
                alt.value("darkred"),
                alt.value("red")
            ),
            tooltip=[alt.Tooltip("station_id:O", title="Station ID")]
        )
        .add_selection(
            alt.selection_single(fields=["station_id"], empty="none")
        )
    )


    map_chart = background + stations
    st.altair_chart(map_chart, width=500, height=500)


st.subheader("📈 Demand Trend (Next 6 Hours)")

rolling_window = []
for t in range(24):
    h = (hour - (23 - t)) % 24
    rolling_window.append([
        season, yr, holiday, weekday_numeric, workingday, weathersit,
        atemp / 50.0, hum, windspeed / 67.0,
        month_sin, month_cos,
        math.sin(2 * math.pi * h / 24),
        math.cos(2 * math.pi * h / 24),
        station_id
    ])

rolling_window = np.array(rolling_window)
trend_predictions = []
current_window = rolling_window.copy()

for step in range(6):
    window_scaled = scaler.transform(current_window).reshape(1, 24, 14)
    pred = model.predict(window_scaled, verbose=0)
    trend_predictions.append(float(pred[0, 0]))

    next_hour = (hour + step + 1) % 24
    next_row = [
        season, yr, holiday, weekday_numeric, workingday, weathersit,
        atemp / 50.0, hum, windspeed / 67.0,
        month_sin, month_cos,
        math.sin(2 * math.pi * next_hour / 24),
        math.cos(2 * math.pi * next_hour / 24),
        station_id
    ]

    current_window = np.vstack([current_window[1:], next_row])

hours_ahead = [(hour + i + 1) % 24 for i in range(6)]
df = pd.DataFrame({
    "Hour": hours_ahead,
    "Predicted Demand": trend_predictions
})

baseline = df["Predicted Demand"].iloc[0]
df["Indexed Demand"] = (df["Predicted Demand"] / baseline - 1) * 100

chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x="Hour:O",
        y="Indexed Demand:Q",
        tooltip=["Hour", "Indexed Demand"]
    )
    .properties(height=400)
)

st.altair_chart(chart, width="stretch")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("LSTM Bike Demand Predictor • Streamlit App")

