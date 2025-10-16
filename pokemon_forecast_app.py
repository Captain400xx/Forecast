import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pokémon Card Restock Forecasts", layout="wide")
st.title("🃏 Pokémon Card Restock Forecasts by Retailer")
st.markdown("Forecasting future restock times based on historical drop data.")

# --- Load dataset ---
df = pd.read_csv("pokemon_drops.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])

# --- Fill missing hours for each retailer ---
all_retailers = df["Retailer"].unique()
filled_data = []

for retailer in all_retailers:
    retailer_df = df[df["Retailer"] == retailer].set_index("DateTime")
    full_range = pd.date_range(start=retailer_df.index.min(), end=retailer_df.index.max(), freq="H")
    retailer_df = retailer_df.reindex(full_range, fill_value=0)
    retailer_df["Retailer"] = retailer
    retailer_df = retailer_df.reset_index().rename(columns={"index": "DateTime"})
    filled_data.append(retailer_df)

df = pd.concat(filled_data)

# --- Sidebar filters ---
retailers = df["Retailer"].unique().tolist()
selected_retailers = st.sidebar.multiselect("Select retailers to forecast:", retailers, default=retailers)
forecast_horizon = st.sidebar.slider("Forecast horizon (hours ahead)", 24, 168, 72)

# --- Forecast and plot ---
fig = go.Figure()

for retailer in selected_retailers:
    data = df[df["Retailer"] == retailer].rename(columns={"DateTime": "ds", "Count": "y"})
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(data)
    
    future = model.make_future_dataframe(periods=forecast_horizon, freq="H")
    forecast = model.predict(future)
    
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name=retailer,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Predicted: %{y:.1f}"
    ))

# --- Layout ---
fig.update_layout(
    title="Predicted Pokémon Card Restocks",
    xaxis_title="Date and Time",
    yaxis_title="Predicted Restock Count",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
**Tips**
- Use the sidebar to toggle which retailers are shown.
- Adjust the forecast horizon to predict more or fewer hours ahead.
- Hover on lines for exact predicted drop counts.
""")
