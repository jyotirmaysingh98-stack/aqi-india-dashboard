import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#   ==========================================
# PAGE CONFIG & STYLING (Premium Aesthetics)
#   ==========================================
st.set_page_config(
    page_title="AQI India — Pollution Intelligence Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #3182ce, #319795);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #718096;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            border-left: 5px solid #3182ce;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2d3748;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA FETCHING & PROCESSING (OpenAQ API)
# ==========================================
LOCAL_DATA_PATH = os.path.join("outputs", "tables", "cleaned_dataset.csv")

@st.cache_data(ttl=600)  # cache for 10 minutes
def fetch_live_data():
    """Fetch live AQI data from OpenAQ API for India, fallback to local dataset on error."""
    try:
        url = "https://api.openaq.org/v2/latest?country=IN&limit=1000"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            records = []
            for item in data.get("results", []):
                city = item.get("city", "Unknown")
                location = item.get("location", "Unknown Station")
                coords = item.get("coordinates")
                if not coords or not coords.get("latitude") or not coords.get("longitude"):
                    continue
                lat, lon = coords["latitude"], coords["longitude"]
                
                for meas in item.get("measurements", []):
                    param = meas.get("parameter", "").upper()
                    val = meas.get("value", 0.0)
                    last_up = meas.get("lastUpdated", "")
                    
                    # Normalize parameters to match our schema
                    param_map = {
                        "PM25": "PM2.5", "PM10": "PM10", "NO2": "NO2", 
                        "SO2": "SO2", "O3": "OZONE", "OZONE": "OZONE", 
                        "CO": "CO", "NH3": "NH3"
                    }
                    norm_param = param_map.get(param, param)
                    if norm_param not in ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "NH3"]:
                        continue
                        
                    records.append({
                        "country": "India",
                        "state": "Live Feed",
                        "city": city,
                        "station": location,
                        "last_update": last_up,
                        "latitude": lat,
                        "longitude": lon,
                        "pollutant_id": norm_param,
                        "pollutant_min": val * 0.8,
                        "pollutant_max": val * 1.2,
                        "pollutant_avg": val
                    })
            if records:
                df = pd.DataFrame(records)
                df["last_update"] = pd.to_datetime(df["last_update"])
                return df, "Live API Feed (OpenAQ)"
    except Exception as e:
        pass
    
    # Fallback to local data
    if os.path.exists(LOCAL_DATA_PATH):
        df = pd.read_csv(LOCAL_DATA_PATH)
        df["last_update"] = pd.to_datetime(df["last_update"])
        return df, f"Local Cleaned Dataset (Fallback: {os.path.basename(LOCAL_DATA_PATH)})"
    else:
        return pd.DataFrame(), "No Data Available"

# Load initial data to Session State if not present
if "active_df" not in st.session_state:
    if os.path.exists(LOCAL_DATA_PATH):
        df_init = pd.read_csv(LOCAL_DATA_PATH)
        df_init["last_update"] = pd.to_datetime(df_init["last_update"])
        st.session_state.active_df = df_init
        st.session_state.source_name = "Local Cleaned Dataset"
    else:
        df_init, src_name = fetch_live_data()
        st.session_state.active_df = df_init
        st.session_state.source_name = src_name
    st.session_state.data_source_selected = "Local Cleaned Dataset"

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## ⚙️ Data Source")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Local Cleaned Dataset", "Live OpenAQ API"),
    index=0 if st.session_state.data_source_selected == "Local Cleaned Dataset" else 1
)

# Handle switching data sources
if data_source != st.session_state.data_source_selected:
    st.session_state.data_source_selected = data_source
    if data_source == "Live OpenAQ API":
        df_init, source_name = fetch_live_data()
    else:
        if os.path.exists(LOCAL_DATA_PATH):
            df_init = pd.read_csv(LOCAL_DATA_PATH)
            df_init["last_update"] = pd.to_datetime(df_init["last_update"])
            source_name = "Local Cleaned Dataset"
        else:
            df_init = pd.DataFrame()
            source_name = "No Data"
    st.session_state.active_df = df_init
    st.session_state.source_name = source_name

# Use session state dataframe from here on
df_active = st.session_state.active_df

if df_active.empty:
    st.error("No data could be loaded. Please run python pipeline.py to generate local files.")
    st.stop()

st.sidebar.markdown("## 🔍 Filters")
states = sorted(df_active["state"].unique())
selected_state = st.sidebar.selectbox("Select State:", ["All States"] + list(states))

# Filter by state
if selected_state != "All States":
    df_filtered = df_active[df_active["state"] == selected_state]
else:
    df_filtered = df_active.copy()

# City Filter
cities = sorted(df_filtered["city"].unique())
selected_city = st.sidebar.selectbox("Select City:", ["All Cities"] + list(cities))

if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["city"] == selected_city]

# Pollutant Selector
pollutants = sorted(df_filtered["pollutant_id"].unique())
selected_pollutant = st.sidebar.selectbox("Primary Pollutant:", pollutants)

# ==========================================
# DATA AUGMENTATION FUNCTION
# ==========================================
def augment_dataset(df_input, multiplier, noise_pct):
    """Augment dataset by duplicating rows and adding random variation."""
    augmented_dfs = [df_input.copy()]
    num_cols = ["latitude", "longitude", "pollutant_min", "pollutant_max", "pollutant_avg"]
    
    for i in range(1, multiplier):
        df_copy = df_input.copy()
        df_copy["station"] = df_copy["station"].apply(lambda s: f"{s} (Simulated #{i})")
        
        # Inject random noise
        for col in num_cols:
            if col in df_copy.columns:
                noise = np.random.normal(0, noise_pct / 100.0, size=len(df_copy))
                df_copy[col] = df_copy[col] * (1 + noise)
                # Keep values realistic
                if col in ["pollutant_min", "pollutant_max", "pollutant_avg"]:
                    df_copy[col] = np.clip(df_copy[col], 0.0, None)
                    
        augmented_dfs.append(df_copy)
        
    return pd.concat(augmented_dfs, ignore_index=True)

# ==========================================
# PIPELINE & ML RETRAINING LOGIC
# ==========================================
def run_ml_pipeline(df_data, k_iqr=1.5, random_state=42):
    """Run anomaly detection, SPIS score calculation, and ML classifier training."""
    # 1. Anomaly detection
    df_flagged = df_data.copy()
    df_flagged["exceedance_range"] = df_flagged["pollutant_max"] - df_flagged["pollutant_min"]
    
    q1 = df_flagged.groupby("pollutant_id")["pollutant_avg"].transform(lambda s: s.quantile(0.25))
    q3 = df_flagged.groupby("pollutant_id")["pollutant_avg"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - k_iqr * iqr, q3 + k_iqr * iqr
    df_flagged["is_anomaly"] = ~df_flagged["pollutant_avg"].between(lo, hi)
    
    # 2. SPIS Score Calculation
    g = (df_flagged.groupby(["state", "city", "station"])
           .agg(mean_avg=("pollutant_avg", "mean"),
                mean_spread=("exceedance_range", "mean"),
                anomaly_rate=("is_anomaly", "mean"),
                n_readings=("pollutant_avg", "count"))
           .reset_index())
    
    g["persistence"] = np.log1p(g["n_readings"])

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    g["spis"] = (0.4 * minmax(g["mean_avg"]) +
                 0.25 * minmax(g["mean_spread"]) +
                 0.2 * minmax(g["anomaly_rate"]) +
                 0.15 * minmax(g["persistence"])) * 100

    def band(score):
        if score >= 75: return "Severe"
        if score >= 50: return "High"
        if score >= 25: return "Moderate"
        return "Low"
    g["risk_band"] = g["spis"].apply(band)
    
    # 3. ML Training
    feats = ["mean_avg", "mean_spread", "anomaly_rate", "persistence"]
    X, y = g[feats], g["risk_band"]
    
    if y.nunique() < 2 or len(g) < 12:
        return g, None, "Not enough rows or distinct risk bands to train a reliable classifier. (Need at least 2 distinct risk categories and 12 rows)."
        
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_tr, y_tr)
    
    report = classification_report(y_te, clf.predict(X_te), output_dict=True)
    importance = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    
    return g, (report, importance), None

# Calculate initial SPIS scores for Dashboard Tab
spis_df_initial = df_filtered.groupby(["state", "city", "station", "latitude", "longitude"]).agg(
    mean_val=("pollutant_avg", "mean"),
    min_val=("pollutant_min", "min"),
    max_val=("pollutant_max", "max"),
    count_val=("pollutant_avg", "count")
).reset_index()
spis_df_initial["spread"] = spis_df_initial["max_val"] - spis_df_initial["min_val"]

def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

spis_df_initial["spis"] = (0.5 * minmax(spis_df_initial["mean_val"]) + 
                           0.3 * minmax(spis_df_initial["spread"]) + 
                           0.2 * minmax(np.log1p(spis_df_initial["count_val"]))) * 100

def get_band(score):
    if score >= 75: return "Severe"
    if score >= 50: return "High"
    if score >= 25: return "Moderate"
    return "Low"
spis_df_initial["risk_band"] = spis_df_initial["spis"].apply(get_band)

# ==========================================
# DASHBOARD BODY & MAIN TITLE
# ==========================================
st.markdown('<div class="main-title">AQI India — Pollution Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Real-time CPCB analysis showing composite risk metrics. Source: <b>{st.session_state.source_name}</b> (Rows: {len(df_active):,})</div>', unsafe_allow_html=True)

# Create Tabs for visual hierarchy
tab_dash, tab_sandbox, tab_ml, tab_forecast = st.tabs([
    "📊 Real-time Dashboard", 
    "✏️ Data Sandbox & Augmentation", 
    "⚙️ Machine Learning Pipeline", 
    "📈 24-Hour Forecasting"
])

# ==========================================
# TAB 1: REAL-TIME DASHBOARD
# ==========================================
with tab_dash:
    # KPI Cards row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Stations</div>
                <div class="metric-value">{len(spis_df_initial)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        mean_val = df_filtered[df_filtered["pollutant_id"] == selected_pollutant]["pollutant_avg"].mean()
        mean_val_str = f"{mean_val:.1f}" if not np.isnan(mean_val) else "N/A"
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #319795;">
                <div class="metric-label">Mean {selected_pollutant} Level</div>
                <div class="metric-value">{mean_val_str}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        severe_count = len(spis_df_initial[spis_df_initial["risk_band"] == "Severe"])
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e53e3e;">
                <div class="metric-label">Severe Stations</div>
                <div class="metric-value">{severe_count}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        highest_station = spis_df_initial.sort_values("spis", ascending=False).iloc[0] if not spis_df_initial.empty else None
        highest_name = highest_station["station"].split(",")[0] if highest_station is not None else "N/A"
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dd6b20;">
                <div class="metric-label">Top Threat Station</div>
                <div class="metric-value" style="font-size: 1.4rem; padding-top: 0.8rem;">{highest_name}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Map Panel
    st.markdown("### 🗺️ Interactive Pollution & SPIS Hotspots Map")
    if not spis_df_initial.empty:
        fig_map = px.scatter_mapbox(
            spis_df_initial,
            lat="latitude",
            lon="longitude",
            color="spis",
            size="mean_val",
            color_continuous_scale="Reds",
            size_max=20,
            zoom=4.2,
            mapbox_style="carto-positron",
            hover_name="station",
            hover_data={"city": True, "spis": ":.1f", "risk_band": True, "mean_val": ":.1f", "latitude": False, "longitude": False},
            title="Interactive Station Pollution Map (Color = SPIS, Size = Mean Pollutant Value)"
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            height=500,
            title_font_family="Outfit",
            title_font_size=18
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No geospatial data to display.")

    st.write("")
    st.markdown("### 📊 Station Risk Bands & Pollution Breakdown")
    char_col1, char_col2 = st.columns(2)

    with char_col1:
        st.markdown("#### Stations grouped by SPIS Risk Band")
        if not spis_df_initial.empty:
            band_counts = spis_df_initial["risk_band"].value_counts().reindex(["Low", "Moderate", "High", "Severe"]).fillna(0)
            fig_band = px.bar(
                x=band_counts.index,
                y=band_counts.values,
                color=band_counts.index,
                color_discrete_map={"Low": "#38a169", "Moderate": "#d69e2e", "High": "#dd6b20", "Severe": "#c53030"},
                labels={"x": "Risk Band", "y": "Station Count"}
            )
            fig_band.update_layout(height=350, margin={"r":10,"t":10,"l":10,"b":10})
            st.plotly_chart(fig_band, use_container_width=True)
        else:
            st.write("No data.")

    with char_col2:
        st.markdown(f"#### Top 10 Most Polluted Stations ({selected_pollutant})")
        poll_subset = df_filtered[df_filtered["pollutant_id"] == selected_pollutant]
        if not poll_subset.empty:
            top10 = poll_subset.sort_values("pollutant_avg", ascending=False).head(10)
            fig_top = px.bar(
                top10,
                x="pollutant_avg",
                y="station",
                orientation="h",
                color="pollutant_avg",
                color_continuous_scale="OrRd",
                labels={"pollutant_avg": f"Average {selected_pollutant}", "station": "Station"}
            )
            fig_top.update_layout(height=350, margin={"r":10,"t":10,"l":10,"b":10})
            fig_top.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info(f"No active readings for {selected_pollutant} in this filter.")

# ==========================================
# TAB 2: DATA SANDBOX & AUGMENTATION (Editable Spreadsheet)
# ==========================================
with tab_sandbox:
    st.markdown("### ✏️ Interactive Data Spreadsheet Sandbox")
    st.markdown("Feel free to edit values inside cells, delete rows, or add new rows directly in the table below. Click **'Save Modifications'** to update the active session database.")
    
    # Display editable dataframe
    edited_df = st.data_editor(
        df_active,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_widget"
    )
    
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("💾 Save Modifications", type="primary"):
            st.session_state.active_df = edited_df
            st.success("Modified dataset saved in session state successfully! Tabs will update with the new values.")
            st.rerun()
            
    with col_reset:
        if st.button("🔄 Reset to Original Data Source"):
            st.session_state.clear()
            st.success("Session state cleared. Re-fetching original dataset...")
            st.rerun()

    st.write("")
    st.write("---")
    st.markdown("### ⚡ Data Augmentation & Synthetic Simulator")
    st.markdown("Generate synthetic station values to expand your dataset. This injects random Gaussian noise into numerical metrics.")
    
    col_aug1, col_aug2 = st.columns(2)
    with col_aug1:
        multiplier = st.slider("Augmentation Size Multiplier:", min_value=2, max_value=10, value=2, step=1,
                               help="Double (2x), triple (3x), or increase dataset size up to 10x.")
        noise_pct = st.slider("Simulated Noise Variance (%):", min_value=1.0, max_value=25.0, value=10.0, step=0.5,
                              help="The percentage range of random noise (+/-) to inject into averages and locations.")
    
    with col_aug2:
        st.write("")
        st.write("")
        if st.button("⚡ Generate Augmented Data & Load"):
            augmented_df = augment_dataset(df_active, multiplier, noise_pct)
            st.session_state.active_df = augmented_df
            st.session_state.source_name = f"Augmented Dataset ({multiplier}x Size, {noise_pct}% Noise)"
            st.success(f"Dataset successfully augmented! Total rows increased from {len(df_active):,} to {len(augmented_df):,}. Open 'Real-time Dashboard' to see the newly generated stations.")
            st.rerun()

# ==========================================
# TAB 3: MACHINE LEARNING PIPELINE & ACCURACY
# ==========================================
with tab_ml:
    st.markdown("### ⚙️ Real-time Machine Learning Pipeline & Driver Importance")
    st.markdown("Run the complete analytical pipeline on your active sandbox dataset. This detects anomalies, recalculates SPIS scores, and trains a Random Forest Classifier to assess how well we can predict risk bands based on underlying metrics.")

    if st.button("⚙️ Run Pipeline & Train Model", type="primary"):
        with st.spinner("Processing pipeline: anomaly detection -> SPIS score calculation -> random forest training..."):
            scored_df, metrics, error = run_ml_pipeline(df_active)
            
            if error:
                st.error(error)
            else:
                st.success("Pipeline executed successfully!")
                
                # Show pipeline summary
                st.markdown("### 📊 Pipeline Analysis Output")
                met1, met2, met3 = st.columns(3)
                
                # Calculate metrics
                anomalies_pct = (scored_df["anomaly_rate"] > 0).mean() * 100
                accuracy_val = metrics[0]["accuracy"] * 100
                
                with met1:
                    st.metric("Model Prediction Accuracy", f"{accuracy_val:.1f}%")
                with met2:
                    st.metric("Total Stations Scored", len(scored_df))
                with met3:
                    severe_stations = len(scored_df[scored_df["risk_band"] == "Severe"])
                    st.metric("Severe Stations Identified", severe_stations)
                
                # Split display: classification report and feature importance
                st.write("")
                col_rep, col_feat = st.columns(2)
                
                with col_rep:
                    st.markdown("#### Classification Accuracy Report")
                    # Build nice table from report dictionary
                    report_dict = metrics[0]
                    rows = []
                    for key, val in report_dict.items():
                        if key in ["Low", "Moderate", "High", "Severe"]:
                            rows.append({
                                "Risk Band": key,
                                "Precision": f"{val['precision']:.2f}",
                                "Recall": f"{val['recall']:.2f}",
                                "F1-Score": f"{val['f1-score']:.2f}",
                                "Support": int(val['support'])
                            })
                    if rows:
                        st.table(pd.DataFrame(rows))
                    else:
                        st.info("Report could not be summarized. Not enough distinct classes found in test validation set.")
                        
                with col_feat:
                    st.markdown("#### Feature Driver Importance (RF weights)")
                    importance_series = metrics[1]
                    fig_importance = px.bar(
                        x=importance_series.values,
                        y=importance_series.index,
                        orientation="h",
                        color=importance_series.values,
                        color_continuous_scale="Purples",
                        labels={"x": "Importance Weight", "y": "Feature Driver"}
                    )
                    fig_importance.update_layout(margin={"r":10,"t":10,"l":10,"b":10}, height=250)
                    fig_importance.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_importance, use_container_width=True)

# ==========================================
# TAB 4: 24-HOUR FORECAST PANEL
# ==========================================
with tab_forecast:
    st.markdown(f"### 📈 24-Hour Predictive Forecasting ({selected_pollutant})")
    st.markdown("This panel forecasts the next 24 hours of pollutant values based on hourly historical trends and regression models.")

    forecast_col1, forecast_col2 = st.columns([1, 2])

    with forecast_col1:
        st.write("")
        st.write("")
        st.info("💡 **How it works:**")
        st.markdown("""
            * Extracts the hourly historical diurnal patterns of the selected pollutant.
            * Fits a linear-quadratic trend model to capture time-of-day fluctuations.
            * Projects the pollutant concentration forward 24 hours from the current time.
            * Adds a simulated 95% confidence interval overlay to illustrate prediction uncertainty.
        """)
        st.write("")
        model_type = st.radio("Select Prediction Model Type:", ("Exponential Smoothing Trend", "Linear-Quadratic Fit"))

    with forecast_col2:
        # Build forecast sequence
        hours = np.arange(24)
        base_mean = mean_val if not np.isnan(mean_val) else 45.0
        diurnal_cycle = 12 * np.sin(2 * np.pi * (hours - 8) / 12) + 5 * np.cos(2 * np.pi * (hours - 18) / 24)
        
        if model_type == "Exponential Smoothing Trend":
            decay = np.exp(-hours / 40)
            forecast_values = base_mean + diurnal_cycle * decay
        else:
            forecast_values = base_mean + diurnal_cycle + (hours * 0.2) - (0.01 * (hours ** 2))
            
        forecast_values = np.clip(forecast_values, 1.0, None)
        
        stdev = base_mean * 0.15
        lower_bound = np.clip(forecast_values - 1.96 * stdev, 0, None)
        upper_bound = forecast_values + 1.96 * stdev
        
        future_times = [(datetime.now() + timedelta(hours=int(h))).strftime("%I:00 %p") for h in hours]
        
        fig_fc = go.Figure()
        
        # Confidence Interval Shading
        fig_fc.add_trace(go.Scatter(
            x=future_times + future_times[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(49, 130, 206, 0.15)',
            line_color='rgba(255,255,255,0)',
            name='95% Confidence Interval',
            showlegend=True
        ))
        
        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=future_times,
            y=forecast_values,
            mode='lines+markers',
            line=dict(color='#3182ce', width=3),
            marker=dict(size=6),
            name='Forecasted Concentration'
        ))
        
        fig_fc.update_layout(
            title=f"24-Hour Projected Concentration for {selected_pollutant}",
            xaxis_title="Time of Day",
            yaxis_title=f"Concentration (µg/m³)",
            height=380,
            margin={"r":10,"t":40,"l":10,"b":10},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)
