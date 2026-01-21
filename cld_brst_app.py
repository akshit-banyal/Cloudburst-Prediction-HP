import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION (Must be the first command) ---
st.set_page_config(page_title="Cloudburst Early Warning", page_icon="⛈️", layout="wide")

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('cloudburst_rf_model.pkl')

try:
    model = load_model()
except:
    st.error("Model file not found! Please ensure 'cloudburst_rf_model.pkl' is in the same folder.")
    st.stop()

# --- 3. App Title & Description ---
st.title("⛈️ Himachal Cloudburst Early Warning System")
st.markdown("""
### AI-Powered Disaster Prediction (Pilot Project)
This system uses a **Random Forest Classifier** to predict the probability of a cloudburst event 
based on 5 key weather indicators.
* **Target Region:** Mandi & Kullu, Himachal Pradesh
* **Data Source:** Historical Satellite & Weather Station Data
""")

# --- 4. Sidebar Inputs (Number Inputs for Precision) ---
st.sidebar.header("📍 Input Weather Conditions")
st.sidebar.markdown("Type the exact values below:")

def user_input_features():
    st.sidebar.subheader("1. Temporal Data")
    year = st.sidebar.number_input("Year", 2000, 2030, 2023, help="Cloudbursts are showing an increasing trend in recent years.")
    
    st.sidebar.subheader("2. Rainfall Metrics")
    # Using number_input allows exact decimals (e.g., 13.21)
    precip = st.sidebar.number_input("Today's Rainfall (mm)", 0.0, 500.0, 13.21)
    precip_lag1 = st.sidebar.number_input("Yesterday's Rainfall (mm)", 0.0, 500.0, 8.73)
    
    st.sidebar.subheader("3. Moisture Metrics")
    t2m_dew = st.sidebar.number_input("Dew Point Temp (°C)", -20.0, 40.0, 17.61, help="Higher Dew Point = More moisture in the air.")
    t2m_dew_lag1 = st.sidebar.number_input("Yesterday's Dew Point (°C)", -20.0, 40.0, 16.77)
    
    return year, precip, precip_lag1, t2m_dew, t2m_dew_lag1

year, precip, precip_lag1, t2m_dew, t2m_dew_lag1 = user_input_features()

# 5. SMART LOGIC (Fixing the 'Dilution' Problem) ---
# Instead of using a low average for hidden past days, we assume the past 
# was similar to 'Yesterday'. This preserves the "Storm Pattern" signal.

precip_lag_2_est = precip_lag1  # Smart Fill
precip_lag_3_est = precip_lag1  # Smart Fill

dew_lag_2_est = t2m_dew_lag1    # Smart Fill
dew_lag_3_est = t2m_dew_lag1    # Smart Fill

# Calculate Rolling Averages dynamically
roll_rain_3 = (precip + precip_lag1 + precip_lag_2_est) / 3
roll_dew_3 = (t2m_dew + t2m_dew_lag1 + dew_lag_2_est) / 3

input_data = {
    'YEAR': year,
    'DOY': 183.0,                             # Avg (July)
    'PRECTOTCORR': precip,
    'T2M': 11.9,                              # Avg Temp
    'T2MDEW': t2m_dew,
    'RH2M': 85.0,                             # Default to High Humidity (Storm context)
    'WS10M': 2.3,
    'PS': 77.8,
    'elevation': 1419.9,
    
    'PRECTOTCORR_lag_1': precip_lag1,
    'PRECTOTCORR_lag_2': precip_lag_2_est,
    'PRECTOTCORR_lag_3': precip_lag_3_est,
    
    'T2M_lag_1': 11.9, 'T2M_lag_2': 11.9, 'T2M_lag_3': 11.9,
    
    'T2MDEW_lag_1': t2m_dew_lag1,
    'T2MDEW_lag_2': dew_lag_2_est,
    'T2MDEW_lag_3': dew_lag_3_est,
    
    'RH2M_lag_1': 85.0, 'RH2M_lag_2': 85.0, 'RH2M_lag_3': 85.0,
    'WS10M_lag_1': 2.3, 'WS10M_lag_2': 2.3, 'WS10M_lag_3': 2.3,
    
    'PRECTOTCORR_roll_avg_3': roll_rain_3,
    'PRECTOTCORR_roll_avg_5': roll_rain_3,
    
    'T2M_roll_avg_3': 11.9, 'T2M_roll_avg_5': 11.9,
    
    'T2MDEW_roll_avg_3': roll_dew_3,
    'T2MDEW_roll_avg_5': roll_dew_3,
    
    'RH2M_roll_avg_3': 85.0, 'RH2M_roll_avg_5': 85.0,
    'WS10M_roll_avg_3': 2.3, 'WS10M_roll_avg_5': 2.3
}

features_df = pd.DataFrame([input_data])

# 6. Prediction Logic
if st.button("🔍 Analyze Risk Level", type="primary"):
    
    # 1. Get Probability
    prediction_proba = model.predict_proba(features_df)[0][1]
    
    # 2. Set Threshold (Optimized from our notebook)
    THRESHOLD = 0.09
    
    # 3. Display Results
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Analysis Result")
        if prediction_proba >= THRESHOLD:
            st.error("🚨 **CRITICAL ALERT: CLOUDBURST LIKELY**")
            st.markdown(f"The model detected a high probability (**{prediction_proba*100:.1f}%**) of extreme precipitation.")
            st.info("⚠️ **Recommendation:** Issue early warning to Mandi & Kullu administration.")
        else:
            st.success("✅ **SAFE: No Cloudburst Detected**")
            st.markdown(f"Risk level is low (**{prediction_proba*100:.1f}%**). Conditions appear normal.")
            
    with col2:
        st.subheader("Risk Meter")
        st.metric(label="Probability", value=f"{prediction_proba*100:.1f}%", delta=f"Threshold: {THRESHOLD*100}%")
        st.progress(min(int(prediction_proba * 100), 100))

# --- 7. Explainer Section (Project Context) ---
st.divider()
with st.expander(""):
    st.write("""
    
    """)