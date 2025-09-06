# Place as the first thing in your Streamlit script

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import random

# --- Joypy: import if available ---
try:
    import joypy
    joypy_imported = True
except ImportError:
    joypy_imported = False

#### OCEAN THEME SETUP ####
st.set_page_config(page_title="Water Consumption Dashboard", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://i.pinimg.com/originals/d0/44/36/d04436335c516adf0ec8cd1e5093b1db.gif');
            background-size: cover;
        }
        .block-container {
            background: rgba(0, 90, 133, 0.92);
            border-radius: 18px !important;
            padding: 2rem !important;
            margin-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #e0f7fa !important;
        }
        .metric-label, .metric-value {
            color: #004d57 !important;
        }
    </style>
""", unsafe_allow_html=True)
sns.set_theme(style="whitegrid")
sns.set_palette("ocean")

st.title("🌊 Water Consumption Dashboard")

st.sidebar.title("Upload Data 🌊")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

#---- MAIN DASHBOARD LOGIC ----#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    orig_shape = df.shape
    # Robust date + month parse
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
        except Exception as e:
            st.warning("Could not parse 'date': " + str(e))
            df['month'] = 1
    else:
        st.warning("Column 'date' not found.")
        df['month'] = 1

    if 'region' not in df.columns or 'consumption_liters' not in df.columns:
        st.error("Your CSV must have columns named 'region' and 'consumption_liters'.")
    else:
        df_cleaned = df.dropna(subset=['region', 'consumption_liters']).copy()
        rows_removed = orig_shape[0] - df_cleaned.shape[0]
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        # Fun summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Points 🌊", df_cleaned.shape[0])
        col2.metric("Unique Regions 📍", df_cleaned['region'].nunique())
        col3.metric("Rows Cleaned 🧼", rows_removed)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        # Ridgeline plot
        st.subheader("🪼 Consumption by Month")
        if joypy_imported:
            fig, ax = joypy.joyplot(df_cleaned, by="month", column="consumption_liters",
                                    colormap=plt.cm.Blues, fade=True, figsize=(12,8))
            plt.xlabel("Consumption (Liters)")
            st.pyplot(fig)
        else:
            st.warning("joypy not installed - run 'pip install joypy' for ridgeline plots.")

        # Violin plot by region
        st.subheader("🐟 Distribution by Region")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.violinplot(x="region", y="consumption_liters", data=df_cleaned, ax=ax, inner="quartile", color="#00bcd4")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Overlapping density per region
        st.subheader("🌊 Density Plots by Region")
        fig, ax = plt.subplots(figsize=(12,6))
        for region in df_cleaned['region'].unique():
            sns.kdeplot(df_cleaned[df_cleaned['region'] == region]['consumption_liters'],
                        ax=ax, fill=True, alpha=0.3, label=region)
        plt.xlabel("Consumption (Liters)")
        plt.legend()
        st.pyplot(fig)

        # Box plot by region
        st.subheader("📦 Consumption in Litres by Region")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x='region', y='consumption_liters', data=df_cleaned, ax=ax, palette="ocean")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        #--------- REGRESSION / PREDICTION SECTION ----------#
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.header("🤖 Regression & Prediction")
        df_model = df_cleaned.copy()
        # Date feature
        df_model['date_ordinal'] = df_model['date'].map(pd.Timestamp.toordinal)
        # One-hot encode region
        df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)
        X = df_model.drop(columns=['date', 'consumption_liters'])
        y = df_model['consumption_liters']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'**R² Score:** {r2:.3f}')
        st.write(f'**Mean Squared Error:** {mse:.2f}')
        fig5, ax5 = plt.subplots(figsize=(6,6))
        ax5.scatter(y_test, y_pred, alpha=0.6, color='#0097A7')
        ax5.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual Consumption (Liters)')
        plt.ylabel('Predicted Consumption (Liters)')
        plt.title('Actual vs Predicted Consumption')
        st.pyplot(fig5)
        st.markdown('</div>', unsafe_allow_html=True)

        #-------- CONTINUOUS ANOMALY MONITORING ----------#
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.header("🌊 Continuous Anomaly Monitoring")

        def generate_dummy_data(seed=None):
            if seed:
                np.random.seed(seed)
            current_time = datetime.now()
            data = {
                'timestamp': current_time,
                'consumption': np.random.normal(100, 15),
                'temperature': np.random.normal(22, 5),
                'day_of_week': current_time.weekday(),
                'hour_of_day': current_time.hour
            }
            actual_anomaly = False
            if np.random.random() < 0.05:
                anomaly_factor = random.choice([0.2, 5.0])
                data['consumption'] *= anomaly_factor
                actual_anomaly = True
            return pd.DataFrame([data]), actual_anomaly

        def predict_anomaly(data):
            consumption = data['consumption'].values[0]
            return 1 if consumption < 30 or consumption > 200 else 0

        tab1, tab2 = st.tabs(["Monitor", "How It Works"])
        with tab1:
            st.write("### 🌊 Simulated Real-Time Anomaly Detection")
            interval = st.slider("Refresh interval (seconds)", min_value=1, max_value=10, value=2)
            seed = st.number_input("Random Seed", min_value=0, max_value=100, value=4)
            dummy_run = st.button("Run Simulation")
            if dummy_run:
                for i in range(10):
                    new_data, actual_anomaly = generate_dummy_data(seed)
                    prediction = predict_anomaly(new_data)
                    st.write(f"**Timestamp:** {new_data['timestamp'].values[0]}")
                    st.write(f"**Water Consumption:** {new_data['consumption'].values[0]:.2f} units")
                    st.write(f"**Prediction:** {'🌊 ANOMALY DETECTED!' if prediction == 1 else 'Normal consumption'}")
                    if actual_anomaly and prediction == 1:
                        st.success("✓ True Positive: Correctly identified anomaly")
                    elif not actual_anomaly and prediction == 0:
                        st.info("✓ True Negative: Correctly identified normal consumption")
                    elif actual_anomaly and prediction == 0:
                        st.error("✗ False Negative: Missed anomaly")
                    else:
                        st.warning("✗ False Positive: False alarm")
                    time.sleep(interval)
        with tab2:
            st.write("""
                _This tab simulates continuous water consumption monitoring, 
                generating dummy data with occasional anomalies and evaluating prediction accuracy. 
                The anomaly logic is threshold-based for demonstration; 
                integrate your model for production scenarios._
            """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload your water consumption data (CSV) to get started 💧")

