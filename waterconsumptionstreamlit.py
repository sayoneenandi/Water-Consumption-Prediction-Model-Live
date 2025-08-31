import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import random
import time

# Page setup
st.set_page_config(page_title="Water Consumption Dashboard", page_icon="💧", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1976d2;'>Water Consumption Analysis & Monitoring Dashboard</h1>", unsafe_allow_html=True)

st.sidebar.header("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload water consumption CSV", type=['csv'])
seed_val = st.sidebar.number_input("Seed value (for simulation)", min_value=0, value=42)
monitor_active = st.sidebar.checkbox("Activate real-time monitoring simulation")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.info("Upload a water consumption CSV to begin. Using example data.")
    regions = ['North', 'East', 'South', 'West']
    dates = pd.date_range('2023-01-01', periods=900)
    df = pd.DataFrame({
        'region': np.random.choice(regions, len(dates)),
        'date': dates,
        'consumption_liters': np.random.normal(14000, 3500, len(dates))
    })
    df['date'] = pd.to_datetime(df['date'])

# Region selection
region_options = list(df['region'].unique())
selected_regions = st.sidebar.multiselect("Select regions for visualization", region_options, default=region_options)
df = df[df['region'].isin(selected_regions)]

# Date range filter
start_date = st.sidebar.date_input("Start Date", value=df['date'].min())
end_date = st.sidebar.date_input("End Date", value=df['date'].max())
df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

st.subheader("Cleaned Data Preview")
st.dataframe(df.head())

# Stats cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", len(df))
col2.metric("Average Consumption (L)", f"{df['consumption_liters'].mean():.2f}")
col3.metric("Date Range", f"{df['date'].min().date()} — {df['date'].max().date()}")

# Boxplot by region
st.markdown("### Consumption Distribution by Region")
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(x='region', y='consumption_liters', data=df, ax=ax)
ax.set_title('Boxplot of Consumption by Region')
st.pyplot(fig)

# Consumption over time by region
st.markdown("### Time Series: Consumption Over Time (per region)")
for region in selected_regions:
    region_data = df[df['region'] == region]
    ts = region_data.groupby('date')['consumption_liters'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(ts['date'], ts['consumption_liters'])
    ax.set_title(f'Region: {region} - Daily Consumption')
    ax.set_xlabel('Date'); ax.set_ylabel('Total Consumption (L)'); ax.grid(alpha=0.4)
    st.pyplot(fig)

# Avg consumption comparison
st.markdown("### Average Consumption Comparison (Bar & Line)")
region_avg = df.groupby('region')['consumption_liters'].mean().reset_index().sort_values('consumption_liters',ascending=False)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='region', y='consumption_liters', data=region_avg, alpha=0.7, ax=ax)
ax2 = ax.twinx()
ax2.plot(region_avg['region'], region_avg['consumption_liters'], marker='o', color='red')
ax2.set_ylabel('Avg Consumption (Line)', color='red'); ax2.tick_params(axis='y', colors='red')
plt.tight_layout()
st.pyplot(fig)

# Monthly patterns
df['month'] = df['date'].dt.month
monthly = df.groupby(['region','month'])['consumption_liters'].mean().reset_index()
st.markdown("### Average Monthly Consumption (per region)")
for region in selected_regions:
    fig, ax = plt.subplots(figsize=(9,4))
    m = monthly[monthly['region']==region]
    sns.lineplot(x='month', y='consumption_liters', data=m, marker='o', ax=ax)
    ax.set_title(f'Average Monthly Consumption: {region}')
    ax.grid(alpha=0.45); ax.set_xticks(range(1,13))
    st.pyplot(fig)

# Monthly fluctuations normalized
st.markdown("### Monthly Consumption Fluctuations (Normalized Across Regions)")
fig, ax = plt.subplots(figsize=(11,7))
for region in selected_regions:
    rm = monthly[monthly['region']==region]
    norm = rm['consumption_liters']/rm['consumption_liters'].mean()
    ax.plot(rm['month'], norm, marker='o', label=region)
ax.set_title('Monthly Fluctuations by Region (Normalized)')
ax.set_xlabel('Month')
ax.set_ylabel('Normalized Consumption')
ax.set_xticks(range(1,13))
ax.grid(alpha=0.5)
ax.legend()
st.pyplot(fig)

# Consumption distribution
st.markdown("### Consumption Distribution by Region (Histogram)")
n = len(selected_regions)
fig, axs = plt.subplots(1, n, figsize=(15, 5))
if n == 1: axs = [axs]
for i, region in enumerate(selected_regions):
    sns.histplot(df[df['region']==region]['consumption_liters'], kde=True, ax=axs[i])
    axs[i].set_title(f'{region} Distribution')
    axs[i].set_xlabel('Consumption (L)')
plt.tight_layout()
st.pyplot(fig)

# Model: Predict consumption
st.markdown("### Simple Linear Regression Forecast")
df_model = df.copy()
df_model['date_ordinal'] = df_model['date'].map(pd.Timestamp.toordinal)
df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)
X = df_model.drop(columns=['date', 'consumption_liters','month'])
y = df_model['consumption_liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.write(f"**R² Score:** {r2:.3f}  |  **MSE:** {mse:.2f}")

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(y_test, y_pred, c='turquoise', alpha=0.6)
lims = [min(y.min(),y_pred.min()), max(y.max(),y_pred.max())]
ax.plot(lims, lims, 'k--', lw=2)
ax.set_xlabel('Actual Consumption (L)')
ax.set_ylabel('Predicted Consumption (L)')
ax.set_title('Actual vs Predicted')
plt.tight_layout()
st.pyplot(fig)

# Real-time simulation (optional)
if monitor_active:
    st.markdown("### Real-Time Water Consumption Monitoring Simulation")
    sim_placeholder = st.empty()
    for i in range(12):
        now = datetime.now()
        consumption = np.random.normal(100, 15)
        temp = np.random.normal(22, 5)
        data = {'timestamp': now, 'consumption': consumption, 'temperature': temp,
                'day_of_week': now.weekday(), 'hour_of_day': now.hour}
        anomaly = 1 if consumption < 30 or consumption > 200 else 0
        status = "ANOMALY DETECTED!" if anomaly else "Normal consumption"
        sim_placeholder.markdown(f"""
        **Time:** {data['timestamp']}  
        **Consumption:** {data['consumption']:.2f} units  
        **Status:** {status}  
        """)
        time.sleep(2)
    st.success("Simulation complete.")

# Show raw data
if st.checkbox("Show raw data table"):
    st.dataframe(df)
