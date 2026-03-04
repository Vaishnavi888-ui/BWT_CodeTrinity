!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("EcoPulse - AI Smart Energy Monitoring")

st.write("AI-Based Smart Energy Monitoring & Optimization System")

# Simulated energy data
np.random.seed(42)
hours = list(range(24))
energy_usage = np.random.normal(50, 10, 24)

# Artificial spikes
energy_usage[8] = 120
energy_usage[18] = 130

data = pd.DataFrame({
    "Hour": hours,
    "Energy Usage": energy_usage
})

st.subheader("Energy Usage Data")
st.write(data)

st.subheader("Energy Consumption Graph")

fig, ax = plt.subplots()
ax.plot(data["Hour"], data["Energy Usage"], marker="o")
ax.set_xlabel("Hour")
ax.set_ylabel("Energy Usage (kWh)")
ax.set_title("Daily Energy Consumption")

st.pyplot(fig)

# AI anomaly detection
model = IsolationForest(contamination=0.1)
data["Anomaly"] = model.fit_predict(data[["Energy Usage"]])

anomalies = data[data["Anomaly"] == -1]

st.subheader("Detected Energy Anomalies")

if len(anomalies) > 0:
    st.write(anomalies)
else:
    st.write("No anomalies detected")

st.subheader("AI Recommendations")

if len(anomalies) > 0:
    st.write("⚠ High energy usage detected")
    st.write("Suggestions:")
    st.write("- Turn off unused lights")
    st.write("- Optimize AC usage")
    st.write("- Shift heavy loads to off-peak hours")
else:
    st.write("Energy usage is optimal")

# Hackathon update - energy monitoring improvement
