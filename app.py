import streamlit as st
import joblib
import numpy as np
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained ML model
model = joblib.load("rc_circuit_model.pkl")

# App title
st.title("RC Circuit Simulator (ML Model)")

st.markdown("""
This app predicts the **output voltage (Vout)** of an RC Low-Pass Filter
given Resistance (R), Capacitance (C), Input Voltage (Vin), and Frequency.
It also plots Vout vs Frequency for better visualization.
""")

# User inputs
R = st.number_input("Resistance R (Ohms)", min_value=1.0, value=1000.0)
C = st.number_input("Capacitance C (Farads)", min_value=1e-12, value=1e-6)
Vin = st.number_input("Input Voltage Vin (Volts)", min_value=0.0, value=5.0)

# Frequency range slider for plotting
freq_min = st.number_input("Minimum Frequency (Hz)", min_value=0.0, value=10.0)
freq_max = st.number_input("Maximum Frequency (Hz)", min_value=freq_min+1, value=10000.0)
freq_points = st.number_input("Number of points for plot", min_value=10, max_value=1000, value=200)

# Generate frequency array
frequencies = np.linspace(freq_min, freq_max, int(freq_points))

# Predict Vout for each frequency
input_data = np.array([[R, C, Vin, f] for f in frequencies])
vout_predictions = model.predict(input_data)

# Predict for user-entered frequency
user_freq = st.number_input("Enter Frequency to Predict Vout (Hz)", min_value=0.0, value=1000.0)
vout_user = model.predict(np.array([[R, C, Vin, user_freq]]))

# Display prediction for user frequency
st.success(f"Predicted Output Voltage (Vout) at {user_freq} Hz: {vout_user[0]:.4f} V")

# Plot Vout vs Frequency
st.subheader("Vout vs Frequency")
fig, ax = plt.subplots()
ax.plot(frequencies, vout_predictions, color='blue')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Vout (V)")
ax.set_title("RC Low-Pass Filter Response")
ax.grid(True)
st.pyplot(fig)

