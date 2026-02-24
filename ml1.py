import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Title
st.title("🌦️ KNN Weather Classification App")

st.write("This app predicts whether the weather is Sunny or Rainy based on Temperature and Humidity.")

# Dataset
X = np.array([[30, 70],
              [25, 80],
              [27, 60],
              [31, 65],
              [23, 85],
              [28, 75]])

y = np.array([0, 1, 0, 0, 1, 1])

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=["Temperature", "Humidity"])
df["Weather"] = y
df["Weather Label"] = df["Weather"].map({0: "Sunny ☀️", 1: "Rainy 🌧️"})

# Sidebar Inputs
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature (°C)", 20, 40, 26)
humidity = st.sidebar.slider("Humidity (%)", 50, 100, 78)

# Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_point = np.array([[temp, humidity]])
prediction = knn.predict(new_point)[0]

# Prediction Output
st.subheader("Prediction Result")

if prediction == 0:
    st.success("Predicted Weather: Sunny ☀️")
else:
    st.info("Predicted Weather: Rainy 🌧️")

# Add new point to dataframe for plotting
new_df = pd.DataFrame({
    "Temperature": [temp],
    "Humidity": [humidity],
    "Weather Label": ["New Prediction ⭐"]
})

plot_df = pd.concat([df[["Temperature", "Humidity", "Weather Label"]], new_df])

# Plotly Scatter Plot
fig = px.scatter(
    plot_df,
    x="Temperature",
    y="Humidity",
    color="Weather Label",
    size=[15]*len(plot_df),
    title="KNN Weather Classification Visualization"
)

st.plotly_chart(fig, use_container_width=True)
