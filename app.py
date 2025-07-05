
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page title
st.title("📊 Student Marks Predictor")
st.write("Predict student marks based on the number of hours studied.")

# Input from user
hours = st.number_input("Enter hours of study", min_value=0.0, step=0.5)

# Sample dataset (can be replaced with actual data or model)
X = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]).reshape(-1, 1)
y = np.array([20, 30, 40, 50, 53, 60, 65, 70, 75, 80])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
if st.button("Predict Marks"):
    predicted_marks = model.predict([[hours]])
    st.success(f"📈 Predicted Marks: {predicted_marks[0]:.2f}")

# Optional: Show data chart
if st.checkbox("Show Sample Training Data"):
    df = pd.DataFrame({"Hours Studied": X.flatten(), "Marks": y})
    st.line_chart(df.set_index("Hours Studied"))
    
