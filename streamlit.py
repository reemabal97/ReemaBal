import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = '/Users/reemabalharith/Desktop/ai/FuelEconomy copy.csv'
data = pd.read_csv(file_path)

# Display column names for debugging
st.write("Columns in the data:", data.columns)

# Clean up column names by stripping any extra spaces
data.columns = data.columns.str.strip()

# Check if 'Extracurricular Activities' exists and handle it
if 'Extracurricular Activities' in data.columns:
    # Handle missing values in 'Extracurricular Activities' if any
    data['Extracurricular Activities'].fillna('Unknown', inplace=True)

    # Convert 'Extracurricular Activities' column to numerical values
    le = LabelEncoder()
    data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])
else:
    st.error("'Extracurricular Activities' column not found in the data.")

# Define inputs and output
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours',
          'Sample Question Papers Practiced']]
y = data['Performance Index']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit interface
st.title("Student Performance Prediction")
st.subheader("This project predicts student performance based on various factors")

# Display the DataFrame
st.write("Below is the original DataFrame:")
st.write(data)

# User inputs for prediction
st.subheader("Enter the following details to predict the performance index:")
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=6)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=80)
extracurricular_activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=8)
sample_papers_practiced = st.number_input("Sample Papers Practiced", min_value=0, max_value=10, value=2)

# Convert extracurricular activities to numerical format for prediction
extracurricular_activities_num = 1 if extracurricular_activities == 'Yes' else 0

# Make a prediction
input_data = [[hours_studied, previous_scores, extracurricular_activities_num, sleep_hours, sample_papers_practiced]]
predicted_performance = model.predict(input_data)

# Display the prediction
st.subheader("Predicted Performance Index:")
st.write(f"The predicted performance index is: {predicted_performance[0]:.2f}")

# Display a random line chart as an example
st.subheader("Random Line Chart Example:")
chart_data = pd.DataFrame(np.random.randn(20, 4), columns=['p', 'q', 'r', 's'])
st.line_chart(chart_data)