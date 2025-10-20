import pandas as pd
import joblib

# Load the saved pipeline
try:
    model = joblib.load('water_potability_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'water_potability_model.pkl' was not found.")
    exit()
except Exception as e:
    print("An unexpected error occurred while loading the pipeline:", e)
    exit()

# Create a new water sample
# IMPORTANT: Use the same feature order and names as in training
sample_data = {
    'ph': [7.2],
    'Hardness': [180],
    'Solids': [15000],
    'Chloramines': [8.3],
    'Sulfate': [350],
    'Conductivity': [450],
    'Organic_carbon': [10],
    'Trihalomethanes': [70],
    'Turbidity': [3]
}

# Convert to DataFrame with proper column names
sample_df = pd.DataFrame(sample_data)

# Make prediction
try:
    prediction = model.predict(sample_df)
    result = "Potable" if prediction[0] == 1 else "Not Potable"
    print("Sample Prediction:", result)
except ValueError as e:
    print("Error: Sample input has incorrect shape or type.", e)
except Exception as e:
    print("An unexpected error occurred during prediction:", e)