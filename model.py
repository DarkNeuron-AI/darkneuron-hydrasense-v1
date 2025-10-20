# =====================================
# | WATER POTABILITY CLASSIFIER MODEL |
# =====================================
# Coder: Gaurav Pandey
# Date Of Creation (DOC): 20 October, 2025

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib

try:
    # Load the dataset
    df = pd.read_csv("water_potability.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'water_potability.csv' was not found.")
    exit()
except Exception as e:
    print("An unexpected error occurred while loading the dataset:", e)
    exit()

try:
    # Handle missing values by filling with median
    df.fillna(df.median(), inplace=True)
    
    # Separate features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except KeyError:
    print("Error: The dataset does not contain the required column 'Potability'.")
    exit()
except Exception as e:
    print("An error occurred during preprocessing:", e)
    exit()

try:
    # Create a Pipeline: Scaling + Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    print("Pipeline trained successfully!")
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate model performance
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Save the pipeline (scaler + model together)
    joblib.dump(pipeline, 'water_potability_model.pkl')
    print("Model saved as 'water_potability_model.pkl'")
except Exception as e:
    print("An error occurred during pipeline training or evaluation:", e)
    exit()

try:
    # Test with a new sample input (pipeline scales automatically)
    sample = [[7.2, 180, 15000, 8.3, 350, 450, 10, 70, 3]]
    prediction = pipeline.predict(sample)
    
    print("Sample Prediction:", "Potable" if prediction[0] == 1 else "Not Potable")
except ValueError as e:
    print("Error: Sample input has incorrect shape or type.", e)
except Exception as e:
    print("An unexpected error occurred during prediction:", e)