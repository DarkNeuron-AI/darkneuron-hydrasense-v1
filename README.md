# 💧 HydraSense - Water Potability Classification Model (v1.0)

A lightweight **Water Potability AI Model** built by **DarkNeuronAI** using **Random Forest Classifier** with **StandardScaler preprocessing**.  
This model predicts whether water is **Potable (1)** or **Not Potable (0)** based on chemical and physical features.

---

## 🚀 Features
- Fast and efficient — runs easily on standard laptops  
- Predicts water potability using features like **pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity**  
- Perfect for **data science projects**, IoT water quality monitoring, or environmental AI applications  
- Uses a **pipeline** to automatically scale new input data  

---

## 🧩 Tech Stack
- **Python 3.9+**  
- **Scikit-learn**  
- **Pandas / NumPy**  
- **Joblib** (for model saving/loading)  

---

## 📦 Files Included
- `water_potability_model.pkl` → Trained Random Forest pipeline (scaler + model)  
- `example_usage.py` → Example usage of the model

---

## 🏷️ Prediction Labels (Binary)
- **0:** Not Potable (Unsafe to drink)  
- **1:** Potable (Safe to drink)

---

## 💡 Usage Example
```python
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("water_potability_model.pkl")

# Create a new water sample
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

sample_df = pd.DataFrame(sample_data)

# Predict potability
prediction = model.predict(sample_df)
print("Potable" if prediction[0] == 1 else "Not Potable")
```

# Made With ❤️ By DarkNeuronAI
