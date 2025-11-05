"""
Train a simple climate classifier model
Classes: 0 = Not Hot, 1 = Hot, 2 = Hot and Humid
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Create synthetic training data
np.random.seed(42)

# Class 0: Not Hot (temp < 25°C)
not_hot_temp = np.random.uniform(15, 25, 100)
not_hot_humidity = np.random.uniform(30, 90, 100)
not_hot_labels = np.zeros(100)

# Class 1: Hot (25°C <= temp < 32°C, humidity < 70%)
hot_temp = np.random.uniform(25, 32, 100)
hot_humidity = np.random.uniform(30, 70, 100)
hot_labels = np.ones(100)

# Class 2: Hot and Humid (temp >= 25°C, humidity >= 70%)
hot_humid_temp = np.random.uniform(25, 40, 100)
hot_humid_humidity = np.random.uniform(70, 95, 100)
hot_humid_labels = np.full(100, 2)

# Combine all data
X = np.vstack(
    [
        np.column_stack([not_hot_temp, not_hot_humidity]),
        np.column_stack([hot_temp, hot_humidity]),
        np.column_stack([hot_humid_temp, hot_humid_humidity]),
    ]
)

y = np.concatenate([not_hot_labels, hot_labels, hot_humid_labels])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training climate classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Test predictions
test_cases = [
    [20, 50],  # Should be 0: Not Hot
    [28, 45],  # Should be 1: Hot
    [30, 85],  # Should be 2: Hot and Humid
]

print("\nTest predictions:")
for temp, hum in test_cases:
    pred = model.predict([[temp, hum]])[0]
    classes = ["Not Hot", "Hot", "Hot and Humid"]
    print(f"  Temp: {temp}°C, Humidity: {hum}% → {classes[int(pred)]}")

# Save model
Path("models").mkdir(exist_ok=True)
with open("models/climate_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to models/climate_classifier.pkl")
