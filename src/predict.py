import pandas as pd
import joblib
from pathlib import Path

from preprocess import preprocess_dataframe

# Paths
MODEL_PATH = Path("models/random_forest_model.pkl")

# Load trained model ONCE
model = joblib.load(MODEL_PATH)


def predict_processing_time(input_data: dict):

    # Convert input dict to DataFrame (single row)
    df_input = pd.DataFrame([input_data])

    # Run preprocessing in inference mode
    X, _ = preprocess_dataframe(df_input, training=False)

    # Predict
    prediction = model.predict(X)[0]

    return round(float(prediction), 2)
