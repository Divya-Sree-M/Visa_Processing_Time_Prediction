import pandas as pd
from src.preprocess import preprocess_dataframe

# Load raw dataset
df = pd.read_csv("data/raw.csv", low_memory=False)

# Run preprocessing (training mode)
X, y = preprocess_dataframe(df, training=True)

print("Preprocessing completed.")
print("Preprocessed data saved to data/preprocessed.csv")
