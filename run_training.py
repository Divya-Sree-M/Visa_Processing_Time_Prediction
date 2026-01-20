import pandas as pd
from src.train import train_models

df = pd.read_csv("data/preprocessed.csv")

results = train_models(df)

print("\nModel Training Results\n" + "-" * 30)
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"R2   : {metrics['R2']:.2f}")

print("\nModels saved successfully in /models folder")
