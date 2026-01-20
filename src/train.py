import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COL = "processing_time_days"

def train_models(df):
    
    # -----------------------------
    # Feature / Target split
    # -----------------------------

    X = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # -----------------------------
    # Linear Regression
    # -----------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    joblib.dump(lr_model, "models/linear_regression_model.pkl")

    results["LinearRegression"] = {
        "MAE": lr_mae,
        "RMSE": lr_rmse,
        "R2": lr_r2,
    }

    # -----------------------------
    # Random Forest Regressor
    # -----------------------------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5
    )

    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    joblib.dump(rf_model, "models/random_forest_model.pkl")

    results["RandomForest"] = {
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2": rf_r2,
    }

    return results
