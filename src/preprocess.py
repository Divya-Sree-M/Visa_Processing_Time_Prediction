import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ENCODER_PATH = Path("src/encoders")
ENCODER_PATH.mkdir(parents=True, exist_ok=True)

TOP_N_COLS = {
    "industry": 15,
    "employer_city": 20,
    "job_title": 20,
    "country_of_citizenship": 15
}

TARGET_COL = "processing_time_days"

def preprocess_dataframe(df: pd.DataFrame, training: bool = True):
    """
    Preprocess visa application data.
    Applies feature engineering, encoding, and prepares X, y.
    """

    # Select relevant columns
    df = df[
        [
            "case_received_date",
            "decision_date",
            "class_of_admission",
            "country_of_citizenship",
            "job_info_job_title",
            "pw_amount_9089",
            "pw_level_9089",
            "employer_state",
            "employer_city",
            "naics_us_title",
            "job_info_education",
            "job_info_experience",
        ]
    ].copy()

    # Rename for clarity
    df.rename(
        columns={
            "job_info_job_title": "job_title",
            "pw_amount_9089": "prevailing_wage",
            "pw_level_9089": "wage_level",
            "naics_us_title": "industry",
            "job_info_education": "education_level",
        },
        inplace=True,
    )

    # Convert dates
    df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

    # Create target
    df[TARGET_COL] = (df["decision_date"] - df["case_received_date"]).dt.days
    df = df[df[TARGET_COL] > 0].dropna(subset=[TARGET_COL])

    # Handle numeric values
    df["prevailing_wage"] = (
        df["prevailing_wage"].astype(str).str.replace(",", "").astype(float)
    )
    df["prevailing_wage"] = df["prevailing_wage"].fillna(df["prevailing_wage"].median())

    # Handle categorical nulls
    categorical_cols = [
        "class_of_admission",
        "country_of_citizenship",
        "job_title",
        "employer_state",
        "employer_city",
        "industry",
        "wage_level",
        "education_level",
    ]
    df[categorical_cols] = df[categorical_cols].fillna("Missing")
    df["job_info_experience"] = df["job_info_experience"].fillna("N")

    # Wage level encoding
    wage_map = {"Level I": 1, "Level II": 2, "Level III": 3, "Level IV": 4, "Missing": 0}
    df["wage_level_encoded"] = df["wage_level"].map(wage_map)

    # Experience encoding
    df["experience_required_encoded"] = df["job_info_experience"].map({"Y": 1, "N": 0})

    # Education normalization
    def clean_education(val):
        val = str(val).lower()
        if "doctor" in val or "phd" in val:
            return "Doctorate"
        if "master" in val:
            return "Master"
        if "bachelor" in val:
            return "Bachelor"
        if "none" in val:
            return "None"
        return "Other"

    df["education_cleaned"] = df["education_level"].apply(clean_education)
    edu_map = {"None": 0, "Other": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
    df["education_encoded"] = df["education_cleaned"].map(edu_map)

    # ---------- GROUP TOP-N ----------
    if training:
        top_n_groups = {}
        for col, n in TOP_N_COLS.items():
            top_vals = df[col].value_counts().nlargest(n).index.tolist()
            top_n_groups[col] = top_vals
            df[f"{col}_grouped"] = np.where(df[col].isin(top_vals), df[col], "Other")

        joblib.dump(top_n_groups, ENCODER_PATH / "top_n_groups.pkl")
    else:
        top_n_groups = joblib.load(ENCODER_PATH / "top_n_groups.pkl")
        for col, top_vals in top_n_groups.items():
            df[f"{col}_grouped"] = np.where(df[col].isin(top_vals), df[col], "Other")

    # ---------- MEAN ENCODING ----------
    if training:
        mean_encoders = {}
        for col in TOP_N_COLS.keys():
            grouped_col = f"{col}_grouped"
            mean_map = df.groupby(grouped_col)[TARGET_COL].mean()
            mean_encoders[col] = mean_map
            df[f"{col}_enc"] = df[grouped_col].map(mean_map)

        joblib.dump(mean_encoders, ENCODER_PATH / "mean_encoders.pkl")
    else:
        mean_encoders = joblib.load(ENCODER_PATH / "mean_encoders.pkl")
        for col, mean_map in mean_encoders.items():
            grouped_col = f"{col}_grouped"
            global_mean = mean_map.mean()
            df[f"{col}_enc"] = df[grouped_col].map(mean_map).fillna(global_mean)


    # ---- SEASONAL INDEX FEATURE ----
    df["app_month"] = df["case_received_date"].dt.month

    if training:
        global_mean = df[TARGET_COL].mean()
        month_means = df.groupby("app_month")[TARGET_COL].mean()
        seasonal_index_map = (month_means / global_mean).to_dict()

        joblib.dump(seasonal_index_map, ENCODER_PATH / "seasonal_index.pkl")
    else:
        seasonal_index_map = joblib.load(ENCODER_PATH / "seasonal_index.pkl")

    df["seasonal_index"] = df["app_month"].map(seasonal_index_map).fillna(1.0)

    # ---- COUNTRY AVERAGE DAYS ----
    if training:
        country_avg_map = df.groupby("country_of_citizenship_grouped")[TARGET_COL].mean()
        joblib.dump(country_avg_map, ENCODER_PATH / "country_avg.pkl")
    else:
        country_avg_map = joblib.load(ENCODER_PATH / "country_avg.pkl")

    df["country_avg_days"] = (
        df["country_of_citizenship_grouped"]
        .map(country_avg_map)
        .fillna(country_avg_map.mean())
    )

    # ---- CATEGORY AVERAGES ----
    if training:
        visa_avg = df.groupby("class_of_admission")[TARGET_COL].mean()
        industry_avg = df.groupby("industry")[TARGET_COL].mean()
        joblib.dump(visa_avg, ENCODER_PATH / "visa_avg.pkl")
        joblib.dump(industry_avg, ENCODER_PATH / "industry_avg.pkl")
    else:
        visa_avg = joblib.load(ENCODER_PATH / "visa_avg.pkl")
        industry_avg = joblib.load(ENCODER_PATH / "industry_avg.pkl")

    df["visa_avg"] = df["class_of_admission"].map(visa_avg).fillna(visa_avg.mean())
    df["industry_avg"] = df["industry"].map(industry_avg).fillna(industry_avg.mean())


    # Final feature set
    df_final = df[
        [
            TARGET_COL,
            "prevailing_wage",
            "wage_level_encoded",
            "experience_required_encoded",
            "education_encoded",
            "industry_enc",
            "employer_city_enc",
            "job_title_enc",
            "country_of_citizenship_enc",
            "seasonal_index",
            "country_avg_days",
            "visa_avg",
            "industry_avg",
        ]
    ]

    # ---- FINAL NaN HANDLING (MANDATORY) ----
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Save preprocessed dataset (training only)
    if training:
        df_final.to_csv("data/preprocessed.csv", index=False)

    X = df_final.drop(columns=TARGET_COL)
    y = df_final[TARGET_COL]

    return X, y
