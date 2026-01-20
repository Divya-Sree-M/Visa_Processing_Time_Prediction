import os
import joblib
import pandas as pd
import numpy as np

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(__file__)
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")
MODEL_PATH = os.path.join(BASE_DIR, "../models/random_forest_model.pkl")

# =====================================================
# Load model (JOBLIB ONLY)
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# Load encoders / aggregates (JOBLIB ONLY)
# =====================================================
top_n_groups = joblib.load(os.path.join(ENCODER_DIR, "top_n_groups.pkl"))
country_avg = joblib.load(os.path.join(ENCODER_DIR, "country_avg.pkl"))
industry_avg = joblib.load(os.path.join(ENCODER_DIR, "industry_avg.pkl"))
visa_avg = joblib.load(os.path.join(ENCODER_DIR, "visa_avg.pkl"))
seasonal_index = joblib.load(os.path.join(ENCODER_DIR, "seasonal_index.pkl"))

# =====================================================
# Helpers
# =====================================================
def encode_with_top_n(value, valid_list):
    if value is None:
        return len(valid_list)
    if value in valid_list:
        return valid_list.index(value)
    return len(valid_list)

def safe_array_mean(arr):
    try:
        return float(np.mean(arr))
    except:
        return 0.0

# =====================================================
# Prediction
# =====================================================
def predict_processing_time(input_dict):

    # ---------------- RAW INPUTS ----------------
    prevailing_wage = float(input_dict.get("prevailing_wage", 0))

    wage_level = input_dict.get("wage_level", "Level I")
    experience_required = input_dict.get("experience_required", "No")
    education = input_dict.get("education", "High School")

    industry = input_dict.get("industry", "Missing")
    employer_city = input_dict.get("employer_city", "Missing")
    job_title = input_dict.get("job_title", "Missing")
    country = input_dict.get("country_of_citizenship", "Missing")

    visa_type = input_dict.get("visa_type", "H1B")
    application_month = int(input_dict.get("application_month", 6))

    # ---------------- ORDINAL ENCODINGS ----------------
    wage_level_map = {"Level I": 0, "Level II": 1, "Level III": 2, "Level IV": 3}
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    experience_map = {"No": 0, "Yes": 1}

    wage_level_encoded = wage_level_map.get(wage_level, 0)
    education_encoded = education_map.get(education, 0)
    experience_required_encoded = experience_map.get(experience_required, 0)

    # ---------------- TOP-N ENCODINGS ----------------
    industry_enc = encode_with_top_n(industry, top_n_groups["industry"])
    employer_city_enc = encode_with_top_n(employer_city, top_n_groups["employer_city"])
    job_title_enc = encode_with_top_n(job_title, top_n_groups["job_title"])
    country_of_citizenship_enc = encode_with_top_n(
        country, top_n_groups["country_of_citizenship"]
    )

    # ---------------- AGGREGATES ----------------
    country_avg_days = safe_array_mean(country_avg)
    industry_avg_days = safe_array_mean(industry_avg)
    visa_avg_days = safe_array_mean(visa_avg)

    try:
        seasonal_idx = seasonal_index[application_month - 1]
    except:
        seasonal_idx = 0.5

    # ---------------- FINAL FEATURES (ORDER LOCKED) ----------------
    X = pd.DataFrame([{
        "prevailing_wage": prevailing_wage,
        "wage_level_encoded": wage_level_encoded,
        "experience_required_encoded": experience_required_encoded,
        "education_encoded": education_encoded,
        "industry_enc": industry_enc,
        "employer_city_enc": employer_city_enc,
        "job_title_enc": job_title_enc,
        "country_of_citizenship_enc": country_of_citizenship_enc,
        "seasonal_index": seasonal_idx,
        "country_avg_days": country_avg_days,
        "visa_avg": visa_avg_days,
        "industry_avg": industry_avg_days
    }])

    # ---------------- PREDICT ----------------
    return round(float(model.predict(X)[0]), 2)
