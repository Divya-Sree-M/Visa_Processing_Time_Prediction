import streamlit as st
from src.predict import predict_processing_time

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Visa Processing Time Predictor",
    page_icon="🛂",
    layout="centered"
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🛂 Visa Processing Time Prediction")
st.write(
    "Predict the estimated visa processing time (in days) "
    "based on applicant, job, and visa details."
)

st.divider()

# -------------------------------------------------
# INPUTS
# -------------------------------------------------

st.subheader("Applicant & Job Details")

prevailing_wage = st.number_input(
    "Prevailing Wage (USD)",
    min_value=0.0,
    step=1000.0
)

wage_level = st.selectbox(
    "Wage Level",
    ["Level I", "Level II", "Level III", "Level IV"]
)

experience_required = st.selectbox(
    "Experience Required",
    ["No", "Yes"]
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelors", "Masters", "PhD"]
)

industry = st.text_input(
    "Industry",
    placeholder="e.g., Software Publishers"
)

employer_city = st.text_input(
    "Employer City",
    placeholder="e.g., San Jose"
)

job_title = st.text_input(
    "Job Title",
    placeholder="e.g., Software Engineer"
)

country_of_citizenship = st.text_input(
    "Country of Citizenship",
    placeholder="e.g., India"
)

visa_type = st.selectbox(
    "Visa Type",
    ["H1B", "L1", "F1", "B1"]
)

application_month = st.slider(
    "Application Month",
    min_value=1,
    max_value=12,
    value=6
)

st.divider()

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("🔮 Predict Processing Time"):
    input_data = {
        "prevailing_wage": prevailing_wage,
        "wage_level": wage_level,
        "experience_required": experience_required,
        "education": education,
        "industry": industry,
        "employer_city": employer_city,
        "job_title": job_title,
        "country_of_citizenship": country_of_citizenship,
        "visa_type": visa_type,
        "application_month": application_month
    }

    try:
        prediction = predict_processing_time(input_data)

        st.success(
            f"✅ Estimated Visa Processing Time: **{prediction} days**"
        )

    except Exception as e:
        st.error("❌ Prediction failed. Please check inputs.")
        st.exception(e)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption(
    "Built using Machine Learning (Random Forest) • "
    "Deployed with Streamlit"
)
