import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Ola Driver Churn Dashboard", layout="wide")

# ---- Load model and dataset ----
@st.cache_resource
def load_model_and_data():
    base = Path("G:/Projects/ola_driver_churn/outputs/models")
    model_path = base / "xgboost_model.pkl"
    data_path = base / "driver_data.csv"

    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    return model, data

model, data = load_model_and_data()

# ---- Reconstruct City & Designation ----
city_cols = [col for col in data.columns if col.startswith('city_first_')]
data['city'] = data[city_cols].idxmax(axis=1).str.replace('city_first_', '')

designation_cols = [col for col in data.columns if col.startswith('joining_designation_first_')]
data['designation'] = data[designation_cols].idxmax(axis=1).str.replace('joining_designation_first_', '')

# ---- Sidebar Filters ----
st.sidebar.header("🔍 Filter Drivers")

cities = sorted(data['city'].dropna().unique())
designations = sorted(data['designation'].dropna().unique())
min_tenure = int(data['tenure_months'].min())
max_tenure = int(data['tenure_months'].max())

selected_city = st.sidebar.selectbox("City", cities)
selected_designation = st.sidebar.selectbox("Joining Designation", designations)
tenure_range = st.sidebar.slider("Tenure Range (months)", min_tenure, max_tenure, (min_tenure, max_tenure))

# ---- Filter data ----
filtered = data[
    (data['city'] == selected_city) &
    (data['designation'] == selected_designation) &
    (data['tenure_months'] >= tenure_range[0]) &
    (data['tenure_months'] <= tenure_range[1])
]

st.title("🚗 Ola Driver Churn Dashboard")
st.markdown(f"### Showing {filtered.shape[0]} driver(s) matching filters")

if filtered.empty:
    st.warning("⚠️ No drivers match the selected criteria.")
else:
    features_to_drop = ['driver_id', 'target_max', 'lastworkingdate_last', 'dateofjoining_first', 'city', 'designation']
    X_filtered = filtered.drop(columns=[col for col in features_to_drop if col in filtered.columns], errors='ignore')

    # ---- Predict churn probabilities ----
    churn_probs = model.predict_proba(X_filtered)[:, 1]
    filtered['churn_probability'] = churn_probs

    # ---- Show driver table ----
    st.subheader("📋 Driver Churn Probabilities")
    st.dataframe(filtered[['driver_id', 'tenure_months', 'quarterly_rating_last', 'income_last', 'churn_probability']]
                 .sort_values(by='churn_probability', ascending=False)
                 .reset_index(drop=True),
                 use_container_width=True)

    # ---- Highlight high risk drivers ----
    high_risk = filtered[filtered['churn_probability'] >= 0.80]

    st.subheader(f"⚠️ High Risk Drivers ({len(high_risk)})")
    if len(high_risk) > 0:
        st.dataframe(high_risk[['driver_id', 'tenure_months', 'quarterly_rating_last', 'churn_probability']]
                     .sort_values(by='churn_probability', ascending=False)
                     .reset_index(drop=True),
                     use_container_width=True)
    else:
        st.success("No high risk drivers found for selected filters.")

st.markdown("---")
st.markdown("Made with ❤️ for the Ola Analytics Team")
