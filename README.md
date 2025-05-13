# 🚗 Ola Driver Churn Prediction

A complete machine learning solution to predict driver churn for Ola using historical performance, demographics, and tenure data.

> This project includes end-to-end modeling, EDA, feature engineering, class imbalance treatment, and a fully interactive Streamlit dashboard.

---

## 📦 Project Structure

(```ola_driver_churn/
├── app/ # Streamlit dashboard
│ └── dashboard.py
├── data/ # Raw or sample input data (optional)
├── notebooks/ # Jupyter notebooks for EDA and modeling
│ └── churn_modeling.ipynb
├── outputs/
│ └── models/ # Saved models & processed driver data
│ ├── random_forest_model.pkl
│ ├── xgboost_model.pkl
│ └── driver_data.csv
├── requirements.txt # Project dependencies
├── README.md # Project overview
└── .gitignore # Files to exclude from git
```)


---

## 🧠 Problem Statement

Driver churn is a serious concern in ride-sharing. Drivers may leave the platform for competitors like Uber based on fluctuating ratings, earnings, or satisfaction.

**Objective**: Predict if a driver will leave the company using demographic, performance, and business data.

---

## 🧪 Modeling Approach

✅ Steps included:

- Exploratory Data Analysis (EDA)
- KNN Imputation of missing values
- Feature engineering (rating trend, income growth, tenure)
- One-hot encoding and standardization
- Model training:
  - 🎯 Random Forest (Bagging)
  - ⚡ XGBoost (Boosting)
- ROC AUC, precision/recall, F1-score comparison
- Threshold tuning for business-optimized decisions

---

## 📊 Streamlit Dashboard Preview

Launch your own interactive dashboard to:

- Filter drivers by **city, designation, tenure**
- See predicted **churn probabilities**
- View a list of **high-risk drivers**

### ▶️ Run Dashboard Locally:
```bash

(```
cd app
streamlit run dashboard.py
<!-- Optional: Replace the link below with an actual screenshot if desired --> <!-- ![Dashboard Screenshot](https://user-images.githubusercontent.com/your-screenshot-path.png) -->
📈 Model Performance Summary
Model	Accuracy	Precision	Recall	ROC-AUC
Random Forest	86%	87%	94%	0.89
XGBoost	86%	90%	88%	✅ 0.91
```)

🧠 Key Insights
Drivers with < 6 months tenure are at highest churn risk

Drop in quarterly rating or income predicts churn

City and designation-specific patterns impact loyalty

✅ Recommendations
Launch a 90-day retention program

Monitor and re-engage low-rated drivers

Focus on high-risk cities with proactive incentives

📎 Requirements
Install dependencies:


pip install -r requirements.txt
🌐 License & Credits
MIT License © 2025
Created by [Pavan Eleti] | LinkedIn | GitHub

🚀 Future Enhancements
SHAP explanations for model transparency

CSV export of filtered drivers

Cloud deployment via Streamlit Cloud or Render


---