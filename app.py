
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Employee Attrition Dashboard",
                   layout="wide", page_icon="👥")

st.title("👥 Employee Attrition Risk Dashboard")
st.markdown("Predict attrition risk, understand drivers, and analyze exit sentiment.")

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load("logistic_regression.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df    = pd.read_csv("hr_attrition.csv")
    exits = pd.read_csv("exit_survey.csv")
    return df, exits

model, scaler = load_assets()
df, exits     = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🔧 Filter Employees")
dept = st.sidebar.multiselect("Department",
       options=df["Department"].unique(),
       default=df["Department"].unique())

df_filtered = df[df["Department"].isin(dept)]

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Risk Predictor", "💬 Exit Sentiment"])

# ══ Tab 1: Overview ══════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    attrition_rate = (df_filtered["Attrition"] == "Yes").mean() * 100
    col1.metric("Total Employees",    len(df_filtered))
    col2.metric("Attrition Rate",     f"{attrition_rate:.1f}%")
    col3.metric("Avg Monthly Income", f"${df_filtered['MonthlyIncome'].mean():,.0f}")
    col4.metric("Overtime %",
        f"{(df_filtered['OverTime']=='Yes').mean()*100:.1f}%")

    st.subheader("Attrition Rate by Department")
    dept_att = (df_filtered.groupby("Department")["Attrition"]
                .apply(lambda x: (x=="Yes").mean()*100)
                .reset_index()
                .rename(columns={"Attrition": "Attrition Rate (%)"}))
    st.bar_chart(dept_att.set_index("Department"))

    st.subheader("Monthly Income Distribution by Attrition")
    fig, ax = plt.subplots(figsize=(8, 3))
    for label, grp in df_filtered.groupby("Attrition"):
        ax.hist(grp["MonthlyIncome"], bins=30, alpha=0.6, label=label)
    ax.legend(); ax.set_xlabel("Monthly Income")
    st.pyplot(fig); plt.close()

# ══ Tab 2: Risk Predictor ════════════════════════════════════
with tab2:
    st.subheader("🔮 Predict Attrition Risk for an Employee")
    st.markdown("Adjust the sliders and the model will predict attrition probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age         = st.slider("Age",               18, 60, 30)
        monthly_inc = st.slider("Monthly Income",  1000, 20000, 5000)
        overtime    = st.selectbox("Overtime", ["No","Yes"])
    with col2:
        env_sat     = st.slider("Environment Satisfaction", 1, 4, 3)
        job_sat     = st.slider("Job Satisfaction",         1, 4, 3)
        wlb         = st.slider("Work-Life Balance",        1, 4, 3)
    with col3:
        yrs_company = st.slider("Years at Company",     0, 40, 5)
        yrs_manager = st.slider("Years with Manager",   0, 20, 3)
        num_comp    = st.slider("Num Companies Worked", 0, 9,  2)

    # Build a representative feature vector (29 features)
    # Fill unshown features with dataset medians
    median_row = df_filtered.median(numeric_only=True)

    feature_vector = {
        "Age": age, "DailyRate": median_row.get("DailyRate", 800),
        "DistanceFromHome": median_row.get("DistanceFromHome", 7),
        "Education": 3, "EnvironmentSatisfaction": env_sat,
        "HourlyRate": median_row.get("HourlyRate", 66),
        "JobInvolvement": 3, "JobSatisfaction": job_sat,
        "MonthlyIncome": monthly_inc, "MonthlyRate": median_row.get("MonthlyRate", 14000),
        "NumCompaniesWorked": num_comp, "PercentSalaryHike": 13,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1, "TotalWorkingYears": max(age-22, 0),
        "TrainingTimesLastYear": 3, "WorkLifeBalance": wlb,
        "YearsAtCompany": yrs_company, "YearsInCurrentRole": max(yrs_company-2, 0),
        "YearsSinceLastPromotion": 2, "YearsWithCurrManager": yrs_manager,
        "BusinessTravel": 1, "Department": 1, "EducationField": 1,
        "Gender": 1, "JobRole": 1, "MaritalStatus": 1,
        "OverTime": 1 if overtime == "Yes" else 0,
    }

    fv = pd.DataFrame([feature_vector])
    feature_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'BusinessTravel', 'Department', 'EducationField', 'Gender',
                'JobRole', 'MaritalStatus', 'OverTime']
    fv = fv[feature_cols]
    fv_scaled = scaler.transform(fv)
    prob = model.predict_proba(fv_scaled)[0][1]

    color = "🔴" if prob > 0.5 else "🟡" if prob > 0.3 else "🟢"
    st.markdown(f"### {color} Attrition Risk: **{prob:.1%}**")
    st.progress(float(prob))

    if prob > 0.5:
        st.error("High risk — immediate HR intervention recommended")
    elif prob > 0.3:
        st.warning("Moderate risk — consider a retention conversation")
    else:
        st.success("Low risk — employee appears stable")

# ══ Tab 3: Exit Sentiment ════════════════════════════════════
with tab3:
    st.subheader("💬 Exit Survey Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exit reasons distribution**")
        reason_counts = exits["exit_reason"].value_counts()
        st.bar_chart(reason_counts)

    with col2:
        st.markdown("**Sentiment breakdown**")
        sent_counts = exits["sentiment_label"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(sent_counts, labels=sent_counts.index,
               autopct="%1.0f%%",
               colors=["#DD8452","#4C72B0","#55A868"])
        st.pyplot(fig); plt.close()

    st.subheader("Sample Exit Comments")
    reason_filter = st.selectbox("Filter by reason", ["All"] + list(exits["exit_reason"].unique()))
    df_exits_show = exits if reason_filter == "All" else exits[exits["exit_reason"] == reason_filter]
    st.dataframe(df_exits_show[["exit_reason","exit_comment","sentiment_label","sentiment_score"]]
                 .sample(min(10, len(df_exits_show)))
                 .reset_index(drop=True), use_container_width=True)

    avg_sent = exits.groupby("exit_reason")["sentiment_score"].mean().sort_values()
    st.subheader("Average Sentiment Score by Exit Reason")
    st.bar_chart(avg_sent)
