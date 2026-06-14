import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Health Insurance Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Real-Time Health Insurance Claim Fraud Detection")
st.markdown("AI-Powered Dashboard for Detecting Fraudulent Health Insurance Claims")

# --------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------

model = joblib.load("fraud_model.pkl")
df = pd.read_csv("dashboard_data.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard Overview",
        "Fraud Analysis",
        "Real-Time Prediction"
    ]
)

# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    total_claims = len(df)

    fraud_claims = len(
        df[df["Predicted_Fraud"] == 1]
    )

    genuine_claims = len(
        df[df["Predicted_Fraud"] == 0]
    )

    fraud_percentage = (
        fraud_claims / total_claims
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims",
        f"{total_claims:,}"
    )

    col2.metric(
        "Fraud Claims",
        f"{fraud_claims:,}"
    )

    col3.metric(
        "Genuine Claims",
        f"{genuine_claims:,}"
    )

    col4.metric(
        "Fraud %",
        f"{fraud_percentage:.2f}%"
    )

    st.markdown("---")

    pie_data = pd.DataFrame({
        "Category": ["Fraud", "Genuine"],
        "Count": [fraud_claims, genuine_claims]
    })

    fig = px.pie(
        pie_data,
        names="Category",
        values="Count",
        title="Fraud vs Genuine Claims"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# FRAUD ANALYSIS
# --------------------------------------------------

elif page == "Fraud Analysis":

    st.header("🔍 Fraud Analysis")

    fig = px.histogram(
        df,
        x="Fraud_Probability",
        nbins=30,
        title="Fraud Probability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    remove_cols = [
        "Predicted_Fraud",
        "Fraud_Probability",
        "Is_Fraudulent"
    ]

    numeric_cols = [
        col
        for col in numeric_cols
        if col not in remove_cols
    ]

    if len(numeric_cols) > 0:

        selected_feature = st.selectbox(
            "Select Feature",
            numeric_cols
        )

        fig2 = px.box(
            df,
            y=selected_feature,
            color=df["Predicted_Fraud"].astype(str),
            title=f"{selected_feature} vs Fraud"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("Top Fraud Cases")

    fraud_cases = df[
        df["Predicted_Fraud"] == 1
    ].sort_values(
        by="Fraud_Probability",
        ascending=False
    )

    st.dataframe(
        fraud_cases.head(20)
    )

# --------------------------------------------------
# REAL-TIME PREDICTION
# --------------------------------------------------

elif page == "Real-Time Prediction":

    st.header("⚡ Real-Time Fraud Prediction")

    st.write(
        "Enter claim information below:"
    )

    claim_amount = st.number_input(
        "Claim Amount",
        min_value=0.0,
        value=50000.0
    )

    patient_age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=120,
        value=45
    )

    procedures = st.number_input(
        "Number of Procedures",
        min_value=0,
        value=2
    )

    stay_days = st.number_input(
        "Length of Stay (Days)",
        min_value=0,
        value=5
    )

    previous_patient_claims = st.number_input(
        "Previous Claims (Patient)",
        min_value=0,
        value=1
    )

    previous_provider_claims = st.number_input(
        "Previous Claims (Provider)",
        min_value=0,
        value=2
    )

    if st.button("Predict Fraud"):

        input_data = pd.DataFrame(
            np.zeros(
                (
                    1,
                    len(model.feature_names_in_)
                )
            ),
            columns=model.feature_names_in_
        )

        if "Claim_Amount" in input_data.columns:
            input_data["Claim_Amount"] = claim_amount

        if "Patient_Age" in input_data.columns:
            input_data["Patient_Age"] = patient_age

        if "Number_of_Procedures" in input_data.columns:
            input_data["Number_of_Procedures"] = procedures

        if "Length_of_Stay_Days" in input_data.columns:
            input_data["Length_of_Stay_Days"] = stay_days

        if "Number_of_Previous_Claims_Patient" in input_data.columns:
            input_data["Number_of_Previous_Claims_Patient"] = previous_patient_claims

        if "Number_of_Previous_Claims_Provider" in input_data.columns:
            input_data["Number_of_Previous_Claims_Provider"] = previous_provider_claims

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                f"🚨 Fraud Detected\n\nProbability: {probability*100:.2f}%"
            )

        else:

            st.success(
                f"✅ Genuine Claim\n\nProbability of Fraud: {probability*100:.2f}%"
            )

        st.progress(float(probability))

        st.write(
            f"Fraud Probability: {probability*100:.2f}%"
        )

        import gdown
import gdown
import joblib
import os

file_id = "1-PPkWg_62EAok6_0ZZwJq4FMcbzv8Kls"
url = f"https://drive.google.com/uc?id={file_id}"

model_path = "fraud_model.pkl"

# Download only if not already present
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model
model = joblib.load(model_path)