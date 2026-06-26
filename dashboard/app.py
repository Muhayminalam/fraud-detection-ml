import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")
st.write("Upload transaction data and detect potentially fraudulent transactions using a trained XGBoost model.")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    if st.button("Run Fraud Detection"):
        with st.spinner("Running fraud detection..."):
            transactions = df.drop(columns=["Class"], errors="ignore").to_dict(orient="records")

            response = requests.post(
                "http://127.0.0.1:8000/predict_batch",
                json=transactions
            )

            predictions = response.json()["predictions"]
            result_df = pd.concat([df, pd.DataFrame(predictions)], axis=1)

        st.subheader("Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(result_df))
        col2.metric("Fraud Cases Detected", int(result_df["is_fraud"].sum()))
        col3.metric("Average Fraud Probability", round(result_df["fraud_probability"].mean(), 4))

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        st.subheader("Fraud Probability Distribution")
        st.bar_chart(result_df["fraud_probability"])

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )