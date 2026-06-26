import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Load feature names
features = joblib.load("models/features.pkl")


def predict_transaction(transaction):

    df = pd.DataFrame([transaction])

    # Keep columns in the same order as training
    df = df[features]

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= 0.5)

    return {
        "fraud_probability": round(float(probability), 4),
        "is_fraud": prediction
    }