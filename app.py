import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "fraud_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "fraud_detection.csv"

st.set_page_config(page_title="Fraud Detection Analytics", page_icon="🔎", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

model, scaler = load_model()
df = load_data()

st.title("🔎 Fraud Detection Analytics")
st.caption("Machine-learning based transaction fraud analysis dashboard")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Fraudulent Transactions", f"{int(df['is_fraud'].sum()):,}")
c3.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
c4.metric("Average Amount", f"₹{df['amount'].mean():,.2f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Fraud Distribution")
    counts = df["is_fraud"].value_counts().rename(index={0:"Legitimate",1:"Fraud"})
    st.bar_chart(counts)

with right:
    st.subheader("Transaction Amount by Class")
    chart_df = df.groupby("is_fraud")["amount"].mean().rename(index={0:"Legitimate",1:"Fraud"})
    st.bar_chart(chart_df)

st.subheader("Recent Transactions")
display_df = df.copy()
display_df["status"] = display_df["is_fraud"].map({0:"Legitimate", 1:"Fraud"})
st.dataframe(display_df.head(20), use_container_width=True)

st.divider()
st.header("🧾 Single Transaction Prediction")

with st.form("prediction_form"):
    a,b,c = st.columns(3)
    amount = a.number_input("Transaction Amount (₹)", min_value=1.0, value=500.0)
    hour = b.slider("Transaction Hour", 0, 23, 14)
    distance = c.number_input("Distance From Home (km)", min_value=0.0, value=5.0)

    d,e,f = st.columns(3)
    failed = d.number_input("Failed Attempts", 0, 10, 0)
    online = e.selectbox("Online Transaction", ["No", "Yes"])
    new_device = f.selectbox("New Device", ["No", "Yes"])

    g,h,i = st.columns(3)
    international = g.selectbox("International Transaction", ["No", "Yes"])
    velocity = h.number_input("Transaction Velocity", 1, 20, 2)
    merchant_risk = i.slider("Merchant Risk Score", 1, 5, 2)

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_df = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": hour,
        "distance_from_home_km": distance,
        "failed_attempts": failed,
        "online_transaction": 1 if online == "Yes" else 0,
        "new_device": 1 if new_device == "Yes" else 0,
        "international_transaction": 1 if international == "Yes" else 0,
        "transaction_velocity": velocity,
        "merchant_risk_score": merchant_risk
    }])

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Potential Fraud Detected — Risk Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Transaction Appears Legitimate — Fraud Probability: {probability*100:.2f}%")

st.divider()
st.info("Note: This project uses a synthetic dataset for educational/demo purposes. It should not be used as a real banking fraud decision system.")
