import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "fraud_detection.csv"
MODEL_FILE = "fraud_model.pkl"
SCALER_FILE = "scaler.pkl"

def main():
    df = pd.read_csv(DATA_FILE)

    features = [
        "amount", "transaction_hour", "distance_from_home_km",
        "failed_attempts", "online_transaction", "new_device",
        "international_transaction", "transaction_velocity",
        "merchant_risk_score"
    ]
    X = df[features]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSaved {MODEL_FILE} and {SCALER_FILE}")

if __name__ == "__main__":
    main()
