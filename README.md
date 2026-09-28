# Fraud Detection Analytics

## Project Overview
Fraud Detection Analytics is a machine-learning project that analyzes transaction data and predicts whether a transaction may be fraudulent.

The project contains:
- Data preprocessing
- Random Forest classification
- Model evaluation
- Fraud analytics dashboard
- Single transaction prediction
- Interactive Streamlit interface

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Dataset
`fraud_detection.csv` is a synthetic dataset created for educational purposes. It contains transaction amount, transaction time, distance, failed attempts, device information, international transaction status, transaction velocity, merchant risk score, and the fraud label.

## Project Structure

```text
fraud-detection-analytics/
├── app.py
├── train_model.py
├── fraud_detection.csv
├── fraud_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── images/
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/fraud-detection-analytics.git
cd fraud-detection-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

This creates:
- `fraud_model.pkl`
- `scaler.pkl`

### 4. Start the dashboard
```bash
streamlit run app.py
```

The application will open in your browser.

## Machine Learning Workflow

```text
Transaction Data
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Train/Test Split
       ↓
Standard Scaling
       ↓
Random Forest Model
       ↓
Fraud Prediction
       ↓
Analytics Dashboard
```

## Features
- Fraud rate calculation
- Transaction analytics
- Fraud distribution visualization
- Average transaction amount
- Recent transaction table
- Individual transaction fraud prediction
- Fraud probability

## Limitations
- The dataset is synthetic and intended for demonstration.
- Real banking data requires stronger privacy, security, validation, and monitoring.
- Model performance on synthetic data does not represent performance in a real financial environment.

## Future Enhancements
- Use a verified real-world fraud dataset.
- Add anomaly detection methods.
- Add model explainability with SHAP.
- Add user authentication.
- Store transactions in a database.
- Add real-time transaction monitoring.
- Deploy the application on a cloud platform.

## Author
Shaik Sofiya

## Disclaimer
This project is for academic and educational purposes only. Predictions should not be used as the sole basis for real financial decisions.
