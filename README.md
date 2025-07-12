# AI Network Threat Detector

This project is an AI-based system for detecting threats and anomalies in network traffic. It combines machine learning models with a Streamlit dashboard to provide real-time, behavior-based traffic analysis — even in encrypted environments.

---

## 🔍 Features

- Supervised traffic classification using Random Forest
- Unsupervised anomaly detection using Isolation Forest
- Hybrid false positive analysis
- Scalable and real-time performance testing
- Privacy-preserving behavior-only detection
- Interactive dashboard for data upload, processing, and visualization

---

## 🧪 Environment & Dependencies

- **Python**: 3.8 or above
- **Recommended**: Use a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt

📁 Project Structure
app/

dashboard.py – Main Streamlit dashboard interface

src/

train_model.py – Trains Random Forest classifier

preprocess.py – Data preprocessing script

anomaly_detector.py – Detects anomalies

predict.py – Predicts using trained classifier

data/

KDDTrain_sample.csv – Training dataset

KDDTest_sample.csv – Testing dataset

models/

traffic_classifier.pkl – Trained ML model

README.md – This file

requirements.txt – List of required packages

🚀 How to Run
1. Train the Classifier

cd src
python train_model.py

2. Launch the Streamlit Dashboard

cd ../app
streamlit run dashboard.py

on the dashboard add the dataset file and click on the browse option and select the dataset file (e.g: .csv file )


📁 models/ Folder
Will contain trained model files:

classifier.pkl (supervised model)
anomaly_model.pkl (unsupervised anomaly detector)


