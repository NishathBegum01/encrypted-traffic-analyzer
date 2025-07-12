# encrypted-traffic-analyzer
A smart AI-based tool that analyzes network traffic, detects anomalies, and classifies threats in real-time — all through a clean, interactive dashboard.

# AI Network Threat Detector

This project is an AI-based network traffic analysis system that performs real-time anomaly detection and behavior-based classification. It uses machine learning models to identify suspicious activity without inspecting encrypted payloads.

---Features
- Automated Traffic Classification using Random Forest
- Anomaly / Threat Detection using Isolation Forest
- False Positive Analysis via hybrid logic
- Scalability Metrics with real-time performance graphs
- Privacy-Preserving: works without decrypting traffic
- Interactive Dashboard built with Streamlit

- 
---

## How to Run

### 1. Clone the Repository

git clone https://github.com/your-username/ai-network-threat-detector.git
cd ai-network-threat-detector

**2. Install Required Packages**
pip install -r requirements.txt
Or install manually:
pip install streamlit pandas numpy scikit-learn matplotlib

**3. Train the Model**
cd src
python train_model.py

**4. Launch the Dashboard**
cd ../app
streamlit run dashboard.py


Project Structure 
AI_Network_Security_Project/

**app/** – Streamlit dashboard

dashboard.py – Main web UI script

**src **– Model training and prediction logic

train_model.py – Trains the Random Forest model

preprocess.py – Preprocesses the dataset

anomaly_detector.py – Detects anomalies using Isolation Forest

predict.py – Runs predictions using the trained model

data/ – Network traffic datasets

KDDTrain_sample.csv – Training dataset

KDDTest_sample.csv – Test dataset

models/ – Saved machine learning models

traffic_classifier.pkl – Trained Random Forest classifier

README.md – Project description and instructions

requirements.txt – List of Python packages

(Optional) .gitignore – Git config to ignore unnecessary files



