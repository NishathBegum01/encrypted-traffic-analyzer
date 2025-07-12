# Encrypted Traffic Analyzer

**Encrypted Traffic Analyzer** is an AI-powered network monitoring tool designed to detect anomalies and classify threats in real-time — without inspecting or decrypting encrypted payloads. It leverages machine learning models to ensure privacy-preserving traffic analysis and provides insights through a clean, interactive dashboard built with Streamlit.

---

## Key Features

- **Automated Traffic Classification** using a Random Forest model  
- **Anomaly and Threat Detection** powered by Isolation Forest  
- **False Positive Reduction** through hybrid logic mechanisms  
- **Real-Time Performance Metrics** with visualizations  
- **Privacy-Preserving Analysis** — no decryption required  
- **User-Friendly Web Interface** developed using Streamlit  

---

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-network-threat-detector.git
cd ai-network-threat-detector
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, install them manually:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 3. Train the Model

Navigate to the `source` directory and run the training script:

```bash
cd source
python train_model.py
```

This will train the traffic classifier and save the model.

### 4. Launch the Dashboard

Navigate to the `app` directory and start the Streamlit dashboard:

```bash
cd ../app
streamlit run dashboard.py
```

---

## Project Structure

```
AI_Network_Security_Project/
│
├── app/                      # Streamlit web interface
│   └── dashboard.py          # Main dashboard script
│
├── source/                   # Model training and logic
│   ├── train_model.py        # Random Forest model training
│   ├── preprocess.py         # Data preprocessing utilities
│   ├── anomaly_detector.py   # Isolation Forest for anomaly detection
│   └── predict.py            # Prediction and classification logic
│
├── data/                     # Sample datasets
│   ├── KDDTrain_sample.csv
│   └── KDDTest_sample.csv
│
├── models/                   # Trained ML models
│   └── traffic_classifier.pkl
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## About the Project

This tool is intended for researchers, security analysts, and network engineers who are exploring ways to analyze encrypted traffic without compromising user privacy. By leveraging ML-based pattern recognition, the system helps in identifying suspicious behaviors while maintaining scalability and performance.
