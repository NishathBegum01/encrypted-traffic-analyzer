import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import time

# Streamlit setup
st.set_page_config(page_title="AI Network Security Dashboard", layout="centered")

# Sidebar
st.sidebar.title("Feature Navigation")
selected_feature = st.sidebar.radio("Select an Analysis Feature", (
    "Project Overview",
    "Traffic Classification",
    "Anomaly / Threat Detection",
    "False Positive Analysis",
    "Performance & Scalability",
    "Privacy-Preserving Detection"
))

st.title("AI-Based Network Traffic Analysis Dashboard")
st.markdown("This dashboard supports real-time anomaly detection and behavior-based traffic classification for secure and encrypted environments.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file (41 raw features - no header)", type=["csv"])

if uploaded_file:
    try:
        start = time.time()

        # Load and validate dataset
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] != 41:
            st.error("Invalid file format. The dataset must contain exactly 41 columns.")
            st.stop()

        # Assign feature names
        column_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        df.columns = column_names

        # Preprocess categorical fields
        for col in ["protocol_type", "service", "flag"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        # Normalize and fill NaNs
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df)

        # Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(X_scaled)
        df['Anomaly'] = [1 if p == -1 else 0 for p in model.predict(X_scaled)]

        # Summary metrics
        normal_count = (df['Anomaly'] == 0).sum()
        anomaly_count = (df['Anomaly'] == 1).sum()
        end = time.time()

        # Feature-specific views
        if selected_feature == "Project Overview":
            st.subheader("Project Overview")
            st.markdown("""
            This AI-based system enables intelligent and automated network traffic analysis.  
            Key capabilities include:
            - Supervised traffic classification
            - Unsupervised anomaly detection
            - Real-time processing with scalable performance
            - Privacy-preserving behavioral analysis
            """)

        elif selected_feature == "Traffic Classification":
            st.subheader("Traffic Classification (Preprocessed Data)")
            st.markdown("Raw input data preprocessed for classification based on traffic behavior.")
            st.dataframe(df.head())

        elif selected_feature == "Anomaly / Threat Detection":
            st.subheader("Anomaly Detection using Isolation Forest")
            st.text(f"Normal Traffic Samples: {normal_count}")
            st.text(f"Detected Anomalies: {anomaly_count}")
            st.markdown("Anomalous records (outliers) are shown below.")
            st.dataframe(df[df['Anomaly'] == 1])

            fig, ax = plt.subplots()
            ax.bar(["Normal", "Anomaly"], [normal_count, anomaly_count], color=["#4CAF50", "#F44336"])
            ax.set_ylabel("Count")
            ax.set_title("Anomaly Detection Results")
            st.pyplot(fig)

        elif selected_feature == "False Positive Analysis":
            st.subheader("False Positive / Negative Summary")
            st.markdown("""
            This section provides basic insights into anomaly label distribution  
            which can later be cross-validated against classification results.
            """)
            count_summary = df['Anomaly'].value_counts().rename(index={0: "Normal", 1: "Anomaly"}).reset_index()
            count_summary.columns = ["Category", "Count"]
            st.dataframe(count_summary)

        elif selected_feature == "Performance & Scalability":
            st.subheader("Real-Time Processing Performance")
            st.text(f"Total rows processed: {len(df)}")
            st.text(f"Processing time: {end - start:.2f} seconds")

            st.markdown("Scalability analysis based on sample sizes:")

            sizes = [100, 300, 500, 800, 1000]
            times = []
            speeds = []

            for s in sizes:
                sample_df = df.sample(min(s, len(df)), random_state=42)
                X_sample = MinMaxScaler().fit_transform(sample_df.drop("Anomaly", axis=1))
                temp_model = IsolationForest(contamination=0.2, random_state=42)
                t0 = time.time()
                temp_model.fit(X_sample)
                t1 = time.time()
                duration = round(t1 - t0, 3)
                times.append(duration)
                speeds.append(round(s / duration, 2) if duration > 0 else 0)

            # Plot: Time vs Rows
            fig1, ax1 = plt.subplots()
            ax1.plot(sizes, times, marker='o', linestyle='--', color='blue')
            ax1.set_title("Processing Time by Row Count")
            ax1.set_xlabel("Sample Size (rows)")
            ax1.set_ylabel("Time (s)")
            st.pyplot(fig1)

            # Plot: Speed vs Rows
            fig2, ax2 = plt.subplots()
            ax2.plot(sizes, speeds, marker='o', linestyle='--', color='green')
            ax2.set_title("Processing Speed by Row Count")
            ax2.set_xlabel("Sample Size (rows)")
            ax2.set_ylabel("Speed (rows/sec)")
            st.pyplot(fig2)

        elif selected_feature == "Privacy-Preserving Detection":
            st.subheader("Privacy-Preserving Analysis")
            st.markdown("""
            This AI model is designed to function in privacy-constrained environments.  
            It uses statistical features derived from metadata like:
            - Duration and byte counts
            - Failed login attempts
            - Session frequency and rates

            No packet inspection or decryption is required.
            """)

        # Export processed results
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Labeled Output", data=csv_download, file_name="anomaly_output.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
