# Cloudburst Prediction System for Himachal Pradesh ⛈️

### A Machine Learning Approach to Early Warning (Kullu & Mandi Districts)

---

## 📌 Project Overview

This project is a **Machine Learning–based Cloudburst Prediction System** focused on extreme weather risk prediction in Himalayan regions. The system focuses on predicting the **likelihood of cloudburst events** in the **Kullu and Mandi districts of Himachal Pradesh**, regions that are highly vulnerable to extreme rainfall events due to their complex Himalayan terrain.

The objective of this project is to support **early warning and risk awareness** using historical weather data and supervised learning techniques.

---

## 🚀 Live Demo

🔗 **Streamlit Cloud App:**
https://cloudburst-prediction-hp-jalrakshak.streamlit.app

---

## 🎯 Problem Statement

Cloudbursts are sudden, localized extreme rainfall events that often result in flash floods, landslides, and severe infrastructural damage in Himalayan regions. Traditional forecasting systems struggle with **localized prediction** due to data sparsity and terrain complexity.

This project aims to:

* Analyze historical weather patterns
* Address data imbalance in extreme event prediction
* Provide a **district-specific ML-based risk prediction tool**

---

## 🛠️ Technical Implementation

### 🔹 Data Preparation & Integration

The model training dataset was constructed by **integrating three distinct categories of weather data** (raw meteorological parameters, derived indicators, and event-labeled records). These datasets were:

* Cleaned and standardized
* Aligned temporally and geographically
* Merged into a **single consolidated CSV file** used for model training and evaluation

This preprocessing step ensured consistency across features and reduced noise before applying Machine Learning techniques.

### 🔹 Machine Learning

* **Algorithm:** Random Forest Classifier
* **Problem Type:** Binary Classification (Cloudburst / No Cloudburst)
* **Class Imbalance Handling:**

  * Stratified Sampling
  * Class Weight Adjustment

### 🔹 Data Sources

* **NASA POWER** – Meteorological and atmospheric parameters
* **IMD (India Meteorological Department)** – Regional historical weather records

### 🔹 Deployment

* **Framework:** Streamlit
* **Platform:** Streamlit Cloud
* **Model Serialization:** Pickle (`model.pkl`)

---

## 💻 Key Features

* **District-Specific Prediction**
  Focused analysis for Kullu and Mandi districts

* **Real-Time Risk Prediction**
  User inputs:

  * Temperature
  * Humidity
  * Precipitation

* **Interactive Data Visualization**
  Charts and plots to analyze weather trends and historical patterns

* **User-Friendly Web Interface**
  Simple and intuitive dashboard built using Streamlit

---

## 📂 Project Structure

The repository contains multiple working and intermediate files used during experimentation. The structure below highlights **only the final and important components** required to understand, run, and evaluate the system.

```
CLOUDBURSTPREDMODEL/
│
├── cld_brst_app.py              # Main Streamlit application
├── cloudburst_rf_model.pkl     # Trained Random Forest model
├── cloud_burst_final_features.csv  # Final consolidated dataset used for training
├── requirements.txt            # Project dependencies
│
├── notebooks/
│   ├── DataPrep.ipynb           # Data cleaning, merging, and preprocessing
│   ├── 02_Model_Training.ipynb  # Model training and evaluation
│   └── dataset_feature_dict.ipynb  # Feature definitions and analysis
│
├── data_sources/                # Raw and reference datasets (ingredients)
│   ├── HP_Cloudburst_Data_2010-2024.csv
│   ├── himachal_cloudburst_past_data.csv
│   └── Kullu_Mandi_DEM_merged.tif
│
└── README.md
```

Files such as plots, checkpoints, IDE configurations, and intermediate artifacts were used during development and experimentation and are not essential for deployment.

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/cloudburst-prediction-system.git
cd cloudburst-prediction-system
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux / Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Model Training Summary

* **Algorithm:** Random Forest
* **Feature Engineering:** Weather parameter normalization and labeling
* **Evaluation Metrics:**

  * Accuracy
  * Precision
  * Recall
  * F1-Score

Special emphasis was placed on **recall**, as missing a cloudburst event is more critical than false positives.

---

## 🔮 Future Enhancements

* Integration with **real-time weather APIs**
* Inclusion of additional districts in Himachal Pradesh
* Deep Learning models (LSTM / CNN-LSTM for time-series forecasting)
* Mobile-friendly alert system
* Government or disaster-management dashboard integration

---

## ⚠️ Limitations

* Limited availability of labeled cloudburst events
* Dependency on historical data quality
* Prediction accuracy may vary during extreme unseen conditions

---

## 🧩 Use Case Context

This system is designed as a **research and application-focused solution** in the domain of climate analytics and disaster risk prediction. It is suitable for demonstrations, experimentation, and portfolio showcasing.

---

## 📜 Disclaimer

This project is intended for **research and demonstration purposes**. Predictions generated by the system should not be considered official weather forecasts.

---

## 🙏 Acknowledgements

* NASA POWER Project
* India Meteorological Department (IMD)
* Open-source Python community

---

## 📬 Contact

**Developer:** Akshit Banyal
**Domain Interest:** Data Science | Machine Learning | Climate Analytics

---

⭐ If you find this project useful, consider giving it a star!
