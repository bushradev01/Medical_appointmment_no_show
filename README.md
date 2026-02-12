# Medical Appointment No-Show Prediction  
## Data Preprocessing & Machine Learning Project

### 📖 Project Overview

This project presents the design and implementation of a complete data preprocessing and machine learning pipeline to predict patient no-shows using a real-world medical appointments dataset from Kaggle.

The objective of this project is to analyze patient appointment data, preprocess it effectively, and evaluate machine learning models to predict whether a patient will attend or miss a scheduled appointment.

---

## 📊 Dataset Information

- **Dataset Name:** Medical Appointment No Shows  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/biralavor/noshow-medical-appoinments-v02  
- **Total Records:** 49,593  
- **Total Features:** 26  
- **Target Variable:** `no_show`  
  - 1 → Patient did not attend  
  - 0 → Patient attended  

The dataset includes:
- Patient demographics
- Appointment details
- Health conditions
- Environmental (weather) factors

---

## 🛠 Project Phases

### 🔹 Phase 1: Data Exploration & Problem Framing
- Loaded and explored the dataset
- Identified missing values
- Detected outliers
- Converted target variable (Yes/No → 1/0)
- Dropped irrelevant columns

---

### 🔹 Phase 2: Feature Engineering & Transformation
- Label Encoding (ordinal features)
- One-Hot Encoding (nominal features)
- Feature Scaling using StandardScaler
- Dimensionality Reduction using PCA

---

### 🔹 Phase 3: Handling Missing & Noisy Data
- Missing values handled using KNN Imputation
- Outliers removed using:
  - Z-Score Method
  - IQR Method

---

### 🔹 Phase 4: Preprocessing Pipeline Design
A complete Scikit-learn pipeline was built using:
- ColumnTransformer
- KNN Imputer
- StandardScaler
- OneHotEncoder

This ensured consistent and safe preprocessing without data leakage.

---

### 🔹 Phase 5: Model Training & Evaluation

The dataset was split using stratified sampling (80-20 split).

Models Evaluated:
- Logistic Regression (with & without PCA)
- Random Forest (with & without PCA)

### 📈 Key Results

- Logistic Regression Accuracy: ~58–59%
- Random Forest Accuracy: ~88–89% (Best performing model)
- Random Forest without PCA gave the best overall performance
- Class imbalance significantly affected minority class ("yes") prediction

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score

---

## ⚠️ Key Observations

- The dataset is imbalanced (majority: attended)
- Random Forest performed significantly better than Logistic Regression
- PCA slightly reduced model performance
- Minority class detection remains a challenge

Future improvements could include:
- SMOTE (Synthetic Oversampling)
- Class weighting
- Threshold tuning

---

## 📂 Project Structure

Medical_appointment_no_show/
│
├── notebook.ipynb
├── DPP_Project_Report.pdf
└── README.md
---

## 🧪 Tools & Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## ▶️ How to Run

1. Download dataset from Kaggle.
2. Place it in the project directory.
3. Open `notebook.ipynb` in Google Colab or Jupyter.
4. Run all cells sequentially.

---

## 👩‍💻 Group Members

- Abiha Nadeem (BSE-001-2023)  
- Bushra Ashraf Bhatti (BSE-015-2023)  
- Khadija Malik (BSE-030-2023)  

Submitted to: Dr. Amir Arsalan  

---

## 📌 Conclusion

This project demonstrates the importance of systematic data preprocessing in machine learning. By building a robust preprocessing pipeline and evaluating multiple models, we identified Random Forest as the most effective approach for predicting patient no-shows. The project highlights real-world challenges such as missing data, outliers, and class imbalance in healthcare analytics.

