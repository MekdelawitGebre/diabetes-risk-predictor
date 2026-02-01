# 🩺 Diabetes Risk Prediction — End-to-End Machine Learning Project

 ### **Group Members: 16**

| No. | Full Name | ID Number |
| --- | --- | --- |
| 1 | Hintsete Hilawe | UGR/6054/15 |
| 2 | Mahider Nardos | UGR/4445/15 |
| 3 | Mekdelawit Gebre | UGR/1386/15 |
| 4 | Yeabsira Yibeltal | UGR/5562/15 |
| 5 | Yohannes Worku | UGR/2047/15 |
| 6 | Zufan Gebrehiwot | UGR/7674/15 |

---

## Overview

This project implements a **complete, end-to-end Machine Learning pipeline** to predict the likelihood of diabetes in patients using health indicators such as glucose level, BMI, insulin, and age.

It demonstrates a full ML workflow — from **data preprocessing and feature engineering** to **model training, evaluation, and deployment**.

**Live App:** [Diabetes Risk Predictor](https://diabetes-ml-cxtwdnpx4nkutk99ddamci.streamlit.app/)

---

## Problem Statement

Given patient diagnostic data, predict whether an individual is diabetic or non-diabetic. This is a **binary classification** problem where the target variable (`Outcome`) is defined as:

* **1** → Diabetic
* **0** → Non-Diabetic

---

## Motivation

Diabetes is a global health challenge. Early detection allows for lifestyle intervention and early treatment, significantly reducing long-term complications. This project showcases how **predictive modeling** can empower healthcare providers to make proactive, data-driven decisions.

---

## Dataset Description

**Source:** [Pima Indians Diabetes Dataset – Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

| Column | Description | Type |
| --- | --- | --- |
| **Pregnancies** | Number of pregnancies | Numeric |
| **Glucose** | Plasma glucose concentration | Numeric |
| **BloodPressure** | Diastolic blood pressure (mm Hg) | Numeric |
| **SkinThickness** | Triceps skin fold thickness (mm) | Numeric |
| **Insulin** | 2-Hour serum insulin (mu U/ml) | Numeric |
| **BMI** | Body Mass Index | Numeric |
| **DiabetesPedigreeFunction** | Family history function | Numeric |
| **Age** | Patient’s age (years) | Numeric |
| **Outcome** | Target (1 = diabetic, 0 = non-diabetic) | Binary |

* **Samples:** 768
* **Features:** 8 input + 1 target
* **Data Challenges:** Handled zero-value artifacts and class imbalance.

---

##  Data Preprocessing & Feature Engineering

### **Preprocessing Steps:**

* **Imputation:** Replaced zero values in medical columns with **median imputation**.
* **Cleaning:** Removed duplicates and handled missing entries.
* **Scaling:** Standardized features using `StandardScaler`.
* **Balancing:** Addressed class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).

### **New Engineered Features:**

* **AgeGroup:** Binned ages to capture non-linear trends.
* **BMI_Category:** Categorized into Underweight, Normal, Overweight, and Obese.
* **BMI_Risk:** Numeric risk scoring (1–4) based on clinical BMI ranges.

---

## Machine Learning Pipeline

### **1. Models Evaluated**

We compared three robust algorithms: **Logistic Regression**, **Random Forest**, and **XGBoost**.

### **2. Hyperparameter Tuning**

We utilized `GridSearchCV` to optimize the Random Forest model.

* **Best Parameters:** `{'max_depth': 7, 'n_estimators': 100}`

### **3. Evaluation Strategy**

A standard **Train/Validation/Test** split was used to ensure the model generalizes to unseen patient data.

---

##  Results & Analysis

### **Validation Performance**

Random Forest emerged as the most reliable model during the validation phase.

| Model | Accuracy | F1-Score | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.80 | 0.79 | 0.89 |
| **Random Forest** | **0.80** | **0.80** | **0.91** |
| XGBoost | 0.77 | 0.76 | 0.86 |

### **Final Test Performance (Unseen Data)**

The optimized Random Forest model achieved the following on the held-out test set:

* **Test Accuracy:** **83.5%** * **Test F1-Score:** **0.85** * **Test ROC-AUC:** **0.89**

#### **Confusion Matrix (Test Set)**

|  | Predicted Non-Diabetic | Predicted Diabetic |
| --- | --- | --- |
| **Actual Non-Diabetic** | 74 (TN) | 26 (FP) |
| **Actual Diabetic** | 7 (FN) | **93 (TP)** |

> **Key Takeaway:** The model demonstrates high sensitivity (Recall), successfully identifying 93 out of 100 actual diabetic cases. This is crucial in a clinical setting to minimize false negatives.

---

## How to Run

### **Local Setup**

1. **Clone & Install:**
```bash
git clone https://github.com/MekdelawitGebre/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.txt

```


2. **Execute Pipeline:**
```bash
python3 main.py

```


3. **Launch Dashboard:**
```bash
streamlit run app.py

```



### **Testing**

Run unit tests to ensure preprocessing integrity:

```bash
pytest -q

```

---

##  Project Structure

```text
diabetes-risk-predictor/
├── data/               # Raw and processed CSV files
├── models/             # Saved .pkl files
├── notebooks/          # Exploratory Data Analysis (EDA)
├── plots/              # EDA Correlation heatmaps, ROC curves
├── src/
│   ├── data/           # Preprocessing & cleaning logic
│   └── models/         # Training & evaluation scripts
├── tests/              # Pytest unit tests
├── main.py             # Pipeline orchestrator
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
└── .gitignore          # Files excluded from Git

```

---

##  Conclusion

By integrating **SMOTE** and custom features like **BMI_Risk**, we developed a Random Forest model capable of an **83.5% accuracy** and a high **0.91 ROC-AUC** on validation. This pipeline provides a reliable foundation for identifying high-risk patients effectively.

---
