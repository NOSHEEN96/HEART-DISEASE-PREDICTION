# 💓 Heart Disease Prediction – Machine Learning Pipeline

### Author: Nosheen Akhter

This repository presents a **complete end-to-end machine learning pipeline** for predicting heart disease from clinical data.  
It demonstrates **data preprocessing, model evaluation, and explainable AI techniques**, making it ideal for both academic research and real-world applications in healthcare analytics.


## Project Overview

The goal of this project is to **classify patients as having heart disease or not** based on their medical attributes such as age, cholesterol, blood pressure, etc.  
The project uses **supervised learning algorithms** and a robust preprocessing + evaluation pipeline built entirely with **Python and scikit-learn**.

---

## ⚙️ Key Features

✅ Data preprocessing with **imputation, scaling, and encoding**  
✅ Model comparison using **Logistic Regression, Random Forest, SVC, and Gradient Boosting**  
✅ **3-Fold Cross Validation** with ROC-AUC and Accuracy metrics  
✅ **Explainable AI (XAI)** via permutation feature importance  
✅ Visualization of **ROC Curve** and **Confusion Matrix**  
✅ Ready-to-use `predict_one()` function for single-patient prediction  
✅ Easy integration with clinical dashboards or Flask APIs  

---

## 📊 Results Summary

| Metric | Value |
|:--|:--|
| **Best Model** | Random Forest |
| **Test Accuracy** | 0.82 |
| **ROC-AUC Score** | 0.91 |
| **Top Features (Permutation Importance)** | Age, Cholesterol, Thalach (Max Heart Rate), Oldpeak, Chest Pain Type |

**Confusion Matrix and ROC Curve:**
- `roc_curve.png`
- `confusion_matrix.png`

---

## 🧩 Data Description

The dataset used (`heart_disease_data.csv`) includes features such as:

| Feature | Description |
|:--|:--|
| age | Age of the individual |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol in mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia (normal, fixed defect, reversible defect) |
| target | 1 = heart disease, 0 = healthy |

---
