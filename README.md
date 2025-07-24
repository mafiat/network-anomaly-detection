# Network Anomaly Detection - Final Project

This project implements and compares various **unsupervised learning techniques** for network intrusion detection using the [Network Intrusion Detection dataset from Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection). The goal is to identify anomalous or malicious network activities without relying on labeled data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Unsupervised Models](#unsupervised-models)
- [Supervised Baselines](#supervised-baselines)
- [Results & Analysis](#results--analysis)
- [Key Insights](#key-insights)
- [References](#references)

---

## Project Overview

- **Objective:** Detect network anomalies using unsupervised learning.
- **Approach:** Analyze network traffic features, preprocess data, apply clustering and anomaly detection algorithms, and evaluate their effectiveness.
- **Dataset:** Derived from the KDD Cup 1999 dataset, containing 41 features and a binary label (`normal` or `anomaly`).

---

## Dataset

- **Source:** [Kaggle - Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
- **Features:** 41 network traffic features (categorical & numerical).
- **Target:** `class` (normal/anomaly).
- **Files:** `Train_data.csv` (with labels), `Test_data.csv` (unlabeled).

---

## Workflow

1. **Exploratory Data Analysis (EDA):**  
    - Statistical summaries, feature distributions, and correlation analysis.

2. **Data Cleaning:**  
    - Remove missing values and duplicates.
    - Drop zero-variance and highly correlated features.

3. **Preprocessing:**  
    - One-hot encode categorical variables.
    - Scale numerical features.

4. **Dimensionality Reduction:**  
    - Apply PCA to reduce feature space while retaining 95% variance.

5. **Model Training & Evaluation:**  
    - Apply unsupervised models: K-Means, Isolation Forest, Autoencoder, DBSCAN, Hierarchical Clustering, GMM.
    - Compare with supervised baselines: Logistic Regression, Random Forest, SVC.

---

## Unsupervised Models

- **K-Means Clustering**
- **Isolation Forest**
- **Autoencoder**
- **DBSCAN**
- **Hierarchical Clustering**
- **Gaussian Mixture Model (GMM)**

---

## Supervised Baselines

- **Logistic Regression**
- **Random Forest**
- **Support Vector Classifier (SVC)**

---

## Results & Analysis

- **Best Unsupervised Models:**  
  - K-Means and Autoencoder achieved the highest accuracy and ROC AUC among unsupervised methods.
- **Supervised Models:**  
  - Outperformed unsupervised models, as expected, with Random Forest and SVC achieving >98% accuracy.
- **Dimensionality Reduction:**  
  - PCA improved clustering efficiency but may reduce neural network performance if too aggressive.

See the [Results and Analysis](#results--analysis) section for detailed metrics and discussion.

---

## Key Insights

- Unsupervised learning can effectively detect anomalies in network traffic, especially with proper preprocessing.
- K-Means and Autoencoders are robust choices for unsupervised anomaly detection in this context.
- Supervised models provide higher accuracy but require labeled data.

---

## References

- [Kaggle Dataset: Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
- [KDD Cup 1999 Data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- Scikit-learn, TensorFlow, Seaborn, Matplotlib documentation

---

*For more details, see the full notebook and code above.*