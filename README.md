#  CreditPathAI — Credit Default Risk Prediction

## 📌 Table of Contents

- [1. Overview](#1-overview)
- [2. What This Project Does](#2-what-this-project-does)
- [3. Business Impact](#3-business-impact)
- [4. Supported Datasets](#4-supported-datasets)
- [5. System Architecture](#5-system-architecture)
- [6. Machine Learning Pipeline](#6-machine-learning-pipeline-flow)
- [7. Quick Start (Run Locally)](#7-quick-start-run-locally)
- [8. Contribution Guidelines](#8-contribution-guidelines)
- [9. Known Limitations & Expected Behavior](#9-known-limitations)
- [10. Project Structure](#10-project-structure)
- [11. Results Summary](#11-results-summary)
- [12. Tech Stack](#12-tech-stack)
- [13. Author](#13-author)
- [14. License](#14-license)

---




**CreditPathAI** is an end-to-end machine learning system for **credit default risk prediction**.  
It models borrower repayment behavior using structured financial data and follows **production-style ML engineering practices**.

> **Core idea:** keep dataset-specific ingestion separate from a **shared, reusable ML engine** so training, evaluation, and inference remain consistent and safe.

---

## 1. Overview

CreditPathAI helps financial institutions **identify high-risk borrowers early** by estimating the probability of loan default.  
The system is modular, reproducible, and deployment-ready, with a Streamlit app for interactive predictions.

---

## 2. What This Project Does

- Predicts **loan default probability**
- Enforces **inference safety** (feature order locking + serialized preprocessing)
- Supports **multiple datasets** with a shared ML pipeline
- Provides an **interactive Streamlit application**


## 3. Business Impact

- Enables early identification of high-risk borrowers  
- Supports proactive intervention instead of reactive recovery  
- Improves underwriting efficiency and risk-based decision making  



---

## 4. Supported Datasets

### Kaggle Loan Default Dataset
- **Source:** Kaggle  
- **Target:** `repay_fail` (0 = No Default, 1 = Default)  
- **Type:** Tabular borrower & financial features  

### Microsoft Loan Dataset
- **Source:** Microsoft sample loan data  
- **Target:** Default indicator  
- **Type:** Relational borrower & loan tables  

Each dataset is processed independently while reusing the same ML pipeline to prevent data leakage.

---

## 5. System Architecture

- Dataset-specific ingestion & artifact storage  
- Shared preprocessing, modeling, evaluation, and inference logic  
- Strict separation between **training** and **inference**  

### 6. Machine Learning Pipeline Flow 

```text
Raw Data
   ↓
Ingestion
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
Saved Artifacts
   ↓
Inference / Streamlit App
```

---

## Machine Learning Pipeline

### Ingestion
- Load CSV / TXT files  
- Merge relational tables when required  
- Validate schema consistency  

### Preprocessing
- Missing value handling  
- Categorical encoding  
- Feature scaling  
- Feature order locking for inference  

### Models Supported
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  

### Evaluation
- Accuracy, Precision, Recall, F1-score  
- ROC-AUC, PR-AUC  
- Confusion Matrix  
- Feature importance (tree-based models)  

### Inference
- Uses saved preprocessing artifacts  
- Enforces strict feature alignment  
- Production-ready serialized models  

---

## 7. Quick Start (Run Locally)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the ML Pipeline
```bash
python main.py
```

### Launch Streamlit App
```bash
streamlit run streamlit/streamlit_app.py
```

---

## 8. Contribution Guidelines

- Keep ML logic modular and dataset-agnostic  
- Do not hardcode file paths inside core logic  
- Ensure feature order consistency for inference  
- Add clear docstrings for new modules  
- Test changes before submitting a pull request  

---

## 9. Known Limitations & Expected Behaviors

### Known Limitations

- Hyperparameter tuning is minimal  
- Feature engineering is dataset-specific  
- Streamlit app is for demonstration, not large-scale deployment  

### Expected Behaviors
- Metrics may vary due to randomized splits  
- SMOTE is applied **only during training**  
- Streamlit app assumes pre-generated artifacts  

---

## 10. Project Structure

```text
CreditPathAI/
│
├── src/                    # Reusable ML engine
│   ├── ingestion.py
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
│
├── kaggle_dataset/
│   ├── data/
│   ├── notebooks/
│   ├── artifacts/
│   └── README.md
│
├── microsoft_dataset/
│   ├── data/
│   ├── notebooks/
│   ├── artifacts/
│   └── README.md
│
├── streamlit/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 11. Results Summary

- Tree-based models outperform linear models  
- Class imbalance handled using SMOTE (training only)  
- ROC-AUC used as the primary metric  

Detailed model comparison metrics are available in dataset-specific reports.


---

## 12. Tech Stack

- Python  
- Pandas, NumPy  
- scikit-learn, imbalanced-learn  
- ML models 
- Matplotlib, Seaborn  
- Joblib  
- Streamlit  

---

## 13 Author

Name : **RAJA RISHI KUMAR V**
Contact : For questions or suggestions, please open an issue on **GitHub**.

---

## 14. License

See the `LICENSE` file.
