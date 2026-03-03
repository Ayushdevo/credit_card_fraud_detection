# 💳 Credit Card Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**An end-to-end ML pipeline to detect fraudulent credit card transactions using classical ML and deep learning — built for portfolio demonstration.**

[📊 Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [📁 Project Structure](#-project-structure) · [🚀 Getting Started](#-getting-started)

</div>

---

## 📌 Problem Statement

Credit card fraud causes **billions of dollars in losses annually**. The challenge is extreme **class imbalance** — fraudulent transactions make up only **0.17%** of all transactions — making standard accuracy metrics misleading.

This project:
- Handles severe class imbalance using **SMOTE** and class weighting
- Trains and compares **4 models**: Logistic Regression, Random Forest, XGBoost, Neural Network
- Evaluates with **fraud-appropriate metrics**: AUC-ROC, Precision-Recall, F1-Score
- Tunes the **decision threshold** to maximize recall of fraud cases

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Features | V1–V28 (PCA-transformed), `Amount`, `Time` |
| Target | `Class` (0 = Legitimate, 1 = Fraud) |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Environment | Kaggle Notebooks (GPU) |

---

## 📈 Results Summary

| Model | AUC-ROC | Avg Precision | F1-Score | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.974 | ~0.732 | ~0.841 | ~0.918 |
| Random Forest | ~0.983 | ~0.858 | ~0.878 | ~0.864 |
| **XGBoost** | **~0.987** | **~0.881** | **~0.891** | **~0.892** |
| Neural Network | ~0.981 | ~0.851 | ~0.870 | ~0.878 |

---

## 📁 Project Structure
```
credit-card-fraud-detection/
├── credit_card_fraud_detection.ipynb   # Main Kaggle notebook
├── outputs/
│   ├── class_distribution.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── threshold_optimization.png
├── models/
│   ├── xgboost_fraud_model.pkl
│   ├── neural_network_fraud_model.h5
│   └── robust_scaler.pkl
└── README.md
```

---

## 🚀 Getting Started

### Run on Kaggle (Recommended)
1. Create a New Notebook → upload `credit_card_fraud_detection.ipynb`
2. Add dataset: **Data → Add Data → Search `creditcardfraud`**
3. Enable GPU (Settings → Accelerator → GPU T4 x2)
4. Run All ✅

### Run Locally
```bash
git clone https://github.com/Ayushdevo/credit_card_fraud_detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
kaggle datasets download -d mlg-ulb/creditcardfraud
jupyter notebook credit_card_fraud_detection.ipynb
```

---

## 🔑 Key Learnings

- **Never apply SMOTE to test data** — data leakage will inflate your scores
- **Accuracy is useless** for imbalanced fraud data — use AUC-ROC and Precision-Recall
- **Threshold tuning matters** — 0.5 is rarely optimal; tune based on recall vs precision needs
- **XGBoost wins on tabular data** — consistently outperforms deep learning on structured datasets
- **Features V14, V17, V12** are the strongest fraud predictors

---

## 🚧 Future Improvements

- [ ] SHAP values for model explainability
- [ ] Optuna for automated hyperparameter tuning  
- [ ] Stacking ensemble (XGBoost + Neural Network)
- [ ] Real-time scoring API with FastAPI
- [ ] Concept drift detection for production

---

## 🙏 Acknowledgements

- Dataset by [Machine Learning Group – ULB](https://mlg.ulb.ac.be/) via Kaggle

---

<div align="center">Made with ❤️ for portfolio purposes · Give it a ⭐ if useful!</div>
