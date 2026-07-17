# Credit Card Risk Analysis

This project analyzes and models real credit card transactions to detect
potential fraud, using machine learning while explicitly managing the
precision/recall tradeoff inherent to severely imbalanced classification.

## 📂 Project Structure
```
├── credit card.ipynb   # Full analysis: EDA, preprocessing, modeling
├── creditcard.csv                    # Dataset (see download note below)
└── README.md
```

**Note:** the dataset is not included in this repository due to file size.
Download it from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
After downloading, place `creditcard.csv` in the project folder before
running the notebook.

## About the Dataset
Real, anonymized transactions made by European cardholders over two days
in September 2013 (Worldline / Universite Libre de Bruxelles Machine
Learning Group). 284,807 transactions, only 492 labeled fraudulent
(0.173% of all transactions) — a genuinely, severely imbalanced dataset.

Features V1–V28 are PCA-transformed components (original features are
confidential and cannot be released); `Time` and `Amount` are the only
untransformed features.

**Target variable:**
- 0 → Legitimate transaction
- 1 → Fraudulent transaction

## 🔑 Key Steps in the Notebook

**1. Data Exploration**
- Class distribution and severity of imbalance (0.173% fraud rate)
- Correlation between features and with the fraud label
- Transaction amount patterns by class

**2. Data Preprocessing**
- Confirmed zero missing values
- Scaled `Time` and `Amount` (the only non-PCA features)
- Stratified train/test split to preserve the real fraud rate in both sets
- **Note:** modeling uses a stratified 100,000-row subsample of the full
  284,807-row dataset, preserving the exact real-world fraud rate, purely
  as a compute-time tradeoff for this environment — the same code runs
  unchanged on the full dataset given more compute.

**3. Modeling & Evaluation**
- Logistic Regression, Decision Tree, Random Forest — all evaluated on
  Accuracy, Precision, Recall, F1, and ROC-AUC (not accuracy alone, which
  is meaningless at this imbalance level)
- Special focus on the precision/recall tradeoff via a decision-threshold
  sweep, since where you set that threshold is a business decision, not
  a modeling one

**4. Handling Imbalance with SMOTE**
- Applied to the training set only (never the test set, to avoid leakage)
- Compared directly against `class_weight`-balanced models — SMOTE
  improved ROC-AUC and recall but reduced precision in this run, a
  genuine, worth-discussing tradeoff rather than a strictly "better" result

**5. Results & Insights**
- Full model comparison table
- Feature importance from the Random Forest
- Concrete recommendations for a real fraud detection system, including
  the honest limitation that PCA-transformed features limit how
  actionable individual feature insights can be

## Results Summary
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.059 | 0.800 | 0.111 | 0.950 |
| Decision Tree | 0.219 | 0.743 | 0.338 | 0.871 |
| **Random Forest** | **0.807** | 0.714 | **0.758** | 0.886 |
| Random Forest + SMOTE | 0.551 | **0.771** | 0.643 | **0.963** |

## 🚀 How to Run
```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook Credit_Card_Risk_Analysis.ipynb
```

## 📈 Future Improvements
- XGBoost / LightGBM comparison
- Hyperparameter tuning (GridSearch/RandomizedSearch)
- Full-dataset training given more compute (subsample used here is a
  documented, deliberate tradeoff, not a data limitation)
- Deploy the best model as a real-time scoring API
