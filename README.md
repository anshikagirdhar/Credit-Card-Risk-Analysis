# Credit Card Fraud Detection using Logistic Regression

## Overview
This project analyzes highly imbalanced credit card transaction data to detect fraudulent transactions. It applies exploratory data analysis (EDA) and under-sampling techniques to address class imbalance, then builds and evaluates a Logistic Regression model to classify transactions as legitimate or fraudulent.

## Key Highlights
- Analyzed highly imbalanced transaction data using Python (Pandas, NumPy), applying under-sampling techniques and exploratory data analysis to identify fraud/default patterns.
- Built and evaluated a Logistic Regression fraud detection model using scikit-learn, achieving **93.9% test accuracy** and evaluating performance with **precision, recall, F1-score, and ROC-AUC** to support financial risk mitigation.

## Dataset
- **File:** `creditcard.csv`
- Contains anonymized/PCA-transformed transaction features (`V1`–`V28`), along with `Time`, `Amount`, and the target label `Class` (0 = legitimate, 1 = fraud).
- Highly imbalanced: fraudulent transactions represent a very small fraction of total transactions.

## Approach
1. **Data Loading & Inspection**
   - Loaded the dataset with Pandas and inspected structure, data types, and missing values (`df.info()`, `df.isnull().sum()`).
2. **Exploratory Data Analysis (EDA)**
   - Examined class distribution (`df['Class'].value_counts()`).
   - Compared transaction amount statistics between legitimate and fraudulent transactions.
   - Analyzed feature means grouped by class (`df.groupby('Class').mean()`).
3. **Handling Class Imbalance (Under-sampling)**
   - Since fraudulent transactions are rare, randomly sampled an equal number of legitimate transactions to match the fraud count, creating a balanced dataset for training.
4. **Train-Test Split**
   - Split the balanced dataset into training (80%) and test (20%) sets using stratified sampling to preserve class ratios.
5. **Model Building**
   - Trained a **Logistic Regression** classifier (scikit-learn) on the balanced training data.
6. **Model Evaluation**
   - Evaluated performance on both training and test sets using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC-AUC
     - Classification report & confusion matrix

## Tech Stack
- Python
- Pandas, NumPy — data manipulation & EDA
- scikit-learn — model training & evaluation

## Results
| Metric | Value |
|---|---|
| Test Accuracy | 93.9% |
| Precision | See notebook output |
| Recall | See notebook output |
| F1-score | See notebook output |
| ROC-AUC | See notebook output |

> Run the notebook end-to-end with `creditcard.csv` in the project directory to reproduce the exact precision, recall, F1, and ROC-AUC values.

## How to Run
1. Place `creditcard.csv` in the same directory as the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Open and run `work.ipynb` in Jupyter Notebook / JupyterLab.

## Project Structure
```
.
├── work.ipynb        # Main notebook: EDA, under-sampling, model training & evaluation
├── creditcard.csv     # Dataset (not included — add your own copy)
└── README.md          # Project documentation
```

## Future Improvements
- Experiment with other algorithms (Random Forest, XGBoost) for comparison.
- Try alternative imbalance-handling techniques (SMOTE, class weighting) instead of random under-sampling.
- Perform hyperparameter tuning and cross-validation for more robust performance estimates.
