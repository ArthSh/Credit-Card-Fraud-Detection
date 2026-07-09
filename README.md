# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions using supervised learning techniques. This project addresses the challenge of severe class imbalance through SMOTE, evaluates multiple classification algorithms, and optimizes a Random Forest classifier using hyperparameter tuning to maximize fraud detection performance.

---

## Project Overview

Credit card fraud causes billions of dollars in financial losses every year. Since fraudulent transactions represent only a tiny fraction of all transactions, traditional machine learning models often become biased toward predicting legitimate transactions.

This project develops a robust fraud detection system capable of identifying fraudulent transactions while minimizing false positives. Multiple machine learning algorithms were compared, and Random Forest was selected as the final model after achieving the highest F1-score on the test set.

---

## Objectives

- Perform exploratory data analysis on highly imbalanced transaction data.
- Handle class imbalance using SMOTE.
- Compare multiple machine learning algorithms.
- Optimize the best-performing model using Randomized Search.
- Evaluate models using fraud-sensitive metrics such as Precision, Recall, F1-score, and ROC-AUC.
- Build a reproducible end-to-end machine learning workflow.

---

## Dataset

Dataset: **Credit Card Fraud Detection Dataset (Kaggle)**

Dataset Characteristics:

- 284,807 transactions
- 31 features
- 492 fraudulent transactions
- Class imbalance of approximately 0.17% fraud cases

Features include:

- Time
- Amount
- PCA-transformed variables (V1–V28)
- Target variable (`Class`)

Target:

- 0 → Legitimate Transaction
- 1 → Fraudulent Transaction

---

## Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Scaling
      │
      ▼
Train-Test Split
      │
      ▼
SMOTE Oversampling
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Final Random Forest Model
```

---

## EDA

The following analyses were performed:

- Missing value analysis
- Duplicate transaction removal
- Transaction amount distribution
- Fraud vs legitimate transaction analysis
- Correlation analysis
- Feature importance visualization
- Class imbalance visualization

---

## Machine Learning Models Evaluated

The following algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

## Model Performance

| Model | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------:|--------:|---------:|---------:|
| Logistic Regression | 0.0527 | 0.7273 | 0.0997 | 0.9657 |
| Decision Tree | 0.4500 | 0.5000 | 0.4735 | 0.8516 |
| **Random Forest** | **0.8990** | **0.7475** | **0.8161** | **0.9634** |
| XGBoost | 0.8458 | 0.6061 | 0.7032 | 0.9730 |
| LightGBM | 0.7500 | 0.7146 | 0.7317 | 0.9639 |
| CatBoost | 0.7895 | 0.4545 | 0.5824 | 0.9601 |

**Final Selected Model:** Random Forest

Reason for selection:

- Highest F1-score
- High precision with strong recall
- Excellent ROC-AUC
- Better balance between detecting fraud and minimizing false alarms

---

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

Accuracy alone was not used for model selection due to the highly imbalanced nature of the dataset.

---

## Repository Structure

```
Credit-Card-Fraud-Detection/

│
├── notebook/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── images/
│
├── data/
│
├── model/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
```

Move into the project directory

```bash
cd Credit-Card-Fraud-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib

---

## Key Learnings

During this project, I gained practical experience in:

- Handling severely imbalanced datasets
- Synthetic oversampling using SMOTE
- Comparative evaluation of multiple classification algorithms
- Hyperparameter optimization using Randomized Search
- Model evaluation using fraud-sensitive metrics
- Building reproducible machine learning pipelines

---

## Future Improvements

Possible extensions include:

- Real-time fraud detection pipeline
- Streamlit web application
- Explainable AI using SHAP
- Deep learning approaches
- Isolation Forest and Autoencoder-based anomaly detection
- Model deployment using Docker and FastAPI

---

## License

This project is licensed under the MIT License.

---

## Author

**Arth Shivhare**
I'm an ungergraduate student at Indian Institute of Technology, Bombay (India). I did this project, simply for my interest in ML-based applications in real-life challenges, one such as credit-card frauds. I wish to pursue this interest more and more.
