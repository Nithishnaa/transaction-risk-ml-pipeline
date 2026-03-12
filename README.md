# 💳 Transaction Risk Analysis & ML Pipeline

> End-to-end fraud detection pipeline on real transaction data — SQL feature engineering, XGBoost vs Logistic Regression, MLflow experiment tracking, and actionable business risk patterns.

---

## 🔑 Key Results

| Metric | Value |
|---|---|
| Dataset | Sparkov Credit Card Transactions (Kaggle) |
| Total Records | 1,296,675 real-pattern transactions |
| Features Engineered | 15+ via SQL (CTEs, Window Functions, JOINs) |
| Logistic Regression ROC-AUC | 0.9309 |
| XGBoost ROC-AUC | 0.9992 |
| Improvement | 7.3% uplift over baseline |
| Fraud Rate | 0.5% (realistic fintech scenario) |

---

## 🚨 Top 3 High-Risk Patterns Found

| Pattern | Transactions Flagged | Fraud Rate | vs Baseline |
|---|---|---|---|
| Late-Night (0–5am) | 72,978 | 0.9% | 1.8x |
| Amount Spike >3x Customer Average | 14,318 | 9.2% | 18x |
| High-Value + High-Risk Category | 1,860 | 36.2% | 72x |

**Key Insight:** High-value transactions in `shopping_net` and `misc_net` categories have a 36% fraud rate — 72x the baseline. This segment should be prioritised for manual review or step-up authentication.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Loading & EDA | Python, Pandas, Plotly |
| Feature Engineering | SQL (SQLite) — CTEs, Window Functions, JOINs |
| ML Models | XGBoost, Logistic Regression, Scikit-learn |
| Experiment Tracking | MLflow |
| Environment | Jupyter Notebook |

---

## 📓 Notebooks

| Notebook | What it covers |
|---|---|
| `01_eda.ipynb` | Load real dataset, fraud distribution by category, hour, age group, amount analysis |
| `02_sql_features.ipynb` | 15+ risk features engineered via SQL — merchant risk, velocity features, customer behaviour patterns, risk segmentation with CTEs |
| `03_ml_model.ipynb` | Logistic Regression vs XGBoost, ROC curve comparison, MLflow tracking, feature importance, top 3 risk pattern report |

---

## 🚀 Run Locally

```bash
# 1. Download dataset from Kaggle
# kaggle.com/datasets/kartik2112/fraud-detection
# Place fraudTrain.csv and fraudTest.csv in this folder

# 2. Install dependencies
pip install pandas numpy plotly scikit-learn xgboost mlflow jupyter

# 3. Run notebooks in order
jupyter notebook 01_eda.ipynb
jupyter notebook 02_sql_features.ipynb
jupyter notebook 03_ml_model.ipynb

# 4. View MLflow experiment dashboard
mlflow ui
```

---

## 💡 Feature Engineering Highlights

All 15+ features were engineered via SQL before passing to Python — mirroring real production pipelines where features live in a data warehouse.

Key features include:
- `amt_vs_customer_avg` — amount relative to customer's historical average (strongest predictor)
- `customer_late_night_ratio` — proportion of customer's transactions at night
- `is_high_risk_category` — flag for shopping_net, misc_net, grocery_pos
- `near_customer_max` — transaction approaching customer's historical maximum
- `unique_states` — number of distinct states customer has transacted in
- `is_multi_state` — flag for customers transacting across 3+ states

---

## 📊 Model Comparison

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.9309 | Strong baseline, scaled features |
| XGBoost | 0.9992 | Best model, handles imbalanced data via scale_pos_weight |

XGBoost was selected as the final model. All experiments were tracked with MLflow for full reproducibility.

---

## 👩‍💻 About

Built by **Nithishna Saravana**
M.Sc. Data Science, Singapore University of Technology and Design (SUTD)
