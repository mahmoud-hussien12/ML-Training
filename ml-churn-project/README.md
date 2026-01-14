# Customer Churn Prediction — End-to-End ML Pipeline

## 📌 Project Overview

Customer churn is one of the most critical problems for subscription-based businesses.  
Acquiring a new customer is significantly more expensive than retaining an existing one.

This project builds an **end-to-end machine learning system** to **predict customer churn** using structured (tabular) data.  
The model outputs a **probability of churn**, enabling CRM systems to proactively:

- offer discounts or promotions
- improve customer support
- resolve service issues early

This helps **reduce churn rate** and **increase overall business profitability**.

---

## 🧠 Business Problem

**Goal:**  
Identify customers who are likely to churn so the business can take preventive actions.

**Why churn prediction matters:**
- Churn directly impacts revenue
- Early detection allows targeted retention strategies
- Even small churn reductions can significantly increase profits

---

## 📊 Dataset

- **Source:** Telco Customer Churn dataset
- **Size:** ~7,000 customers
- **Target:** `Churn`  
  - `1` → customer churned  
  - `0` → customer retained

### Feature Types

- **Categorical features (16):**
```

gender, SeniorCitizen, Partner, Dependents, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod

```

- **Numerical features:**
```

tenure, MonthlyCharges, TotalCharges

```

The project explicitly separates **categorical and numerical features** to ensure correct preprocessing and avoid data leakage.

---

## 🏗️ Project Structure

```

ml-churn-project/
├── data/
│   └── raw/
├── configs/
│   └── base.yaml
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── pipeline.py
│   ├── evaluation/
│   ├── inference/
│   ├── utils.py
│   ├── config.py
│   └── logger.py
├── tests/
├── train.py
├── predict.py
├── README.md
└── MENTORSHIP_STATUS.md

````

---

## ⚙️ ML Pipeline Design

This project follows **production-oriented ML engineering practices**:

### 1️⃣ Configuration-driven execution
- All paths, parameters, and settings are controlled via YAML config files
- Enables reproducibility and clean experiment tracking

### 2️⃣ Data handling
- Raw data loading with validation
- Stratified train/validation/test split to handle class imbalance
- No preprocessing before splitting (prevents data leakage)

### 3️⃣ Feature preprocessing
Implemented using `sklearn` pipelines:

- **Numerical features**
  - Median imputation
  - Standard scaling

- **Categorical features**
  - Most-frequent imputation
  - One-hot encoding (`handle_unknown="ignore"`)

### 4️⃣ Modeling
- Baseline model: **Logistic Regression**
- Entire preprocessing + model combined in a single `Pipeline`
- Allows easy model swapping (e.g., XGBoost, LightGBM)

---

## 📈 Evaluation Metrics

Evaluation is performed on the **validation set** only.

### Metrics used:
- **ROC-AUC** (primary metric)
- Precision
- Recall

### Business reasoning:
- Recall for churners is important to avoid missing customers who are likely to leave
- Precision–recall trade-offs are discussed from a business perspective

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
poetry install
```

### 2️⃣ Train the model
* Models are evaluated using stratified cross-validation and tracked with MLflow for reproducibility and comparison.
```bash
python train.py --config configs/base.yaml
```

### 3️⃣ Threshold tuning

```bash
python threshold.py --config configs/base.yaml # threshold tuning
```

### 4️⃣ Run inference (example)

```bash
python predict.py --input data/raw/sample.csv
```

---

## 📦 Artifacts

After training, the following artifacts are saved:

* trained model
* preprocessing pipeline

These artifacts can be reused for:

* batch predictions
* REST API deployment
* integration with CRM systems

---

### Input Validation & Monitoring

The API validates input feature schemas to prevent malformed requests and logs prediction statistics for basic monitoring and debugging.

----

## 🔍 Key Engineering Decisions

* Used **stratified splits** due to class imbalance
* Kept preprocessing inside sklearn pipelines to avoid leakage
* Chose Logistic Regression as a strong, interpretable baseline
* Avoided notebooks for training to simulate real production workflows
* Model selection is performed based on cross-validated ROC-AUC and business considerations, with the final production model logged separately in MLflow.
* Selected top features using feature importance for lightgbm and cofficient for logistic regression
* Selected threshold based on recall
---

## 🔮 Future Improvements

If this were extended further:

* Hyperparameter tuning with cross-validation
* Gradient boosting models (XGBoost / LightGBM)
* Threshold optimization based on business costs
* Model monitoring and drift detection
* Deployment as a FastAPI service

---

### Production Considerations

* Strict feature schema validation to prevent malformed inference requests
* Logging of prediction outputs for basic monitoring and debugging
* Health check endpoint for deployment readiness
* Config-driven model selection and parameters

---

### LLM Integration (Work in Progress)

* Modular LLM client abstraction
* Prompt templates stored separately
* Designed for future API-based LLM providers
* Focus on explainability for business users
* Integrated LLM-based explanation service with prompt templating, API abstraction, cost controls, and fallback handling.

---

### LLM Safety & Cost Controls

* Latency measurement
* Token usage logging
* Confidence-based gating
* Timeout & graceful fallback

---

## 👤 Author

Built by a senior software engineer transitioning into Machine Learning Engineering,
with a focus on **production-quality ML systems**, not just model accuracy.

```
“This project is under active development.”
