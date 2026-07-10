# 🩺 Diabetes Prediction Dashboard

An end-to-end Machine Learning and Power BI project that predicts diabetes risk from patient health metrics — covering the full workflow from raw data to an interactive business-facing dashboard.

## 📌 Overview

This project uses the Pima Indians Diabetes dataset to build, compare, and deploy a classification model that predicts whether a patient is likely to have diabetes, then surfaces the model's predictions and performance through an interactive Power BI dashboard with AI-driven insights.

## 🛠️ Technologies

- **Python** — Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib
- **Power BI** — Interactive dashboarding, Key Influencers, Decomposition Tree
- **Jupyter Notebook** — Data cleaning, EDA, feature engineering, modeling

## 🔁 Workflow

### 1. Data Cleaning
The raw dataset (768 rows × 9 columns, no true nulls) contained biologically invalid zero-values standing in for missing data:

| Column | Invalid (0) Values |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

These zeros were replaced with `NaN`, then imputed using the **median** of each column (`fillna(median(), inplace=True)`), preserving the distribution shape without letting outliers skew the fill value. Verified with `df.isnull().sum()` and `df.info()` post-cleaning — final dataset: 768 non-null rows across all 9 columns.

### 2. Exploratory Data Analysis (EDA)
- **Class balance:** 500 non-diabetic (Outcome = 0) vs. 268 diabetic (Outcome = 1) — a moderate imbalance factored into model evaluation
- **Distributions:** Histograms across all features showed right-skew in Insulin, SkinThickness, DiabetesPedigreeFunction, and Age; Glucose and BMI were closer to normal
- **Outlier detection:** Boxplots flagged Insulin as the most outlier-heavy feature, with BMI, BloodPressure, and SkinThickness also showing some extreme values
- **Correlation heatmap:** Glucose (0.47) and BMI (0.29) had the strongest correlation with Outcome; Age and Pregnancies were also positively correlated (0.54 between Age and Pregnancies themselves)
- **Bivariate analysis:** Boxplots of Glucose vs. Outcome and BMI vs. Outcome both showed visibly higher medians in the diabetic group
- **Pairplot:** Full pairwise scatter matrix (hue = Outcome) confirmed Glucose as the most visually separable feature between classes

### 3. Feature Engineering
Train/test split (80/20, stratified) and feature standardization using `StandardScaler`.

### 4. Model Training & Comparison
Trained six classifiers on the same split:

   | Model | Accuracy |
   |---|---|
   | Logistic Regression | 0.7143 |
   | Decision Tree | 0.7273 |
   | KNN | 0.7013 |
   | SVM | 0.7532 |
   | XGBoost | 0.7338 |
   | **Random Forest** | **0.7597** ✅ |

### 5. Model Selection
Random Forest selected as the best-performing model (`n_estimators=100, random_state=42`).

### 6. Evaluation
Confusion matrix, classification report, ROC/AUC, and 5-fold cross-validation.

### 7. Feature Importance
Ranked drivers of prediction: Glucose, BMI, Age, DiabetesPedigreeFunction, BloodPressure, Pregnancies, Insulin, SkinThickness.

### 8. Risk Scoring
Added `Predicted_Outcome`, `Prediction_Probability`, and a `Risk_Level` classification (Low / Medium / High) based on probability thresholds.

### 9. Deployment Prep
Model and scaler serialized with `joblib` (`Diabetes_Model.pkl`, `diabetes_scaler.pkl`); predictions exported to `Diabetes_Predictions.csv`.

### 10. Power BI Dashboard
4-page interactive report (Executive Dashboard, Patient Analysis, Prediction Analysis, AI & ML Insights) built on top of the exported predictions.

## 📊 Model Performance (Random Forest)

- **Accuracy:** 0.76
- **AUC Score:** 0.81
- **Cross-Validation:** 5-fold average accuracy of 0.767 (scores ranged 0.740–0.837)
- **Precision / Recall (Class 1 – Diabetic):** 0.68 / 0.59
- **Confusion Matrix:** 85 True Negatives, 15 False Positives, 22 False Negatives, 32 True Positives

## 📈 Power BI Dashboard Features

- **Feature Importance** chart driven directly from the trained model
- **ROC Curve** and **Confusion Matrix** visuals for model transparency
- **Key Influencers** — AI visual identifying that Insulin > 235 and rising Glucose most increase predicted diabetes likelihood
- **Top Segments / Decomposition Tree** — breakdown of predictions by Risk Level (High / Medium / Low)
- **Model Performance card** — Accuracy, Precision, Recall, F1, and AUC at a glance
- Filterable by Risk Level, Predicted Outcome, and Age Range

## 📂 Project Structure

```
Diabetes/
├── data_cleaning.ipynb
├── eda.ipynb
├── feature_engineering.ipynb
├── diabetes.csv
├── Diabetes_Predictions.csv
├── Diabetes_Model.pkl
├── diabetes_scaler.pkl
└── diabetes_report.pbix
```

## 🎯 Key Takeaways

- Ensemble methods (Random Forest) outperformed linear and distance-based models on this dataset
- Glucose and BMI were consistently the strongest predictors of diabetes risk — this held true from the raw correlation heatmap (0.47 and 0.29 with Outcome) through to the model's feature importance and Power BI's Key Influencers analysis, giving strong end-to-end validation of the signal
- Median imputation on the ~50% of Insulin values and ~30% of SkinThickness values that were invalid (recorded as 0) was a key data quality decision — the dataset couldn't be modeled reliably without addressing this
- Combining a Python ML pipeline with Power BI's AI visuals allowed model insights to be delivered in a business-friendly, decision-support format rather than a static notebook

---

**Author:** Mkhululi van Heerden
🔗 [Portfolio](https://datascienceportfol.io/mkhululivanheerden16) • [GitHub](https://github.com/mkhululi160)
