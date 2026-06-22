# 🎓 Student Performance Tracker using Machine Learning

> Predicting student final grades and grade categories using academic, behavioral, and demographic features with multiple ML models.

---

## 📋 Project Details

| Field | Details |
|---|---|
| **Intern ID** | CITS1939 |
| **Full Name** | Arpit Tiwari |
| **No. of Weeks** | 4 Weeks |
| **Project Name** | Student Performance Tracker |
| **Project Scope** | Build and compare multiple ML models to predict student final grades (regression) and grade categories A/B/C/D/F (classification) based on study habits, attendance, and personal factors |

---

## 📁 Project Structure
---

## 🧠 Models Used & Results

### Regression — Final Grade Prediction

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | 4.42 | 0.9040 |
| Ridge Regression | 4.42 | 0.9041 |
| XGBoost | 5.56 | 0.8480 |
| Gradient Boosting | 5.23 | 0.8654 |
| Random Forest | 6.12 | 0.8124 |

✅ **Best Regressor: Linear Regression** (MAE = 4.42, R² = 0.904)

### Classification — Grade Category Prediction

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | 61.51% | 0.5824 |
| XGBoost | 63.92% | 0.6278 |
| Gradient Boosting | 64.60% | 0.6273 |

✅ **Best Classifier: Gradient Boosting** (Accuracy = 64.60%)

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/arpittiwari2005-prog/internship-project4.git
cd internship-project4
```

### 2. Create virtual environment
```bash
python -m venv venv --without-pip
venv\Scripts\activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset
```bash
python generate_data.py
```

### 5. Train all models
```bash
python main.py
```

### 6. Predict a new student
```bash
python src/predict.py
```

---

## 📊 Output Graphs

| Graph | Description |
|---|---|
| `regression_comparison.png` | MAE and R² comparison across regression models |
| `classification_comparison.png` | Accuracy and F1 comparison across classifiers |
| `actual_vs_predicted.png` | Scatter plot of actual vs predicted final grades |
| `confusion_matrix.png` | Predicted vs actual grade categories |
| `feature_importance_classification.png` | Most influential features for grade classification |
| `grade_distribution.png` | Actual vs predicted grade distribution histogram |

See the `output_images/` folder for all generated plots.

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `age` | Student age (15–22) |
| `gender` | Male / Female |
| `study_hours` | Daily study hours (0–10) |
| `attendance` | Attendance percentage (40–100%) |
| `prev_grade` | Previous semester grade |
| `parent_education` | Parent education level |
| `internet_access` | Whether student has internet access |
| `extra_classes` | Whether attending extra coaching |
| `sleep_hours` | Average sleep hours per day |
| `stress_level` | Low / Medium / High |
| `assignments_done` | Number of assignments completed |
| `library_visits` | Number of library visits |
| `sports` | Whether involved in sports |
| `part_time_job` | Whether doing a part-time job |
| `study_efficiency` | Engineered — study hours × attendance rate |
| `engagement_score` | Engineered — weighted academic engagement |
| `good_sleep` | Engineered — optimal sleep flag (7–9 hrs) |
| `grade_gap` | Engineered — room for improvement |

---

## 🏷️ Target Variables

| Target | Type | Values |
|---|---|---|
| `final_grade` | Regression | 0 – 100 |
| `grade_category` | Classification | F, D, C, B, A |
| `pass_fail` | Binary | 0 = Fail, 1 = Pass |

---

## 📦 Dependencies

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

---

## 📸 Screenshots

See the `screenshots/` folder for:
- VS Code project structure
- Terminal training output
- Generated graph previews

---

## 📝 Documentation

This project builds a dual ML pipeline for student performance tracking:

1. **Data Generation** — Simulates 2000 realistic student records with academic features (study hours, attendance, assignments), behavioral features (sleep, stress, sports), and demographic features (age, gender, parent education). Final grades are generated using a weighted scoring formula with added noise for realism.

2. **Feature Engineering** — Derives high-value features including study efficiency (study hours weighted by attendance), academic engagement score (assignments, library visits, extra classes), good sleep flag (7–9 hours), and grade improvement gap.

3. **Dual Pipeline** — Two separate ML pipelines run in parallel: a regression pipeline predicting the exact final grade (0–100) and a classification pipeline predicting the grade category (A/B/C/D/F).

4. **Regression Results** — Linear Regression achieved the best MAE of 4.42 with an R² of 0.904, meaning the model explains 90% of variance in student grades — an outstanding result.

5. **Classification Results** — Gradient Boosting achieved 64.6% accuracy on the 5-class grade category prediction. The model performs best on A and B grades due to higher representation in the dataset.

6. **Evaluation** — Generates regression comparison charts, classification comparison charts, actual vs predicted scatter plots, confusion matrix, feature importance, and grade distribution histograms.

7. **Prediction** — Both saved models (grade_predictor.pkl and category_classifier.pkl) can instantly predict a new student's expected grade and category from their profile.

---

## 👤 Author

**Arpit Tiwari**
Intern — CODTECG
GitHub: [@arpittiwari2005-prog](https://github.com/arpittiwari2005-prog)
