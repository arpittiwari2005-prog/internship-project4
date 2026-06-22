# src/model.py
# Trains regression and classification models separately

import joblib
import os
import time
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, f1_score)

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, random_state=42, verbosity=0
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=42
    )
}

CLASSIFICATION_MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, random_state=42,
        verbosity=0, eval_metric='mlogloss'
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=42
    )
}

def train_regression(X_train, y_train, X_test, y_test, log):
    """Train all regression models — predict final grade"""
    results = {}
    log.write("=== REGRESSION (Final Grade Prediction) ===\n\n")

    for name, model in REGRESSION_MODELS.items():
        print(f"  Training Regressor: {name}...")
        start   = time.time()
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        mae     = mean_absolute_error(y_test, y_pred)
        r2      = r2_score(y_test, y_pred)
        elapsed = time.time() - start

        results[name] = {
            'model' : model,
            'y_pred': y_pred,
            'MAE'   : round(mae, 2),
            'R2'    : round(r2, 4)
        }
        line = f"{name}: MAE={mae:.2f}, R²={r2:.4f}, Time={elapsed:.1f}s"
        print(f"  {line}")
        log.write(line + "\n")

    best = min(results, key=lambda k: results[k]['MAE'])
    print(f"\n  ✅ Best Regressor: {best} (MAE={results[best]['MAE']})")
    log.write(f"\nBest Regressor: {best}\n\n")
    return best, results

def train_classification(X_train, y_train, X_test, y_test, log):
    """Train all classification models — predict grade category"""
    results = {}
    log.write("=== CLASSIFICATION (Grade Category Prediction) ===\n\n")

    for name, model in CLASSIFICATION_MODELS.items():
        print(f"  Training Classifier: {name}...")
        start    = time.time()
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average='weighted')
        elapsed  = time.time() - start

        results[name] = {
            'model'   : model,
            'y_pred'  : y_pred,
            'accuracy': round(accuracy * 100, 2),
            'f1'      : round(f1, 4)
        }
        line = (f"{name}: Accuracy={accuracy*100:.2f}%, "
                f"F1={f1:.4f}, Time={elapsed:.1f}s")
        print(f"  {line}")
        log.write(line + "\n")

    best = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  ✅ Best Classifier: {best} "
          f"(Accuracy={results[best]['accuracy']}%)")
    log.write(f"\nBest Classifier: {best}\n")
    return best, results

def train_all_models(X_train, y_train_g, y_test_g,
                     X_test, y_train_c, y_test_c,
                     log_path='logs/training_log.txt'):
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w') as log:
        log.write("=== STUDENT PERFORMANCE TRAINING LOG ===\n\n")
        print("\n── Regression Models ──────────────────")
        best_reg,  reg_results  = train_regression(
            X_train, y_train_g, X_test, y_test_g, log)
        print("\n── Classification Models ──────────────")
        best_clf,  clf_results  = train_classification(
            X_train, y_train_c, X_test, y_test_c, log)

    return best_reg, reg_results, best_clf, clf_results

def save_models(reg_model, clf_model):
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_model, 'models/grade_predictor.pkl')
    joblib.dump(clf_model, 'models/category_classifier.pkl')
    print("\nModels saved → models/")