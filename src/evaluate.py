# src/evaluate.py
# All evaluation plots for student performance tracker

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs('output_images', exist_ok=True)
GRADE_LABELS = ['F', 'D', 'C', 'B', 'A']

def plot_regression_comparison(reg_results):
    """Bar chart of MAE and R² across regression models"""
    names = list(reg_results.keys())
    maes  = [reg_results[n]['MAE'] for n in names]
    r2s   = [reg_results[n]['R2']  for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(names, maes, color='#4C72B0')
    axes[0].set_title('Regression — MAE (lower is better)')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    axes[1].bar(names, r2s, color='#55A868')
    axes[1].set_title('Regression — R² Score (higher is better)')
    axes[1].set_ylabel('R² Score')
    axes[1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.005, str(v), ha='center', fontweight='bold')

    plt.suptitle('Grade Prediction — Regression Model Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_images/regression_comparison.png', dpi=150)
    plt.close()
    print("Saved: regression_comparison.png")

def plot_classification_comparison(clf_results):
    """Bar chart of accuracy and F1 across classifiers"""
    names = list(clf_results.keys())
    accs  = [clf_results[n]['accuracy'] for n in names]
    f1s   = [clf_results[n]['f1'] * 100 for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width,
                   label='Accuracy %', color='#4C72B0')
    bars2 = ax.bar(x + width/2, f1s,  width,
                   label='F1 Score %', color='#DD8452')

    ax.set_title('Grade Category Classification — Model Comparison')
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 115)
    ax.bar_label(bars1, fmt='%.1f', padding=2)
    ax.bar_label(bars2, fmt='%.1f', padding=2)

    plt.tight_layout()
    plt.savefig('output_images/classification_comparison.png', dpi=150)
    plt.close()
    print("Saved: classification_comparison.png")

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Scatter plot of actual vs predicted grades"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', edgecolors='white')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    plt.title(f'Actual vs Predicted Final Grade — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_images/actual_vs_predicted.png', dpi=150)
    plt.close()
    print("Saved: actual_vs_predicted.png")

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Confusion matrix for grade category classifier"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GRADE_LABELS,
                yticklabels=GRADE_LABELS,
                linewidths=1, linecolor='white')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.tight_layout()
    plt.savefig('output_images/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

def plot_feature_importance(model, feature_names, model_name, filename):
    """Feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Skipping feature importance for {model_name}")
        return
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:12]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)),
            importances[indices], color='steelblue')
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title(f'Top Feature Importances — {model_name}')
    plt.tight_layout()
    plt.savefig(f'output_images/{filename}', dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def plot_grade_distribution(y_test, y_pred_reg):
    """Distribution of actual vs predicted grades"""
    plt.figure(figsize=(10, 5))
    plt.hist(y_test,     bins=20, alpha=0.6,
             color='steelblue', label='Actual Grades')
    plt.hist(y_pred_reg, bins=20, alpha=0.6,
             color='coral',     label='Predicted Grades')
    plt.xlabel('Grade')
    plt.ylabel('Count')
    plt.title('Grade Distribution — Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_images/grade_distribution.png', dpi=150)
    plt.close()
    print("Saved: grade_distribution.png")

def print_report(y_test, y_pred, model_name):
    print(f"\n--- Classification Report: {model_name} ---")
    print(classification_report(y_test, y_pred,
                                target_names=GRADE_LABELS))