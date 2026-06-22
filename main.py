# main.py
# Run this to execute the complete student performance pipeline

from src.data_preprocessing import load_and_preprocess
from src.model import train_all_models, save_models
from src.evaluate import (plot_regression_comparison,
                           plot_classification_comparison,
                           plot_actual_vs_predicted,
                           plot_confusion_matrix,
                           plot_feature_importance,
                           plot_grade_distribution,
                           print_report)

print("=" * 55)
print("   STUDENT PERFORMANCE TRACKER — ML PIPELINE")
print("=" * 55)

# Step 1: Load and preprocess
(X_train, X_test,
 y_train_g, y_test_g,
 y_train_c, y_test_c) = load_and_preprocess('data/student_data.csv')

# Step 2: Train all models
(best_reg, reg_results,
 best_clf, clf_results) = train_all_models(
    X_train, y_train_g, y_test_g,
    X_test,  y_train_c, y_test_c
)

# Step 3: Save both models
best_reg_model = reg_results[best_reg]['model']
best_clf_model = clf_results[best_clf]['model']
save_models(best_reg_model, best_clf_model)

# Step 4: Generate all plots
plot_regression_comparison(reg_results)
plot_classification_comparison(clf_results)
plot_actual_vs_predicted(
    y_test_g, reg_results[best_reg]['y_pred'], best_reg)
plot_confusion_matrix(
    y_test_c, clf_results[best_clf]['y_pred'], best_clf)
plot_feature_importance(
    best_reg_model, list(X_train.columns),
    best_reg, 'feature_importance_regression.png')
plot_feature_importance(
    best_clf_model, list(X_train.columns),
    best_clf, 'feature_importance_classification.png')
plot_grade_distribution(
    y_test_g, reg_results[best_reg]['y_pred'])
print_report(
    y_test_c, clf_results[best_clf]['y_pred'], best_clf)

print("\n" + "=" * 55)
print(f"DONE!")
print(f"Best Regressor   : {best_reg}")
print(f"Best Classifier  : {best_clf}")
print("Check output_images/ for all graphs")
print("=" * 55)