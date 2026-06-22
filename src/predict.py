# src/predict.py
# Predict student grade and category for a new student

import joblib
import pandas as pd
import numpy as np

GRADE_MAP = {0: 'F', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}

def predict_student(student: dict):
    """
    Pass student details as dictionary.
    Returns predicted grade and grade category.
    """
    reg_model = joblib.load('models/grade_predictor.pkl')
    clf_model = joblib.load('models/category_classifier.pkl')

    d = student.copy()

    # ── Encode same as preprocessing ──────────────────────
    d['gender']           = 1 if d['gender'] == 'Female' else 0
    edu_map               = {'None': 0, 'High School': 1,
                             'Bachelor': 2, 'Master': 3}
    stress_map            = {'Low': 0, 'Medium': 1, 'High': 2}
    d['parent_education'] = edu_map[d['parent_education']]
    d['stress_level']     = stress_map[d['stress_level']]

    # ── Engineer same features ─────────────────────────────
    d['study_efficiency'] = d['study_hours'] * (d['attendance'] / 100)
    d['engagement_score'] = (d['assignments_done'] * 2 +
                             d['library_visits'] * 1.5 +
                             d['extra_classes'] * 5)
    d['good_sleep']       = 1 if 7 <= d['sleep_hours'] <= 9 else 0
    d['grade_gap']        = 100 - d['prev_grade']

    df             = pd.DataFrame([d])
    predicted_grade    = reg_model.predict(df)[0]
    predicted_category = clf_model.predict(df)[0]
    grade_label        = GRADE_MAP[predicted_category]

    print(f"\n🎓 Student Performance Prediction")
    print(f"   Predicted Final Grade : {predicted_grade:.1f} / 100")
    print(f"   Predicted Grade Category: {grade_label}")
    print(f"   Status: {'✅ PASS' if predicted_grade >= 40 else '❌ FAIL'}")
    return predicted_grade, grade_label

if __name__ == '__main__':
    student = {
        'age'             : 18,
        'gender'          : 'Female',
        'study_hours'     : 6.5,
        'attendance'      : 85.0,
        'prev_grade'      : 72,
        'parent_education': 'Bachelor',
        'internet_access' : 1,
        'extra_classes'   : 1,
        'sleep_hours'     : 7.5,
        'stress_level'    : 'Medium',
        'assignments_done': 15,
        'library_visits'  : 8,
        'sports'          : 1,
        'part_time_job'   : 0,
    }
    predict_student(student)