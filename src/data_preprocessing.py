# src/data_preprocessing.py
# Loads, encodes, and splits student performance dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    """
    Load student dataset, engineer features, encode categoricals,
    scale numerics, and return train/test splits for both
    regression (final_grade) and classification (grade_category).
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")
    print(f"\nGrade Distribution:\n{df['grade_category'].value_counts()}")
    print(f"Pass Rate: {df['pass_fail'].mean()*100:.1f}%")

    # ── Encode Categoricals ────────────────────────────────
    df['gender'] = (df['gender'] == 'Female').astype(int)

    edu_map = {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3}
    df['parent_education'] = df['parent_education'].map(edu_map)

    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['stress_level'] = df['stress_level'].map(stress_map)

    grade_map = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
    df['grade_encoded'] = df['grade_category'].map(grade_map)

    # ── Feature Engineering ────────────────────────────────
    df['study_efficiency'] = df['study_hours'] * (df['attendance'] / 100)
    df['engagement_score'] = (
        df['assignments_done'] * 2 +
        df['library_visits'] * 1.5 +
        df['extra_classes'] * 5
    )
    df['good_sleep'] = ((df['sleep_hours'] >= 7) &
                        (df['sleep_hours'] <= 9)).astype(int)
    df['grade_gap'] = 100 - df['prev_grade']

    # ── Drop NaN rows ──────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Features and Targets ───────────────────────────────
    X = df.drop(columns=['final_grade', 'pass_fail',
                          'grade_category', 'grade_encoded'])
    y_grade    = df['final_grade']
    y_category = df['grade_encoded']

    # ── Train/Test Split ───────────────────────────────────
    X_train, X_test, y_train_g, y_test_g = train_test_split(
        X, y_grade, test_size=0.2, random_state=42
    )
    _, _, y_train_c, y_test_c = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )

    # ── Scale Features ─────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features: {X.shape[1]} columns")
    return (X_train_scaled, X_test_scaled,
            y_train_g, y_test_g,
            y_train_c, y_test_c)