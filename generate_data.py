# generate_data.py
# Run this ONCE to generate your student performance dataset

import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 2000

# ── Student Features ───────────────────────────────────────
age                = np.random.randint(15, 22, n)
gender             = np.random.choice(['Male', 'Female'], n)
study_hours        = np.random.uniform(0, 10, n).round(1)
attendance         = np.random.uniform(40, 100, n).round(1)
prev_grade         = np.random.randint(40, 100, n)
parent_education   = np.random.choice(['None', 'High School', 'Bachelor', 'Master'], n)
internet_access    = np.random.choice([0, 1], n, p=[0.2, 0.8])
extra_classes      = np.random.choice([0, 1], n, p=[0.5, 0.5])
sleep_hours        = np.random.uniform(4, 10, n).round(1)
stress_level       = np.random.choice(['Low', 'Medium', 'High'], n)
assignments_done   = np.random.randint(0, 20, n)
library_visits     = np.random.randint(0, 15, n)
sports             = np.random.choice([0, 1], n, p=[0.6, 0.4])
part_time_job      = np.random.choice([0, 1], n, p=[0.7, 0.3])

# ── Generate realistic final grade ─────────────────────────
final_grade = []
for i in range(n):
    score = 0

    score += study_hours[i] * 3.5
    score += (attendance[i] - 40) * 0.4
    score += prev_grade[i] * 0.3
    score += assignments_done[i] * 1.2
    score += library_visits[i] * 0.8

    if internet_access[i] == 1:  score += 5
    if extra_classes[i] == 1:    score += 8
    if sports[i] == 1:           score += 3
    if part_time_job[i] == 1:    score -= 5

    if stress_level[i] == 'High':   score -= 8
    elif stress_level[i] == 'Low':  score += 4

    if sleep_hours[i] < 6:       score -= 5
    elif sleep_hours[i] >= 8:    score += 3

    if parent_education[i] == 'Master':      score += 6
    elif parent_education[i] == 'Bachelor':  score += 4
    elif parent_education[i] == 'High School': score += 2

    score += np.random.normal(0, 5)
    final_grade.append(np.clip(score, 0, 100))

final_grade = np.array(final_grade).round(1)

# ── Pass/Fail based on grade ───────────────────────────────
pass_fail = (final_grade >= 40).astype(int)

# ── Grade Category ─────────────────────────────────────────
def grade_category(g):
    if g >= 80:   return 'A'
    elif g >= 65: return 'B'
    elif g >= 50: return 'C'
    elif g >= 40: return 'D'
    else:         return 'F'

grade_cat = [grade_category(g) for g in final_grade]

df = pd.DataFrame({
    'age'              : age,
    'gender'           : gender,
    'study_hours'      : study_hours,
    'attendance'       : attendance,
    'prev_grade'       : prev_grade,
    'parent_education' : parent_education,
    'internet_access'  : internet_access,
    'extra_classes'    : extra_classes,
    'sleep_hours'      : sleep_hours,
    'stress_level'     : stress_level,
    'assignments_done' : assignments_done,
    'library_visits'   : library_visits,
    'sports'           : sports,
    'part_time_job'    : part_time_job,
    'final_grade'      : final_grade,   # regression target
    'pass_fail'        : pass_fail,     # classification target
    'grade_category'   : grade_cat      # multiclass target
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)
print(f"Dataset created with {len(df)} rows!")
print(f"\nGrade Distribution:\n{pd.Series(grade_cat).value_counts()}")
print(f"\nPass Rate: {pass_fail.mean()*100:.1f}%")
print(df.head())
