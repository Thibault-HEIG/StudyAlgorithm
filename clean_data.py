import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Step 1 — Load the Source of Truth
print("Step 1: Loading student_habits_performance.csv...")
df = pd.read_csv("data/Kaggle/student_habits_performance.csv")
df.drop(columns=["student_id"], inplace=True)

# Step 2 — Impute Nulls
print("Step 2: Imputing nulls in parental_education_level...")
mode_edu = df["parental_education_level"].mode()[0]
df["parental_education_level"] = df["parental_education_level"].fillna(mode_edu)

# Step 3 — Encode Categorical Columns
print("Step 3: Encoding categorical columns...")
# 3a. Binary
binary_map = {"Yes": 1, "No": 0}
df["part_time_job"] = df["part_time_job"].map(binary_map)
df["extracurricular_participation"] = df["extracurricular_participation"].map(binary_map)

# 3b. Ordinal
df["diet_quality"] = df["diet_quality"].map({"Poor": 0, "Fair": 1, "Good": 2})
df["internet_quality"] = df["internet_quality"].map({"Poor": 0, "Average": 1, "Good": 2})
df["parental_education_level"] = df["parental_education_level"].map({
    "High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3
})

# 3c. Nominal (One-hot gender)
df = pd.get_dummies(df, columns=["gender"], drop_first=False, dtype=int)

# Step 4 — Clip Impossible Values
print("Step 4: Clipping impossible values...")
clips = {
    "study_hours_per_day":  (0, 16),
    "sleep_hours":          (3, 12),
    "attendance_percentage":(0, 100),
    "social_media_hours":   (0, 16),
    "netflix_hours":        (0, 16),
    "exercise_frequency":   (0, 7),
    "mental_health_rating": (1, 10),
    "exam_score":           (0, 100),
}
for col, (lo, hi) in clips.items():
    if col in df.columns:
        df[col] = df[col].clip(lo, hi)

# Step 5 — Format Continuous Features
print("Step 5: Formatting continuous features...")
df["attendance_percentage"] = df["attendance_percentage"] / 100.0
# Age and hours are kept in natural units as requested.
# We no longer use a global MinMaxScaler for these columns.

df_main = df.copy()

# Step 6 — Integrate StudentPerformanceFactors.csv
print("Step 6: Integrating StudentPerformanceFactors.csv...")
spf = pd.read_csv("data/Kaggle/StudentPerformanceFactors.csv")
spf["Exam_Score"] = spf["Exam_Score"].clip(0, 100)
spf["Parental_Education_Level"] = spf["Parental_Education_Level"].fillna(spf["Parental_Education_Level"].mode()[0])

spf_map = {
    "Exam_Score":                  "exam_score",
    "Sleep_Hours":                 "sleep_hours",
    "Attendance":                  "attendance_percentage",
    "Physical_Activity":           "exercise_frequency",
    "Tutoring_Sessions":           "tutoring_sessions",
    "Extracurricular_Activities":  "extracurricular_participation",
}
spf.rename(columns=spf_map, inplace=True)
spf["study_hours_per_day"] = spf["Hours_Studied"] / 7
spf["parental_education_level"] = spf["Parental_Education_Level"].map({
    "High School": 0, "College": 1, "Postgraduate": 3
})
spf["extracurricular_participation"] = spf["extracurricular_participation"].map({"Yes": 1, "No": 0})
spf["peer_influence"] = spf["Peer_Influence"].map({"Negative": 0, "Neutral": 1, "Positive": 2})
spf["learning_disabilities"] = spf["Learning_Disabilities"].map({"Yes": 1, "No": 0})

spf = pd.get_dummies(spf, columns=["Gender"], drop_first=False, dtype=int)
spf.rename(columns={"Gender_Male": "gender_Male", "Gender_Female": "gender_Female"}, inplace=True)
if "gender_Other" not in spf.columns:
    spf["gender_Other"] = 0

DROP_SPF = [
    "Hours_Studied", "Parental_Involvement", "Access_to_Resources",
    "Motivation_Level", "Internet_Access", "Tutoring_Sessions_raw",
    "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
    "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home",
    "Previous_Scores"
]
spf.drop(columns=[c for c in DROP_SPF if c in spf.columns], inplace=True)

# Clip continuous columns to the same ranges as main before scaling
for col, (lo, hi) in clips.items():
    if col in spf.columns:
        spf[col] = spf[col].clip(lo, hi)

# Add missing columns from main schema with NaN, then fill
for col in df_main.columns:
    if col not in spf.columns:
        spf[col] = np.nan

# Manual scaling for attendance
spf["attendance_percentage"] = spf["attendance_percentage"] / 100.0

df_spf = spf[df_main.columns.tolist() + ["tutoring_sessions", "peer_influence", "learning_disabilities"]]

# Step 7 — Integrate student_performance_interactions.csv
print("Step 7: Integrating student_performance_interactions.csv...")
spi = pd.read_csv("data/Kaggle/student_performance_interactions.csv")
spi["homework_completion_rate"] = spi["homework_completion_rate"].clip(0, 100)

spi_map = {
    "final_score":          "exam_score",
    "daily_study_hours":    "study_hours_per_day",
    "sleep_hours":          "sleep_hours",
    "attendance_percentage":"attendance_percentage",
    "motivation_score":     "motivation_score",
    "exam_anxiety_score":   "exam_anxiety_score",
    "parent_education_level":"parental_education_level",
}
spi.rename(columns=spi_map, inplace=True)
spi["parental_education_level"] = spi["parental_education_level"].map({
    "High School": 0, "Bachelor": 1, "Master": 2
})
spi["motivation_score"] = spi["motivation_score"] / 10
spi["exam_anxiety_score"] = spi["exam_anxiety_score"] / 10

DROP_SPI = [
    "student_id", "grade", "pass_fail",
    "previous_score", "math_prev_score", "science_prev_score", "language_prev_score",
    "screen_time_hours", "physical_activity_minutes", "homework_completion_rate",
    "study_environment"
]
spi.drop(columns=[c for c in DROP_SPI if c in spi.columns], inplace=True)

# Clip continuous columns
for col, (lo, hi) in clips.items():
    if col in spi.columns:
        spi[col] = spi[col].clip(lo, hi)

# Manual scaling for attendance
spi["attendance_percentage"] = spi["attendance_percentage"] / 100.0

# Align SPI with the full schema
spi_aligned = spi.copy()
for col in df_main.columns:
    if col not in spi_aligned.columns:
        spi_aligned[col] = np.nan
for col in ["tutoring_sessions", "peer_influence", "learning_disabilities"]:
    if col not in spi_aligned.columns:
        spi_aligned[col] = np.nan

df_spi_aligned = spi_aligned[df_spf.columns.tolist() + ["motivation_score", "exam_anxiety_score"]]

# Step 8 — Excluded Datasets (Handled by not including them)

# Step 9 — Concatenate, Fill, and Validate
print("Step 9: Concatenating and validating...")
final_columns = list(set(df_main.columns) | set(df_spf.columns) | set(df_spi_aligned.columns))

# Use copies to avoid modifying originals during column addition
df_main_final = df_main.copy()
df_spf_final = df_spf.copy()
df_spi_final = df_spi_aligned.copy()

for df_tmp in [df_main_final, df_spf_final, df_spi_final]:
    for col in final_columns:
        if col not in df_tmp.columns:
            df_tmp[col] = np.nan

df_final = pd.concat([df_main_final, df_spf_final, df_spi_final], ignore_index=True)

# Fill ALL missing values with median
for col in df_final.columns:
    if col != "exam_score":
        df_final[col] = df_final[col].fillna(df_final[col].median())

# Ensure age is integer
df_final["age"] = df_final["age"].astype(int)

# Scale tutoring and peer influence (they were added after initial scaling)
# peer_influence was ordinal 0-2 -> map to [0, 1]
df_final["peer_influence"] = df_final["peer_influence"] / 2
# tutoring_sessions range 0-8 -> map to [0, 1]
df_final["tutoring_sessions"] = df_final["tutoring_sessions"] / 8

df_final.drop_duplicates(inplace=True)

# Validation
assert df_final.isnull().sum().sum() == 0, \
    f"Nulls remain:\n{df_final.isnull().sum()[df_final.isnull().sum() > 0]}"

assert df_final["exam_score"].between(0, 100).all(), \
    "exam_score contains values outside [0, 100]"

assert df_final["attendance_percentage"].between(0, 1.000001).all(), \
    "attendance_percentage contains values outside [0, 1]"

assert df_final["age"].dtype == int or df_final["age"].dtype == np.int64, \
    f"age is not integer, got {df_final['age'].dtype}"

print(f"✓ Final dataset: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
print("Sample of formatted features:")
print(df_final[["age", "study_hours_per_day", "attendance_percentage"]].head())

# Step 10 — Save
print("Step 10: Saving to data/cleaned_data.csv...")
os.makedirs("data", exist_ok=True)
df_final.to_csv("data/cleaned_data.csv", index=False)
