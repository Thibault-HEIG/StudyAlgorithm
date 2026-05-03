# Skill: cleandata — Student Performance Data Cleaning & Normalization

## Trigger
Activate when asked to: clean the data, normalize the datasets, prepare data for training, run the data pipeline, or build `cleaned_data.csv`.

## Purpose
Produce a single `data/cleaned_data.csv` ready for ML training. Source of truth: `student_habits_performance.csv` (1,000 rows). Two supplementary datasets are selectively appended after schema alignment. Five datasets are explicitly excluded. See §8.

## Output Contract
- File: `data/cleaned_data.csv`
- Target column: `exam_score` (float, 0–100) — **never scaled**
- All continuous features: Min-Max normalized to [0, 1]
- All ordinal features: integer-encoded (0–N)
- All binary features: 0 or 1
- Zero nulls, zero impossible values, no ID columns

---

## Step 1 — Load the Source of Truth

```python
import pandas as pd

df = pd.read_csv("data/student_habits_performance.csv")
df.drop(columns=["student_id"], inplace=True)
```

`student_id` is an identifier, not a feature. Drop it immediately.

---

## Step 2 — Impute Nulls

Only `parental_education_level` has nulls (91/1000).

**Why mode, not mean:** This column is categorical. Mean is undefined for strings. Mode preserves the most frequent real value without inventing a new category.

```python
mode_edu = df["parental_education_level"].mode()[0]
df["parental_education_level"].fillna(mode_edu, inplace=True)
```

---

## Step 3 — Encode Categorical Columns

### 3a. Binary (Yes/No → 0/1)

**Why not one-hot:** Two-value columns have no information gain from a second dummy column. One integer is sufficient and avoids multicollinearity.

```python
binary_map = {"Yes": 1, "No": 0}
df["part_time_job"] = df["part_time_job"].map(binary_map)
df["extracurricular_participation"] = df["extracurricular_participation"].map(binary_map)
```

### 3b. Ordinal (ordered categories → integers)

**Why not one-hot:** These have a real ordering (Poor < Fair < Good). One-hot would discard that ordering and inflate dimensionality.

```python
df["diet_quality"] = df["diet_quality"].map({"Poor": 0, "Fair": 1, "Good": 2})
df["internet_quality"] = df["internet_quality"].map({"Poor": 0, "Average": 1, "Good": 2})
df["parental_education_level"] = df["parental_education_level"].map({
    "High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3
})
```

### 3c. Nominal (no order → one-hot)

**Why one-hot:** `gender` has no natural ranking between Female/Male/Other. Assigning arbitrary integers (0/1/2) would imply Male > Female, which is meaningless and would corrupt model weights.

```python
df = pd.get_dummies(df, columns=["gender"], drop_first=False)
# Result: gender_Female, gender_Male, gender_Other (int 0/1)
```

---

## Step 4 — Clip Impossible Values

Before scaling, enforce physically valid ranges. Clipping, not dropping, preserves row count.

```python
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
    df[col] = df[col].clip(lo, hi)
```

---

## Step 5 — Min-Max Scale Continuous Features

**Why Min-Max and not StandardScaler:**
StandardScaler assumes a Gaussian distribution (centers at mean=0, scales by standard deviation). These lifestyle variables (sleep, social media, study hours) are not Gaussian — they are bounded and likely skewed. Min-Max scaling preserves the original distribution shape and keeps all values in [0, 1] regardless of distribution. Do NOT scale ordinal columns (already 0–N, implicitly normalized). Do NOT scale the target `exam_score`.

```python
from sklearn.preprocessing import MinMaxScaler
import joblib

scale_cols = [
    "study_hours_per_day", "social_media_hours", "netflix_hours",
    "attendance_percentage", "sleep_hours", "exercise_frequency",
    "mental_health_rating", "age"
]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Save scaler — required for inference. Never refit on new data.
# Refitting on inference data causes data leakage (the scaler learns the test distribution).
joblib.dump(scaler, "models/scaler.pkl")
```

At this point `df` is the clean, normalized main dataset. Save a checkpoint:

```python
df_main = df.copy()
```

---

## Step 6 — Integrate `StudentPerformanceFactors.csv`

**Why this dataset:** 6,607 rows. Same target scale (Exam_Score 0–100). Provides three features absent from the main dataset: `tutoring_sessions`, `peer_influence`, `learning_disabilities`. Large enough to meaningfully shift model performance.

### 6a. Load and filter

```python
spf = pd.read_csv("data/StudentPerformanceFactors.csv")

# Exam_Score = 101 exists in 1 row — clip, do not drop (preserve row)
spf["Exam_Score"] = spf["Exam_Score"].clip(0, 100)
```

### 6b. Handle SPF nulls before mapping

```python
# Parental_Education_Level: 90 nulls
spf["Parental_Education_Level"].fillna(
    spf["Parental_Education_Level"].mode()[0], inplace=True
)
# Teacher_Quality: 78 nulls — column will be dropped, no action needed
# Distance_from_Home: 67 nulls — column will be dropped, no action needed
```

### 6c. Map to unified schema

```python
spf_map = {
    "Exam_Score":                  "exam_score",
    "Sleep_Hours":                 "sleep_hours",
    "Attendance":                  "attendance_percentage",
    "Physical_Activity":           "exercise_frequency",
    "Tutoring_Sessions":           "tutoring_sessions",
    "Extracurricular_Activities":  "extracurricular_participation",
}
spf.rename(columns=spf_map, inplace=True)

# Hours_Studied is per week — convert to per day
spf["study_hours_per_day"] = spf["Hours_Studied"] / 7

# Ordinal encodings
spf["parental_education_level"] = spf["Parental_Education_Level"].map({
    "High School": 0, "College": 1, "Postgraduate": 3
})
spf["extracurricular_participation"] = spf["extracurricular_participation"].map(
    {"Yes": 1, "No": 0}
)
spf["peer_influence"] = spf["Peer_Influence"].map(
    {"Negative": 0, "Neutral": 1, "Positive": 2}
)
spf["learning_disabilities"] = spf["Learning_Disabilities"].map({"Yes": 1, "No": 0})

# One-hot gender
spf = pd.get_dummies(spf, columns=["Gender"], drop_first=False)
spf.rename(columns={
    "Gender_Male": "gender_Male",
    "Gender_Female": "gender_Female"
}, inplace=True)
if "gender_Other" not in spf.columns:
    spf["gender_Other"] = 0
```

### 6d. Drop columns with no main-schema equivalent

**Why drop instead of add:** Columns present only in SPF rows create structural NaN blocks in main dataset rows. A model that sees NaN-for-main-rows vs value-for-SPF-rows will learn the missingness pattern as a signal — this is hidden data leakage. Only retain SPF-exclusive columns if they will be imputed for all rows.

```python
DROP_SPF = [
    "Hours_Studied", "Parental_Involvement", "Access_to_Resources",
    "Motivation_Level", "Internet_Access", "Tutoring_Sessions_raw",
    "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence",
    "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home",
    "Previous_Scores", "Peer_Influence"
]
spf.drop(columns=[c for c in DROP_SPF if c in spf.columns], inplace=True)
```

### 6e. Apply same scaler to SPF continuous features

**Critical:** Use the already-fitted scaler from Step 5. Do not refit.

```python
# Add missing columns from main schema with NaN, then fill
for col in df_main.columns:
    if col not in spf.columns:
        spf[col] = float("nan")

# Scale same continuous cols using the saved scaler (no refit)
spf[scale_cols] = scaler.transform(spf[scale_cols])

df_spf = spf[df_main.columns.tolist() +
             ["tutoring_sessions", "peer_influence", "learning_disabilities"]]
```

---

## Step 7 — Integrate `student_performance_interactions.csv`

**Why this dataset:** 1,000 rows. Same target scale (final_score 0–100). Adds `motivation_score` and `exam_anxiety_score` — psychometric features absent from the main dataset.

**Exclusion warning on sub-features:** `grade`, `pass_fail` are derived from `final_score`. `previous_score` and subject scores (`math_prev_score`, etc.) partially encode the target variable (past grade predicts future grade). Including them makes the model learn score history → score, not behavior → score. Drop all of them.

### 7a. Load and clean

```python
spi = pd.read_csv("data/student_performance_interactions.csv")

# Some values exceed 100 due to synthetic generation error
spi["homework_completion_rate"] = spi["homework_completion_rate"].clip(0, 100)
```

### 7b. Map to unified schema

```python
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

# Normalize psychometric scores (range 0–10) to [0, 1]
spi["motivation_score"] = spi["motivation_score"] / 10
spi["exam_anxiety_score"] = spi["exam_anxiety_score"] / 10
```

### 7c. Drop leaky and unmappable columns

```python
DROP_SPI = [
    "student_id", "grade", "pass_fail",
    "previous_score", "math_prev_score", "science_prev_score", "language_prev_score",
    "screen_time_hours", "physical_activity_minutes", "homework_completion_rate",
    "study_environment"
]
spi.drop(columns=[c for c in DROP_SPI if c in spi.columns], inplace=True)

# Scale continuous cols using saved scaler (no refit)
for col in scale_cols:
    if col not in spi.columns:
        spi[col] = float("nan")
spi[scale_cols] = scaler.transform(spi[scale_cols])
```

---

## Step 8 — Excluded Datasets (Do Not Use)

| Dataset | Reason |
|---|---|
| `student_lifestyle_dataset` | Target is GPA 0–4.0. Different scale and grading system. Normalizing both to [0,1] makes distributions superficially comparable but distributions remain from different grading contexts — model learns mixed signals. |
| `student_lifestyle_performance_dataset` | Target is CGPA 0–10 (Indian university system). Same scale mismatch. |
| `student_performance_updated_1000` | Impossible values: `Attendance (%)` reaches 200, `Study Hours` reaches -5. High null rate (~4–5% per column). Indicates corrupt or poorly generated synthetic data. Not recoverable without original source. |
| `student_study_habits` | Already pre-normalized (all values 0–1 already). Reverse-engineering original scale is impossible. Importing unknown normalization choices silently corrupts the pipeline. |
| `hours_study` | 25 rows only. Cannot meaningfully affect model training. Adds noise to distribution estimates. |

---

## Step 9 — Concatenate, Fill, and Validate

### 9a. Add new feature columns to main dataset (fill with median)

`tutoring_sessions`, `peer_influence`, `learning_disabilities` (from SPF) and `motivation_score`, `exam_anxiety_score` (from SPI) are NaN in rows that don't originate from those datasets. Fill with median — not mean — because these distributions are unknown and likely non-Gaussian.

```python
df_final = pd.concat([df_main, df_spf, df_spi_aligned], ignore_index=True)

fill_cols = [
    "tutoring_sessions", "peer_influence", "learning_disabilities",
    "motivation_score", "exam_anxiety_score"
]
for col in fill_cols:
    if col in df_final.columns:
        df_final[col].fillna(df_final[col].median(), inplace=True)
```

### 9b. Scale new continuous features

```python
# peer_influence was ordinal 0–2, normalize to [0,1]
df_final["peer_influence"] = df_final["peer_influence"] / 2
# tutoring_sessions range 0–8, normalize to [0,1]
df_final["tutoring_sessions"] = df_final["tutoring_sessions"] / 8
# learning_disabilities already binary 0/1 — no scaling needed
```

### 9c. Deduplicate

```python
df_final.drop_duplicates(inplace=True)
```

### 9d. Validate — raise errors, do not silently proceed

```python
assert df_final.isnull().sum().sum() == 0, \
    f"Nulls remain:\n{df_final.isnull().sum()[df_final.isnull().sum() > 0]}"

assert df_final["exam_score"].between(0, 100).all(), \
    "exam_score contains values outside [0, 100]"

assert df_final[scale_cols].between(0, 1).all().all(), \
    "Scaled features contain values outside [0, 1]"

print(f"✓ Final dataset: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
```

### 9e. Save

```python
df_final.to_csv("data/cleaned_data.csv", index=False)
```

---

## Step 10 — Final Schema Reference

| Column | Type | Range | Source | Notes |
|---|---|---|---|---|
| `exam_score` | float | 0–100 | All | **Target — never scale** |
| `study_hours_per_day` | float | 0–1 | All | Min-Max scaled |
| `sleep_hours` | float | 0–1 | All | Min-Max scaled |
| `attendance_percentage` | float | 0–1 | All | Min-Max scaled |
| `social_media_hours` | float | 0–1 | Main | Min-Max scaled |
| `netflix_hours` | float | 0–1 | Main | Min-Max scaled |
| `exercise_frequency` | float | 0–1 | Main + SPF | Min-Max scaled |
| `mental_health_rating` | float | 0–1 | Main | Min-Max scaled |
| `age` | float | 0–1 | Main | Min-Max scaled |
| `diet_quality` | int | 0–2 | Main | Ordinal: Poor/Fair/Good |
| `internet_quality` | int | 0–2 | Main | Ordinal: Poor/Average/Good |
| `parental_education_level` | int | 0–3 | Main + SPF + SPI | Ordinal |
| `part_time_job` | int | 0–1 | Main | Binary |
| `extracurricular_participation` | int | 0–1 | Main + SPF | Binary |
| `gender_Female` | int | 0–1 | Main + SPF | One-hot |
| `gender_Male` | int | 0–1 | Main + SPF | One-hot |
| `gender_Other` | int | 0–1 | Main | One-hot |
| `tutoring_sessions` | float | 0–1 | SPF | Scaled /8; median-filled elsewhere |
| `peer_influence` | float | 0–1 | SPF | Ordinal /2; median-filled elsewhere |
| `learning_disabilities` | int | 0–1 | SPF | Binary; median-filled elsewhere |
| `motivation_score` | float | 0–1 | SPI | Scaled /10; median-filled elsewhere |
| `exam_anxiety_score` | float | 0–1 | SPI | Scaled /10; median-filled elsewhere |

**Expected final row count:** ~7,500–8,000 (after deduplication across ~8,600 combined rows)
