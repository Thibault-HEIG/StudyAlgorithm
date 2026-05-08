-- Student Performance Data Analysis

-- 1. Study Hours vs Exam Score
-- Shows the direct correlation between daily study time and academic results.
SELECT 
    ROUND(study_hours_per_day) AS rounded_study_hours, 
    ROUND(AVG(exam_score), 2) AS avg_score, 
    COUNT(*) as student_count 
FROM student_performance 
GROUP BY rounded_study_hours 
ORDER BY rounded_study_hours;

-- 2. Lifestyle: Screen Time Impact
-- Compares students with high screen time (>5hrs total social media + netflix) vs others.
SELECT 
    CASE 
        WHEN (social_media_hours + netflix_hours) > 5 THEN 'High Screen Time (>5hrs)' 
        ELSE 'Low Screen Time (<=5hrs)' 
    END AS lifestyle_category, 
    ROUND(AVG(exam_score), 2) AS avg_score, 
    COUNT(*) as student_count 
FROM student_performance 
GROUP BY lifestyle_category;

-- 3. Mental Health & Anxiety
-- Explores the relationship between mental wellbeing, exam anxiety, and performance.
SELECT 
    mental_health_rating, 
    ROUND(AVG(exam_score), 2) AS avg_score, 
    ROUND(AVG(exam_anxiety_score), 4) AS avg_anxiety 
FROM student_performance 
GROUP BY mental_health_rating 
ORDER BY mental_health_rating;

-- 4. Attendance & Part-time Jobs
-- Analyzes if having a part-time job affects attendance and final scores.
SELECT 
    part_time_job,
    ROUND(AVG(attendance_percentage) * 100, 2) AS avg_attendance_pct,
    ROUND(AVG(exam_score), 2) AS avg_score,
    COUNT(*) as student_count
FROM student_performance
GROUP BY part_time_job;

-- 5. Gender Performance Overview
-- Compares performance across gender categories.
SELECT 
    CASE 
        WHEN gender_Female = 1.0 THEN 'Female'
        WHEN gender_Male = 1.0 THEN 'Male'
        WHEN gender_Other = 1.0 THEN 'Other'
    END AS gender,
    ROUND(AVG(exam_score), 2) AS avg_score,
    COUNT(*) as student_count
FROM student_performance
GROUP BY gender;
