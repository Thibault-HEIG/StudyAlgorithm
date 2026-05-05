import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('data/cleaned_data.csv')
X = df[['sleep_hours']]
y = df['exam_score']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Output results
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")
