import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
df = pd.read_csv("New_York_Air_Quality.csv")

# Drop missing values
df = df.dropna(subset=['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI'])

# Features and target
X = df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]
y = df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print("R² Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

# ------------------- PLOTS -------------------

# 1. Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Scatter Plot: Actual vs Predicted AQI")
plt.grid(True)
plt.show()

# 2. Box Plot of Features
plt.figure(figsize=(8, 5))
sns.boxplot(data=X)
plt.title("Box Plot: Air Quality Features")
plt.show()

# 3. Heatmap of Correlation
plt.figure(figsize=(7, 5))
sns.heatmap(df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap: Feature Correlation")
plt.show()

# 4. Bar Graph of Feature Importance
plt.figure(figsize=(6, 4))
plt.bar(X.columns, model.coef_, color='green')
plt.title("Bar Graph: Feature Importance (Coefficients)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 5. Radar Plot (using Matplotlib)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
feature_means = np.mean(X_scaled, axis=0)

labels = list(X.columns)
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
feature_means = np.concatenate((feature_means, [feature_means[0]]))
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, feature_means, 'o-', linewidth=2)
ax.fill(angles, feature_means, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title("Radar Plot: Normalized Feature Means")
plt.grid(True)
plt.show()
