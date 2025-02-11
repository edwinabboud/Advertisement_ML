import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# Ensure the 'img' directory exists
img_folder = "img/"
os.makedirs(img_folder, exist_ok=True)

# Load dataset
df = pd.read_csv("Advertising.csv")

# Display correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Save correlation matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig(img_folder + "correlation_matrix.png")
plt.show()

# Train model WITHOUT newspaper
X = df[['TV', 'radio']]  # Only using TV and radio
y = df['sales']

# Split dataset (without scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation Results:")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

# Save the trained model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")

# 📂 **Save Evaluation Results to CSV**
evaluation_results = {
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R² Score": [r2]
}

eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv("evaluation_results.csv", index=False)
print("\n📂 Evaluation results saved to 'evaluation_results.csv'.")

# 📊 **New Graphs for Better Display** 📊

# 1️⃣ **Predicted vs Actual Line Chart**
plt.figure(figsize=(8, 5))
sorted_indices = np.argsort(y_test)
plt.plot(np.array(y_test)[sorted_indices], label="Actual Sales", marker="o", linestyle="-")
plt.plot(np.array(y_pred)[sorted_indices], label="Predicted Sales", marker="s", linestyle="--")
plt.xlabel("Data Points (Sorted by Actual Sales)")
plt.ylabel("Sales")
plt.title("Predicted vs. Actual Sales (Without Scaling)")
plt.legend()
plt.savefig(img_folder + "pred_vs_actual.png")
plt.show()

# 2️⃣ **Residual Plot**
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution (Without Scaling)")
plt.savefig(img_folder + "residuals.png")
plt.show()

# 3️⃣ **Model Performance Bar Chart**
performance_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(performance_metrics.keys()), y=list(performance_metrics.values()))
plt.title("Model Performance Metrics (Without Scaling)")
plt.ylabel("Error Value")
plt.savefig(img_folder + "model_performance.png")
plt.show()
