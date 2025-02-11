import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os

# Ensure the 'img' and 'models' directories exist
img_folder = "img/"
models_folder = "models/"
os.makedirs(img_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

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

# Prepare datasets
X_no_news = df[['TV', 'radio']]  # Only TV and radio
X_with_news = df[['TV', 'radio', 'newspaper']]  # TV, radio, and newspaper
y = df['sales']

# Convert y into binary labels for Logistic Regression
y_median = y.median()
y_binary = (y > y_median).astype(int)  # 1 if sales > median, else 0

# Splitting the dataset
X_train_no_news, X_test_no_news, y_train, y_test = train_test_split(X_no_news, y, test_size=0.2, random_state=42)
X_train_with_news, X_test_with_news, _, _ = train_test_split(X_with_news, y, test_size=0.2, random_state=42)

# Splitting for Logistic Regression (binary classification)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_no_news, y_binary, test_size=0.2, random_state=42)

# Scaling
standard_scaler = StandardScaler()
X_train_std = standard_scaler.fit_transform(X_train_no_news)
X_test_std = standard_scaler.transform(X_test_no_news)

minmax_scaler = MinMaxScaler()
X_train_norm = minmax_scaler.fit_transform(X_train_no_news)
X_test_norm = minmax_scaler.transform(X_test_no_news)

# Model Training & Evaluation (Adding Logistic Regression)
models = {
    "Linear (No Scaling, No Newspaper)": (LinearRegression(), X_train_no_news, X_test_no_news),
    "Linear (Standardized, No Newspaper)": (LinearRegression(), X_train_std, X_test_std),
    "Linear (Normalized, No Newspaper)": (LinearRegression(), X_train_norm, X_test_norm),
    "Linear (No Scaling, With Newspaper)": (LinearRegression(), X_train_with_news, X_test_with_news),
    "Ridge Regression": (Ridge(), X_train_no_news, X_test_no_news),
    "Lasso Regression": (Lasso(), X_train_no_news, X_test_no_news),
    "Decision Tree Regressor": (DecisionTreeRegressor(), X_train_no_news, X_test_no_news),
    "Random Forest Regressor": (RandomForestRegressor(n_estimators=100), X_train_no_news, X_test_no_news),
    "Logistic Regression": (LogisticRegression(), X_train_log, X_test_log)  # Added logistic regression
}

results = {}

for name, (model, X_train, X_test) in models.items():
    model.fit(X_train, y_train if "Logistic" not in name else y_train_log)  # Use binary y for Logistic Regression
    y_pred = model.predict(X_test)

    # Compute Train and Test Scores
    train_r2 = model.score(X_train, y_train if "Logistic" not in name else y_train_log)  # Training R² or accuracy
    test_r2 = model.score(X_test, y_test if "Logistic" not in name else y_test_log)  # Test R² or accuracy

    mae = mean_absolute_error(y_test if "Logistic" not in name else y_test_log, y_pred)
    mse = mean_squared_error(y_test if "Logistic" not in name else y_test_log, y_pred)
    rmse = np.sqrt(mse)

    
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Overfitting Risk": "High" if (train_r2 - test_r2) > 0.1 else "Low"
    }

    # Save trained models
    model_filename = f"{models_folder}{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

# Save Scalers
with open(models_folder + "standard_scaler.pkl", "wb") as f:
    pickle.dump(standard_scaler, f)
with open(models_folder + "minmax_scaler.pkl", "wb") as f:
    pickle.dump(minmax_scaler, f)

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results).T

# 📊 **Save Performance Chart**
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df[["Train R²", "Test R²"]].dropna(), palette="coolwarm")  # Exclude NaN values (Logistic Regression)
plt.title("Train vs Test R² Score (Overfitting Detection)")
plt.ylabel("R² Score")
plt.xticks(rotation=45, ha='right')
plt.savefig(img_folder + "train_vs_test_r2.png")
plt.show()

# Save evaluation results to CSV
results_df.to_csv("evaluation_results.csv", index=True)

# Display results
print("\n📂 Model Evaluation Results:")
print(results_df)
print("\n✅ Evaluation results saved to 'evaluation_results.csv'.")
