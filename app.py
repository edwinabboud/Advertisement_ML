import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Ensure this is first
st.set_page_config(layout="wide", page_title="Sales Prediction", page_icon="📊")

st.title("🚀 AI-Powered Sales Prediction & Model Performance Analysis 📊")

# 📖 **Explain Feature Selection (Moved to Top)**
st.subheader("📖 Why Did We Remove Newspaper?")
st.write("""
- The correlation matrix (shown below) reveals that **newspaper** has a very weak correlation with sales.
- This means newspaper spending does not significantly impact sales prediction.
- Our best models **exclude newspaper** to improve accuracy.
""")
st.image("img/correlation_matrix.png", caption="Feature Correlation Heatmap")

# Load dataset
df = pd.read_csv("Advertising.csv")

# Load trained model
with open("Linear_Standardized_No_Newspaper.pkl", "rb") as f:
    model = pickle.load(f)

# Load evaluation metrics and reset index for correct selection
eval_df = pd.read_csv("evaluation_results.csv")

# Rename model column correctly and ensure names are preserved
eval_df.rename(columns={"Unnamed: 0": "Model"}, inplace=True)

# Remove Decision Tree and Random Forest from the results (if present)
models_to_remove = ["Decision Tree Regressor", "Random Forest Regressor"]
eval_df = eval_df[~eval_df["Model"].isin(models_to_remove)]  # Exclude unwanted models

# **Sort so that models with "Newspaper" appear at the top**
eval_df["Has Newspaper"] = eval_df["Model"].apply(lambda x: "Newspaper" in x)
eval_df = eval_df.sort_values(by="Has Newspaper", ascending=False).drop(columns=["Has Newspaper"])

# Create a numerical index for charts while keeping full names for the table
eval_df["Model Index"] = range(1, len(eval_df) + 1)  # 1,2,3.. instead of names

# Streamlit UI
st.title("📊 Sales Prediction & Model Evaluation")

st.write("""
### Explore how well our regression models perform!
- View key **evaluation metrics** (MAE, MSE, RMSE, R² Score).
- Make **sales predictions** based on input values.
- Analyze **model performance** with interactive visualizations.
""")

# 📌 **Split Layout for Better Display**
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Model Performance Summary")
    display_df = eval_df.drop(columns=["Model Index"], errors="ignore")  # Table keeps full model names
    st.dataframe(display_df)

with col2:
    st.subheader("🔍 Select a Model for Detailed Metrics")
    selected_model = st.selectbox("Choose a model to inspect:", display_df["Model"])

    # Retrieve model metrics
    model_metrics = display_df[display_df["Model"] == selected_model].drop(columns=["Model"], errors="ignore").T
    model_metrics = model_metrics.apply(pd.to_numeric, errors="coerce")  # Convert to numeric if needed

    st.write(f"### 🔹 **{selected_model}**")
    st.dataframe(model_metrics.style.format("{:.4f}"))  # Safely apply formatting

# --- CHART 1: Model Comparison (Smaller Size) ---
st.subheader("📊 Compare Models with Different Metrics")
metric_option = st.selectbox(
    "Choose a metric to visualize:",
    ["None", "MAE", "MSE", "RMSE", "Train R²", "Test R²"]
)
if metric_option != "None":
    fig, ax = plt.subplots(figsize=(4, 2.5))  # **Smaller Chart Size**
    sns.barplot(x="Model Index", y=metric_option, data=eval_df, palette="coolwarm", ax=ax, ci=None)
    
    ax.set_title(f"Model Comparison: {metric_option}", fontsize=10, pad=6)
    ax.set_ylabel(metric_option, fontsize=9)
    ax.set_xlabel("Model Index", fontsize=9)

    # Fixing the tick locations and labels
    ax.set_xticks(range(1, len(eval_df) + 1))  
    ax.set_xticklabels(range(1, len(eval_df) + 1), rotation=0, ha="center", fontsize=8)  # Keep upright
    
    # Remove extra padding
    plt.tight_layout()
    
    st.pyplot(fig)

# --- CHART 2: Train vs Test R² Score (Smaller Size) ---
st.subheader("📈 Train vs Test R² Score (Overfitting Detection)")
toggle_overfit_chart = st.selectbox(
    "Choose an option:",
    ["None", "Show Train vs Test R² Chart"]
)
if toggle_overfit_chart == "Show Train vs Test R² Chart":
    fig, ax = plt.subplots(figsize=(4, 2.5))  # **Smaller Chart Size**

    # Create a bar plot for Train vs Test R²
    display_df_melted = eval_df.melt(id_vars=["Model Index"], value_vars=["Train R²", "Test R²"], var_name="Dataset", value_name="R² Score")
    sns.barplot(x="Model Index", y="R² Score", hue="Dataset", data=display_df_melted, palette="coolwarm", ax=ax, ci=None)
    
    ax.set_title("Train vs Test R² Score (Detecting Overfitting)", fontsize=10, pad=6)
    ax.set_ylabel("R² Score", fontsize=9)
    ax.set_xlabel("Model Index", fontsize=9)
    ax.set_ylim(0, 1.05)  # Slightly above 1 for clarity

    # Fix tick labels
    ax.set_xticks(range(1, len(eval_df) + 1))
    ax.set_xticklabels(range(1, len(eval_df) + 1), rotation=0, ha="center", fontsize=8)  # Keep upright

    # Remove extra padding
    plt.tight_layout()

    st.pyplot(fig)

# 📊 **Additional Visualizations**
st.subheader("📊 Additional Insights")
visualization_option = st.selectbox(
    "Choose an additional visualization:",
    ["None", "Predicted vs. Actual Sales Chart"]
)

if visualization_option == "Predicted vs. Actual Sales Chart":
    st.subheader("📉 Predicted vs. Actual Sales")
    st.image("img/pred_vs_actual.png", caption="Model Predictions vs. Actual Sales")

elif visualization_option == "Residual Distribution":
    st.subheader("📊 Residual Distribution")
    st.image("img/residuals.png", caption="Residual Errors in Prediction")

# 📌 **User Inputs for Prediction**
st.subheader("📊 Enter Feature Values to Predict Sales")
tv = st.number_input("TV Advertising Budget")
radio = st.number_input("Radio Advertising Budget")

# Prediction
if st.button("🔍 Predict Sales"):
    prediction = model.predict([[tv, radio]])  # No scaling needed
    st.write(f"📢 **Predicted Sales:** {prediction[0]:.4f}")

# 🎯 **Final Takeaways**
st.write("""
### 🎯 Key Takeaways
- **MAE, MSE, RMSE, and R² Score** help evaluate model accuracy.
- **Train vs Test R² Score** helps detect overfitting.
- **Feature scaling and feature selection impact model performance.**
""")
