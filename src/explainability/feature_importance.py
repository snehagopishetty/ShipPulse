
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.inspection import permutation_importance

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/processed/modeling_dataset.csv"
MODEL_DIR = BASE_DIR / "src/models/artifacts"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("MODEL EXPLAINABILITY")
print("==================================\n")


# ==========================================
# 2. LOAD DATA AND MODELS
# ==========================================

df = pd.read_csv(DATA_PATH)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df = (
    df.dropna(subset=["Order_Date"])
    .sort_values("Order_Date")
    .reset_index(drop=True)
)

eta_model = joblib.load(
    MODEL_DIR / "eta_regression_model.joblib"
)

delay_model = joblib.load(
    MODEL_DIR / "optimized_delay_classification_model.joblib"
)

categorical_features = [
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Order_Country",
    "Customer_Segment"
]

numerical_features = [
    "Scheduled_Days",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek",
    "Order_Day",
    "Is_Weekend",
    "Total_Quantity",
    "Total_Order_Value",
    "Number_Of_Items"
]

feature_columns = (
    categorical_features + numerical_features
)

X = df[feature_columns]


# ==========================================
# 3. CREATE THE ORIGINAL TEST WINDOWS
# ==========================================

# ETA model was trained using the first 80%.

eta_split = int(len(df) * 0.80)

X_eta_test = X.iloc[eta_split:]
y_eta_test = df[
    "Actual_Delivery_Days"
].iloc[eta_split:]


# Delay model was trained using the first 70%
# and threshold validation used the next 15%.

delay_split = int(len(df) * 0.85)

X_delay_test = X.iloc[delay_split:]
y_delay_test = df[
    "Is_Delayed"
].iloc[delay_split:].astype(int)


# ==========================================
# 4. ETA REGRESSION IMPORTANCE
# ==========================================

print("\nCalculating ETA feature importance...")

eta_importance = permutation_importance(
    eta_model,
    X_eta_test,
    y_eta_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

eta_results = pd.DataFrame({
    "Feature": feature_columns,
    "Importance_Mean": eta_importance.importances_mean,
    "Importance_Std": eta_importance.importances_std
})

eta_results = eta_results.sort_values(
    "Importance_Mean",
    ascending=False
)

eta_results.to_csv(
    REPORT_DIR / "eta_feature_importance.csv",
    index=False
)

print("\nTop ETA prediction factors:")
print(eta_results.head(10).to_string(index=False))


# ==========================================
# 5. DELAY CLASSIFICATION IMPORTANCE
# ==========================================

print("\nCalculating delay feature importance...")

delay_importance = permutation_importance(
    delay_model,
    X_delay_test,
    y_delay_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

delay_results = pd.DataFrame({
    "Feature": feature_columns,
    "Importance_Mean": delay_importance.importances_mean,
    "Importance_Std": delay_importance.importances_std
})

delay_results = delay_results.sort_values(
    "Importance_Mean",
    ascending=False
)

delay_results.to_csv(
    REPORT_DIR / "delay_feature_importance.csv",
    index=False
)

print("\nTop delay prediction factors:")
print(delay_results.head(10).to_string(index=False))


# ==========================================
# 6. VISUALIZE ETA IMPORTANCE
# ==========================================

eta_plot = eta_results.head(12).sort_values(
    "Importance_Mean"
)

plt.figure(figsize=(10, 6))

plt.barh(
    eta_plot["Feature"],
    eta_plot["Importance_Mean"],
    xerr=eta_plot["Importance_Std"]
)

plt.xlabel(
    "Increase in MAE when feature is shuffled (days)"
)

plt.title("ETA Prediction - Permutation Importance")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_eta_feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. VISUALIZE DELAY IMPORTANCE
# ==========================================

delay_plot = delay_results.head(12).sort_values(
    "Importance_Mean"
)

plt.figure(figsize=(10, 6))

plt.barh(
    delay_plot["Feature"],
    delay_plot["Importance_Mean"],
    xerr=delay_plot["Importance_Std"]
)

plt.xlabel(
    "Decrease in ROC-AUC when feature is shuffled"
)

plt.title("Delay Classification - Permutation Importance")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_delay_feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n==================================")
print("EXPLAINABILITY COMPLETED")
print("==================================")

print("\nGenerated files:")
print("1. reports/eta_feature_importance.csv")
print("2. reports/delay_feature_importance.csv")
print("3. reports/figures/05_eta_feature_importance.png")
print("4. reports/figures/06_delay_feature_importance.png")

print("\nShipPulse AI explainability completed!")
