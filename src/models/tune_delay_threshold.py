
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/processed/modeling_dataset.csv"
MODEL_DIR = BASE_DIR / "src/models/artifacts"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("DELAY THRESHOLD OPTIMIZATION")
print("==================================\n")


# ==========================================
# 2. LOAD DATA
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
y = df["Is_Delayed"].astype(int)


# ==========================================
# 3. CHRONOLOGICAL TRAIN / VALIDATION / TEST
# ==========================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))

print("\nValidation period:")
print(
    df["Order_Date"].iloc[train_end],
    "to",
    df["Order_Date"].iloc[validation_end - 1]
)

print("\nTesting period:")
print(
    df["Order_Date"].iloc[validation_end],
    "to",
    df["Order_Date"].iloc[-1]
)


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 5. TRAIN CLASSIFIER
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

model.fit(X_train, y_train)

print("\nClassifier training completed.")


# ==========================================
# 6. SELECT THRESHOLD ON VALIDATION DATA
# ==========================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

precision, recall, thresholds = (
    precision_recall_curve(
        y_validation,
        validation_probabilities
    )
)

# precision and recall contain one extra
# endpoint compared with thresholds.

threshold_results = []

for i, threshold in enumerate(thresholds):

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision[i],
        "Recall": recall[i],
        "F1": f1_score(
            y_validation,
            validation_probabilities >= threshold,
            zero_division=0
        ),
        "F2": fbeta_score(
            y_validation,
            validation_probabilities >= threshold,
            beta=2,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORT_DIR / "threshold_validation_results.csv",
    index=False
)

# Business objective:
# First try to achieve recall >= 75%.
# Among qualifying thresholds, choose the one
# with the highest precision.

eligible = threshold_df[
    threshold_df["Recall"] >= 0.75
]

if not eligible.empty:

    best_row = eligible.sort_values(
        ["Precision", "F2"],
        ascending=False
    ).iloc[0]

    selection_reason = (
        "Highest precision among thresholds "
        "achieving at least 75% recall."
    )

else:

    # If 75% recall cannot be achieved,
    # select the threshold with the highest F2.
    best_row = threshold_df.loc[
        threshold_df["F2"].idxmax()
    ]

    selection_reason = (
        "75% recall was not achievable. "
        "Selected the threshold with the highest F2."
    )

best_threshold = float(best_row["Threshold"])

print("\n========== SELECTED THRESHOLD ==========")
print("Threshold:", round(best_threshold, 4))
print("Validation precision:", round(best_row["Precision"], 4))
print("Validation recall:", round(best_row["Recall"], 4))
print("Validation F1:", round(best_row["F1"], 4))
print("Selection reason:", selection_reason)


# ==========================================
# 7. FINAL EVALUATION ON UNSEEN TEST DATA
# ==========================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = {
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(
        y_test, test_predictions
    ),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "F2": fbeta_score(
        y_test, test_predictions, beta=2,
        zero_division=0
    ),
    "ROC_AUC": roc_auc_score(
        y_test, test_probabilities
    )
}

print("\n========== FINAL TEST RESULTS ==========")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test, test_predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Not Delayed", "Delayed"],
        zero_division=0
    )
)


# ==========================================
# 8. SAVE RESULTS AND MODEL
# ==========================================

pd.DataFrame([test_metrics]).to_csv(
    REPORT_DIR / "optimized_delay_model_metrics.csv",
    index=False
)

test_output = pd.DataFrame({
    "Order_Id": df["Order_Id"].iloc[validation_end:].values,
    "Actual_Is_Delayed": y_test.values,
    "Predicted_Is_Delayed": test_predictions,
    "Delay_Probability": np.round(
        test_probabilities, 4
    )
})

test_output.to_csv(
    REPORT_DIR / "optimized_delay_test_predictions.csv",
    index=False
)

with open(
    REPORT_DIR / "selected_delay_threshold.txt",
    "w"
) as file:
    file.write(
        f"Selected threshold: {best_threshold:.6f}\n"
        f"Selection reason: {selection_reason}\n"
    )

joblib.dump(
    model,
    MODEL_DIR / "optimized_delay_classification_model.joblib"
)

print("\nGenerated files:")
print("1. reports/threshold_validation_results.csv")
print("2. reports/optimized_delay_model_metrics.csv")
print("3. reports/optimized_delay_test_predictions.csv")
print("4. reports/selected_delay_threshold.txt")
print("5. src/models/artifacts/optimized_delay_classification_model.joblib")

print("\nThreshold optimization completed!")
