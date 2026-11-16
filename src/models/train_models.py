
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.dummy import (
    DummyRegressor,
    DummyClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/processed/modeling_dataset.csv"

MODEL_DIR = BASE_DIR / "src/models/artifacts"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("ETA AND DELAY MODEL TRAINING")
print("==================================\n")


# ==========================================
# 2. LOAD MODELLING DATASET
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Modelling dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df = df.dropna(subset=["Order_Date"]).copy()

print("Dataset loaded:", len(df), "orders")


# ==========================================
# 3. DEFINE FEATURES AND TARGETS
# ==========================================

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

regression_target = "Actual_Delivery_Days"
classification_target = "Is_Delayed"

required_columns = (
    feature_columns
    + [
        regression_target,
        classification_target,
        "Order_Id",
        "Order_Date"
    ]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[feature_columns].copy()

y_regression = df[regression_target].copy()

y_classification = df[
    classification_target
].astype(int).copy()

assert not X.isnull().any().any(), (
    "Missing feature values detected."
)


# ==========================================
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

# Sort by order placement date.
# Older orders are used for training.
# Newer orders are reserved for testing.

df = df.sort_values(
    "Order_Date"
).reset_index(drop=True)

X = df[feature_columns].copy()

y_regression = df[
    regression_target
].copy()

y_classification = df[
    classification_target
].astype(int).copy()

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_reg_train = y_regression.iloc[:split_index]
y_reg_test = y_regression.iloc[split_index:]

y_cls_train = y_classification.iloc[:split_index]
y_cls_test = y_classification.iloc[split_index:]

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training date range:",
    df["Order_Date"].iloc[0],
    "to",
    df["Order_Date"].iloc[split_index - 1]
)

print(
    "Testing date range:",
    df["Order_Date"].iloc[split_index],
    "to",
    df["Order_Date"].iloc[-1]
)

if y_cls_train.nunique() < 2:
    raise ValueError(
        "Training data must contain both delayed "
        "and non-delayed orders."
    )


# ==========================================
# 5. CREATE PREPROCESSING PIPELINE
# ==========================================

# Convert categorical columns into numerical
# representations using one-hot encoding.
#
# Numerical features are passed through unchanged.

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
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
# 6. REGRESSION MODEL - ETA PREDICTION
# ==========================================

print("\n========== ETA REGRESSION ==========")

# Baseline model

baseline_regressor = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            DummyRegressor(strategy="median")
        )
    ]
)

baseline_regressor.fit(
    X_train,
    y_reg_train
)

baseline_predictions = baseline_regressor.predict(
    X_test
)

# Random Forest regression model

eta_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

eta_model.fit(
    X_train,
    y_reg_train
)

eta_predictions = eta_model.predict(X_test)

# Regression evaluation

def regression_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(
            actual, predicted
        )
    }


baseline_reg_metrics = regression_metrics(
    y_reg_test,
    baseline_predictions
)

eta_metrics = regression_metrics(
    y_reg_test,
    eta_predictions
)

print("\nBaseline Regression Results:")
print(baseline_reg_metrics)

print("\nRandom Forest Regression Results:")
print(eta_metrics)


# ==========================================
# 7. CLASSIFICATION MODEL - DELAY PREDICTION
# ==========================================

print("\n========== DELAY CLASSIFICATION ==========")

# Baseline classifier

baseline_classifier = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)

baseline_classifier.fit(
    X_train,
    y_cls_train
)

baseline_class_predictions = (
    baseline_classifier.predict(X_test)
)

# Random Forest classifier

delay_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
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

delay_model.fit(
    X_train,
    y_cls_train
)

class_predictions = delay_model.predict(X_test)

class_probabilities = (
    delay_model.predict_proba(X_test)[:, 1]
)

# Classification evaluation

def classification_metrics(actual, predicted, probabilities=None):

    results = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1": f1_score(
            actual, predicted, zero_division=0
        )
    }

    if (
        probabilities is not None
        and len(np.unique(actual)) == 2
    ):
        results["ROC_AUC"] = roc_auc_score(
            actual, probabilities
        )
    else:
        results["ROC_AUC"] = np.nan

    return results


baseline_cls_metrics = classification_metrics(
    y_cls_test,
    baseline_class_predictions
)

delay_metrics = classification_metrics(
    y_cls_test,
    class_predictions,
    class_probabilities
)

print("\nBaseline Classification Results:")
print(baseline_cls_metrics)

print("\nRandom Forest Classification Results:")
print(delay_metrics)

print("\nClassification Report:")
print(
    classification_report(
        y_cls_test,
        class_predictions,
        target_names=["Not Delayed", "Delayed"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_cls_test,
        class_predictions
    )
)


# ==========================================
# 8. SAVE MODEL COMPARISON
# ==========================================

metrics_df = pd.DataFrame([
    {
        "Task": "ETA Regression",
        "Model": "Baseline - Median",
        **baseline_reg_metrics
    },
    {
        "Task": "ETA Regression",
        "Model": "Random Forest",
        **eta_metrics
    },
    {
        "Task": "Delay Classification",
        "Model": "Baseline - Most Frequent",
        **baseline_cls_metrics
    },
    {
        "Task": "Delay Classification",
        "Model": "Random Forest",
        **delay_metrics
    }
])

metrics_path = REPORT_DIR / "model_comparison.csv"

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ==========================================
# 9. SAVE TEST PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Order_Id": df["Order_Id"].iloc[split_index:].values,
    "Order_Date": df["Order_Date"].iloc[split_index:].values,

    "Actual_Delivery_Days": y_reg_test.values,
    "Predicted_Delivery_Days": np.round(
        eta_predictions, 2
    ),

    "Actual_Delay_Days": (
        y_reg_test.values
        - X_test["Scheduled_Days"].values
    ),

    "Actual_Is_Delayed": y_cls_test.values,
    "Predicted_Is_Delayed": class_predictions,

    "Delay_Probability": np.round(
        class_probabilities, 4
    )
})

predictions_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)


# ==========================================
# 10. SAVE TRAINED MODELS
# ==========================================

joblib.dump(
    eta_model,
    MODEL_DIR / "eta_regression_model.joblib"
)

joblib.dump(
    delay_model,
    MODEL_DIR / "delay_classification_model.joblib"
)

print("\nModels saved successfully.")


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== TRAINING COMPLETED ==========")

print("ETA model: Random Forest Regressor")
print("Delay model: Random Forest Classifier")

print("\nGenerated files:")

print("1. reports/model_comparison.csv")
print("2. reports/test_predictions.csv")

print("3. src/models/artifacts/eta_regression_model.joblib")
print("4. src/models/artifacts/delay_classification_model.joblib")

print("\nShipPulse AI model training completed!")
