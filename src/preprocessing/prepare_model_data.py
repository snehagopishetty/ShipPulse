
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/processed/orders.csv"
OUTPUT_DIR = BASE_DIR / "data/processed"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("FEATURE ENGINEERING PIPELINE")
print("==================================\n")




if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Order-level dataset not found: {INPUT_PATH}\n"
        "Run src/preprocessing/eda.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")
print("Original orders:", len(df))


column_mapping = {
    "Order Id": "Order_Id",
    "Delivery Status": "Delivery_Status",
    "Shipping Mode": "Shipping_Mode",
    "Order Region": "Order_Region",
    "Order Country": "Order_Country",
    "Customer Segment": "Customer_Segment",
    "order date (DateOrders)": "Order_Date",
    "Days for shipment (scheduled)": "Scheduled_Days",
    "Days for shipping (real)": "Actual_Days",
}

df = df.rename(columns=column_mapping)

required_columns = [
    "Order_Id",
    "Delivery_Status",
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Order_Country",
    "Customer_Segment",
    "Order_Date",
    "Scheduled_Days",
    "Actual_Days",
    "Total_Quantity",
    "Total_Order_Value",
    "Number_Of_Items",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns missing from orders.csv: "
        f"{missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

print("Column validation passed.")




original_count = len(df)

df = df[
    df["Delivery_Status"] != "Shipping canceled"
].copy()

cancelled_count = original_count - len(df)

print("\nCancelled orders excluded:", cancelled_count)
print("Orders after cancellation filtering:", len(df))


df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

invalid_dates = df["Order_Date"].isna().sum()

print("\nInvalid order dates:", invalid_dates)

df = df.dropna(
    subset=["Order_Date"]
).copy()

print("Orders after date validation:", len(df))




numeric_columns = [
    "Scheduled_Days",
    "Actual_Days",
    "Total_Quantity",
    "Total_Order_Value",
    "Number_Of_Items",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

invalid_numeric_rows = df[
    numeric_columns
].isna().any(axis=1).sum()

print("Rows with invalid numeric values:", invalid_numeric_rows)



df = df.dropna(
    subset=numeric_columns
).copy()


assert (df["Scheduled_Days"] >= 0).all(), (
    "Negative scheduled delivery duration found."
)

assert (df["Actual_Days"] >= 0).all(), (
    "Negative actual delivery duration found."
)

assert (df["Total_Quantity"] > 0).all(), (
    "Non-positive order quantity found."
)

assert (df["Number_Of_Items"] > 0).all(), (
    "Non-positive number of order items found."
)




df["Order_Year"] = df["Order_Date"].dt.year

df["Order_Month"] = df["Order_Date"].dt.month

df["Order_DayOfWeek"] = (
    df["Order_Date"].dt.dayofweek
)

df["Order_Day"] = df["Order_Date"].dt.day

df["Is_Weekend"] = (
    df["Order_DayOfWeek"] >= 5
).astype(int)



df["Actual_Delivery_Days"] = df["Actual_Days"]


df["Delay_Days"] = (
    df["Actual_Delivery_Days"]
    - df["Scheduled_Days"]
)

df["Delay_Hours"] = df["Delay_Days"] * 24


df["Is_Delayed"] = (
    df["Delay_Days"] > 0
).astype(int)


feature_columns = [
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Order_Country",
    "Customer_Segment",
    "Scheduled_Days",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek",
    "Order_Day",
    "Is_Weekend",
    "Total_Quantity",
    "Total_Order_Value",
    "Number_Of_Items",
]

target_columns = [
    "Actual_Delivery_Days",
    "Delay_Days",
    "Delay_Hours",
    "Is_Delayed",
]


metadata_columns = [
    "Order_Id",
    "Order_Date",
]

model_df = df[
    metadata_columns
    + feature_columns
    + target_columns
].copy()




leakage_columns = {
    "Actual_Days",
    "Actual_Delivery_Days",
    "Delay_Days",
    "Delay_Hours",
    "Is_Delayed",
    "Delivery_Status",
    "Late_delivery_risk",
    "Order_Status",
    "shipping date (DateOrders)",
}

leaking_features = (
    set(feature_columns) & leakage_columns
)

assert not leaking_features, (
    f"Target leakage detected: {leaking_features}"
)

print("\nTarget leakage check passed.")




print("\n========== DATA QUALITY CHECKS ==========")

duplicate_count = (
    model_df["Order_Id"].duplicated().sum()
)

missing_feature_values = (
    model_df[feature_columns]
    .isnull()
    .sum()
    .sum()
)

missing_target_values = (
    model_df[target_columns]
    .isnull()
    .sum()
    .sum()
)

print("Duplicate orders:", duplicate_count)

print(
    "Missing feature values:",
    missing_feature_values
)

print(
    "Missing target values:",
    missing_target_values
)

assert model_df["Order_Id"].is_unique, (
    "Duplicate order IDs found!"
)

assert not model_df[
    feature_columns
].isnull().any().any(), (
    "Missing feature values found. "
    "Investigate before modelling."
)

assert not model_df[
    target_columns
].isnull().any().any(), (
    "Missing target values found."
)

assert model_df["Is_Delayed"].isin(
    [0, 1]
).all(), (
    "Invalid delay classification values."
)

print("All data quality checks passed.")



model_df = model_df.sort_values(
    by=["Order_Date", "Order_Id"]
).reset_index(drop=True)



model_path = (
    OUTPUT_DIR / "modeling_dataset.csv"
)

model_df.to_csv(
    model_path,
    index=False
)

print("\nModelling dataset saved:", model_path)


feature_descriptions = [
    "Selected shipping service at order placement",
    "Geographical market",
    "Destination region",
    "Destination country",
    "Customer segment",
    "Scheduled shipping duration in days",
    "Year of order placement",
    "Month of order placement",
    "Day of week of order placement (Monday=0)",
    "Day of month of order placement",
    "Whether the order was placed on a weekend",
    "Total quantity across all order items",
    "Aggregated monetary value of order items",
    "Number of order-item records",
]

feature_types = [
    "Categorical",
    "Categorical",
    "Categorical",
    "Categorical",
    "Categorical",
    "Numerical",
    "Numerical",
    "Numerical",
    "Numerical",
    "Numerical",
    "Binary",
    "Numerical",
    "Numerical",
    "Numerical",
]

feature_documentation = pd.DataFrame({
    "Feature": feature_columns,
    "Description": feature_descriptions,
    "Feature_Type": feature_types,
    "Available_At_Prediction": [
        "Yes"
    ] * len(feature_columns),
})

feature_documentation.to_csv(
    REPORT_DIR / "feature_documentation.csv",
    index=False
)

print("Feature documentation saved.")



preparation_summary = pd.DataFrame([
    {
        "Metric": "Original orders",
        "Value": original_count,
    },
    {
        "Metric": "Cancelled orders excluded",
        "Value": cancelled_count,
    },
    {
        "Metric": "Invalid order dates",
        "Value": invalid_dates,
    },
    {
        "Metric": "Invalid numeric rows",
        "Value": invalid_numeric_rows,
    },
    {
        "Metric": "Final modelling records",
        "Value": len(model_df),
    },
    {
        "Metric": "Number of input features",
        "Value": len(feature_columns),
    },
    {
        "Metric": "Delayed orders",
        "Value": int(model_df["Is_Delayed"].sum()),
    },
    {
        "Metric": "Not delayed orders",
        "Value": int(
            (model_df["Is_Delayed"] == 0).sum()
        ),
    },
])

preparation_summary.to_csv(
    REPORT_DIR / "data_preparation_summary.csv",
    index=False
)



print("\n========== TARGET DISTRIBUTION ==========")

print("\nActual shipping duration (days):")
print(
    model_df["Actual_Delivery_Days"].describe()
)

print("\nDelay classification:")

print(
    model_df["Is_Delayed"]
    .value_counts()
    .rename(index={
        0: "Not Delayed",
        1: "Delayed",
    })
)

print(
    "\nAverage delay hours:",
    round(model_df["Delay_Hours"].mean(), 2)
)

print(
    "Delay rate (%):",
    round(
        model_df["Is_Delayed"].mean() * 100,
        2
    )
)

print(
    "\nDate range:",
    model_df["Order_Date"].min(),
    "to",
    model_df["Order_Date"].max()
)


print("\n==================================")
print("FEATURE ENGINEERING COMPLETED")
print("==================================")

print("Final modelling records:", len(model_df))
print("Total input features:", len(feature_columns))

print("\nRegression target: Actual_Delivery_Days")
print("Classification target: Is_Delayed")

print("\nGenerated files:")
print("1. data/processed/modeling_dataset.csv")
print("2. reports/feature_documentation.csv")
print("3. reports/data_preparation_summary.csv")

print("\nShipPulse AI preparation completed!")
