
import pandas as pd
import numpy as np
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"



print("\n========== SHIPPULSE ==========")
print("DATASET INSPECTION & QUALITY AUDIT")
print("==================================\n")

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Dataset loaded successfully!\n")



print("STEP 1: DATASET DIMENSIONS")

print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

print(
    "Memory usage:",
    round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    "MB"
)



print("\nSTEP 2: COLUMN INFORMATION\n")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Non_Null_Count": df.notna().sum().values,
    "Missing_Count": df.isna().sum().values,
    "Unique_Values": df.nunique().values
})

print(column_info.to_string(index=False))

# Save report
report_dir = BASE_DIR / "reports"
report_dir.mkdir(exist_ok=True)

column_info.to_csv(
    report_dir / "column_information.csv",
    index=False
)


print("\nSTEP 3: MISSING VALUE ANALYSIS\n")

missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

missing_report = missing_report.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print(
    missing_report[missing_report["Missing_Count"] > 0]
    .to_string()
)

missing_report.to_csv(
    report_dir / "missing_value_report.csv"
)


print("\nSTEP 4: DUPLICATE ANALYSIS\n")

print("Exact duplicate rows:", df.duplicated().sum())

print(
    "Duplicate Order Item IDs:",
    df["Order Item Id"].duplicated().sum()
)

print("Unique Orders:", df["Order Id"].nunique())

print("Unique Customers:", df["Customer Id"].nunique())

print("Unique Products:", df["Product Card Id"].nunique())



print("\nSTEP 5: DATE RANGE ANALYSIS\n")

date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for col in date_columns:

    dates = pd.to_datetime(
        df[col],
        errors="coerce"
    )

    print(f"\n{col}")
    print("Minimum:", dates.min())
    print("Maximum:", dates.max())
    print("Invalid dates:", dates.isna().sum())



print("\nSTEP 6: CATEGORICAL DISTRIBUTION\n")

categorical_columns = [
    "Shipping Mode",
    "Delivery Status",
    "Market",
    "Order Region",
    "Order Status",
    "Customer Segment"
]

for col in categorical_columns:

    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))



print("\nSTEP 7: ORDER LEVEL ANALYSIS\n")

order_counts = df.groupby("Order Id").size()

print("Total unique orders:", len(order_counts))

print("Average items per order:", round(order_counts.mean(), 2))

print("Maximum items in an order:", order_counts.max())


print("\n==================================")
print("INITIAL DATA AUDIT COMPLETED!")
print("Reports saved inside reports/")
print("==================================")
