
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/raw/DataCoSupplyChainDataset.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("SHIPPING PERFORMANCE VALIDATION")
print("==================================\n")


df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Original records:", len(df))


order_df = (
    df.groupby("Order Id")
    .agg(
        Actual_Days=("Days for shipping (real)", "first"),
        Scheduled_Days=("Days for shipment (scheduled)", "first"),
        Shipping_Mode=("Shipping Mode", "first"),
        Delivery_Status=("Delivery Status", "first"),
        Order_Status=("Order Status", "first")
    )
    .reset_index()
)

print("\nORDER LEVEL INFORMATION")
print("Total unique orders:", len(order_df))



valid_orders = order_df[
    order_df["Delivery_Status"] != "Shipping canceled"
].copy()

print("Cancelled shipments excluded:",
      len(order_df) - len(valid_orders))

print("Eligible shipments:", len(valid_orders))


valid_orders["Delay_Days"] = (
    valid_orders["Actual_Days"]
    - valid_orders["Scheduled_Days"]
)

valid_orders["Delay_Hours"] = (
    valid_orders["Delay_Days"] * 24
)

valid_orders["Is_Delayed"] = (
    valid_orders["Delay_Days"] > 0
)



shipping_analysis = (
    valid_orders.groupby("Shipping_Mode")
    .agg(
        Total_Orders=("Order Id", "count"),
        Average_Actual_Days=("Actual_Days", "mean"),
        Average_Scheduled_Days=("Scheduled_Days", "mean"),
        Average_Delay_Hours=("Delay_Hours", "mean"),
        Delayed_Orders=("Is_Delayed", "sum")
    )
)

shipping_analysis["Delay_Rate_Percentage"] = (
    shipping_analysis["Delayed_Orders"]
    / shipping_analysis["Total_Orders"] * 100
)

print("\n========== SHIPPING MODE PERFORMANCE ==========")

print(shipping_analysis.round(2))

shipping_analysis.to_csv(
    REPORT_DIR / "shipping_analysis_cleaned.csv"
)


duration_comparison = pd.crosstab(
    valid_orders["Shipping_Mode"],
    [
        valid_orders["Scheduled_Days"],
        valid_orders["Actual_Days"]
    ]
)

print("\n========== SCHEDULED VS ACTUAL DURATION ==========")

print(duration_comparison)

duration_comparison.to_csv(
    REPORT_DIR / "scheduled_vs_actual_by_mode.csv"
)


valid_orders["Calculated_Status"] = valid_orders[
    "Delay_Days"
].apply(
    lambda x: (
        "Late delivery" if x > 0
        else "Advance shipping" if x < 0
        else "Shipping on time"
    )
)


status_validation = pd.crosstab(
    valid_orders["Delivery_Status"],
    valid_orders["Calculated_Status"]
)

print("\n========== DELIVERY STATUS VALIDATION ==========")

print(status_validation)

status_validation.to_csv(
    REPORT_DIR / "delivery_status_validation.csv"
)



valid_orders["Label_Mismatch"] = (
    valid_orders["Delivery_Status"]
    != valid_orders["Calculated_Status"]
)

mismatch_count = valid_orders["Label_Mismatch"].sum()

mismatch_percentage = (
    mismatch_count / len(valid_orders) * 100
)

print("\n========== LABEL VALIDATION RESULTS ==========")

print("Total eligible shipments:", len(valid_orders))

print("Matching labels:",
      len(valid_orders) - mismatch_count)

print("Mismatched labels:", mismatch_count)

print("Mismatch percentage:",
      round(mismatch_percentage, 2), "%")



mismatched_orders = valid_orders[
    valid_orders["Label_Mismatch"]
]

mismatched_orders.to_csv(
    REPORT_DIR / "mismatched_delivery_labels.csv",
    index=False
)


delay_distribution = (
    valid_orders["Delay_Hours"]
    .value_counts()
    .sort_index()
    .rename_axis("Delay_Hours")
    .reset_index(name="Number_Of_Orders")
)

print("\n========== DELAY DISTRIBUTION ==========")

print(delay_distribution.to_string(index=False))

delay_distribution.to_csv(
    REPORT_DIR / "delay_distribution.csv",
    index=False
)



print("\n==================================")
print("SHIPPULSE AI - VALIDATION COMPLETED")
print("==================================")

print("\nGenerated reports:")

print("1. shipping_analysis_cleaned.csv")
print("2. scheduled_vs_actual_by_mode.csv")
print("3. delivery_status_validation.csv")
print("4. mismatched_delivery_labels.csv")
print("5. delay_distribution.csv")

print("\nAll reports saved inside:", REPORT_DIR)
