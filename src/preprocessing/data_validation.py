
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/raw/DataCoSupplyChainDataset.csv"
REPORT_DIR = BASE_DIR / "reports"

df = pd.read_csv(DATA_PATH, encoding="latin1")

# Columns that should remain consistent within an order
order_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Shipping Mode",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Status",
    "Market",
    "Order Region",
    "Order Country"
]

# Check consistency across order-item records
consistency = (
    df.groupby("Order Id")[order_columns]
    .nunique(dropna=False)
)

inconsistent_orders = consistency[
    (consistency > 1).any(axis=1)
]

print("Total unique orders:", df["Order Id"].nunique())
print("Orders with inconsistent shipping information:",
      len(inconsistent_orders))

if len(inconsistent_orders) > 0:
    print(inconsistent_orders.head(20))

inconsistent_orders.to_csv(
    REPORT_DIR / "inconsistent_orders.csv"
)

# Inspect delivery and order statuses
print("\nDelivery Status Distribution:")
print(df.groupby("Order Id")["Delivery Status"].first().value_counts())

print("\nOrder Status Distribution:")
print(df.groupby("Order Id")["Order Status"].first().value_counts())

print("\nDelivery Status vs Order Status:")
status_comparison = pd.crosstab(
    df["Delivery Status"],
    df["Order Status"]
)

print(status_comparison)

status_comparison.to_csv(
    REPORT_DIR / "delivery_order_status_comparison.csv"
)

# Exact delay distribution at order level
order_df = df.groupby("Order Id").agg(
    Actual_Days=("Days for shipping (real)", "first"),
    Scheduled_Days=("Days for shipment (scheduled)", "first"),
    Shipping_Mode=("Shipping Mode", "first"),
    Delivery_Status=("Delivery Status", "first")
).reset_index()

order_df["Delay_Hours"] = (
    order_df["Actual_Days"] - order_df["Scheduled_Days"]
) * 24

print("\nExact delay distribution:")
print(order_df["Delay_Hours"].value_counts().sort_index())

print("\nDelay statistics:")
print(order_df["Delay_Hours"].describe())

# Calculate percentage-based shipping performance
mode_analysis = pd.crosstab(
    order_df["Shipping_Mode"],
    order_df["Delay_Hours"].gt(0),
    normalize="index"
) * 100

mode_analysis.columns = ["On_Time_or_Early_%", "Delayed_%"]

print("\nShipping mode delay percentages:")
print(mode_analysis.round(2))

mode_analysis.to_csv(
    REPORT_DIR / "shipping_mode_delay_percentages.csv"
)

print("\nVALIDATION COMPLETED!")
