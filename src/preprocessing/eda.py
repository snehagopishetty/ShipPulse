
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Dataset loaded:", df.shape)


df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"], errors="coerce"
)

df["shipping date (DateOrders)"] = pd.to_datetime(
    df["shipping date (DateOrders)"], errors="coerce"
)


order_columns = [
    "Order Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order City",
    "Order Status",
    "Customer Segment"
]

order_df = (
    df.groupby("Order Id")[order_columns[1:]]
    .first()
    .reset_index()
)


item_summary = (
    df.groupby("Order Id")
    .agg(
        Total_Quantity=("Order Item Quantity", "sum"),
        Total_Order_Value=("Order Item Total", "sum"),
        Number_Of_Items=("Order Item Id", "count")
    )
    .reset_index()
)

order_df = order_df.merge(
    item_summary,
    on="Order Id",
    how="left",
    validate="one_to_one"
)


order_df["Delay_Days"] = (
    order_df["Days for shipping (real)"]
    - order_df["Days for shipment (scheduled)"]
)

order_df["Delay_Hours"] = order_df["Delay_Days"] * 24

order_df["Delay_Category"] = np.select(
    [
        order_df["Delay_Days"] > 0,
        order_df["Delay_Days"] == 0
    ],
    [
        "Delayed",
        "On Time"
    ],
    default="Early"
)

order_df.to_csv(
    BASE_DIR / "data" / "processed" / "orders.csv",
    index=False
)

print("\nORDER LEVEL DATA CREATED")
print("Original records:", len(df))
print("Unique orders:", len(order_df))
print("Average items per order:", round(order_df["Number_Of_Items"].mean(), 2))

print("\nDelay distribution:")
print(order_df["Delay_Category"].value_counts())


# part B



plt.figure(figsize=(9, 5))

sns.countplot(
    data=order_df,
    x="Delay_Category",
    order=["Early", "On Time", "Delayed"]
)

plt.title("Shipment Delivery Performance")
plt.xlabel("Delivery Category")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(FIGURE_DIR / "01_delivery_performance.png", dpi=150)
plt.close()




plt.figure(figsize=(9, 5))

sns.histplot(
    data=order_df,
    x="Delay_Hours",
    discrete=True,
    kde=False
)

plt.title("Distribution of Shipment Delay Hours")
plt.xlabel("Delay (Hours)")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(FIGURE_DIR / "02_delay_distribution.png", dpi=150)
plt.close()




plt.figure(figsize=(10, 5))

sns.countplot(
    data=order_df,
    x="Shipping Mode",
    hue="Delay_Category",
    hue_order=["Early", "On Time", "Delayed"]
)

plt.title("Delivery Performance by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Number of Orders")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(FIGURE_DIR / "03_shipping_mode_analysis.png", dpi=150)
plt.close()




plt.figure(figsize=(11, 5))

sns.countplot(
    data=order_df,
    y="Market",
    hue="Delay_Category",
    hue_order=["Early", "On Time", "Delayed"]
)

plt.title("Delivery Performance Across Markets")
plt.xlabel("Number of Orders")
plt.ylabel("Market")
plt.tight_layout()

plt.savefig(FIGURE_DIR / "04_market_analysis.png", dpi=150)
plt.close()


print("\nEDA visualizations generated successfully!")



shipping_analysis = (
    order_df.groupby("Shipping Mode")
    .agg(
        Total_Orders=("Order Id", "count"),
        Average_Delay_Hours=("Delay_Hours", "mean"),
        Median_Delay_Hours=("Delay_Hours", "median"),
        Delayed_Orders=(
            "Delay_Days",
            lambda x: (x > 0).sum()
        )
    )
)

shipping_analysis["Delay_Rate_Percentage"] = (
    shipping_analysis["Delayed_Orders"]
    / shipping_analysis["Total_Orders"] * 100
)

shipping_analysis.to_csv(
    REPORT_DIR / "shipping_mode_analysis.csv"
)



market_analysis = (
    order_df.groupby("Market")
    .agg(
        Total_Orders=("Order Id", "count"),
        Average_Delay_Hours=("Delay_Hours", "mean"),
        Delayed_Orders=(
            "Delay_Days",
            lambda x: (x > 0).sum()
        )
    )
)

market_analysis["Delay_Rate_Percentage"] = (
    market_analysis["Delayed_Orders"]
    / market_analysis["Total_Orders"] * 100
)

market_analysis.to_csv(
    REPORT_DIR / "market_analysis.csv"
)



delay_summary = order_df["Delay_Hours"].describe()

delay_summary.to_csv(
    REPORT_DIR / "delay_summary.csv"
)

print("\nSHIPPING MODE ANALYSIS")
print(shipping_analysis.round(2))

print("\nMARKET ANALYSIS")
print(market_analysis.round(2))

print("\nDELAY STATISTICS")
print(delay_summary)

print("\nALL EDA REPORTS GENERATED!")
