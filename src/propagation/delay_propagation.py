
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import timedelta


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/processed/modeling_dataset.csv"
MODEL_PATH = (
    BASE_DIR
    / "src/models/artifacts/eta_regression_model.joblib"
)

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SHIPPULSE AI ==========")
print("DELAY PROPAGATION ENGINE")
print("==================================\n")


STAGES = [
    {
        "stage": "Order Processing",
        "duration_share": 0.10,
        "buffer_hours": 2
    },
    {
        "stage": "Warehouse Handling",
        "duration_share": 0.20,
        "buffer_hours": 3
    },
    {
        "stage": "Carrier Transit",
        "duration_share": 0.40,
        "buffer_hours": 4
    },
    {
        "stage": "Destination Handling",
        "duration_share": 0.15,
        "buffer_hours": 2
    },
    {
        "stage": "Final-Mile Delivery",
        "duration_share": 0.15,
        "buffer_hours": 1
    }
]

assert np.isclose(
    sum(stage["duration_share"] for stage in STAGES),
    1.0
), "Stage duration shares must add up to 100%."



SCENARIOS = {
    "Baseline": {},

    "Order Processing Delay": {
        "Order Processing": 8
    },

    "Warehouse Delay": {
        "Warehouse Handling": 12
    },

    "Carrier Disruption": {
        "Carrier Transit": 24
    },

    "Destination Delay": {
        "Destination Handling": 12
    },

    "Final Mile Delay": {
        "Final-Mile Delivery": 6
    },

    "Multiple Disruptions": {
        "Warehouse Handling": 12,
        "Carrier Transit": 24,
        "Destination Handling": 12
    }
}


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Modelling dataset not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ETA model not found: {MODEL_PATH}. "
        "Run train_models.py first."
    )

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

eta_model = joblib.load(MODEL_PATH)

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
    "Number_Of_Items"
]

missing_features = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

test_start = int(len(df) * 0.80)

if test_start >= len(df):
    raise ValueError("No ETA test records available.")

sample = df.iloc[[test_start]].copy()

sample_features = sample[feature_columns]

predicted_eta_days = float(
    eta_model.predict(sample_features)[0]
)


predicted_eta_days = max(
    0.0,
    predicted_eta_days
)

order_id = sample.iloc[0]["Order_Id"]
order_date = sample.iloc[0]["Order_Date"]

scheduled_days = float(
    sample.iloc[0]["Scheduled_Days"]
)

print("Sample order:", order_id)
print("Order date:", order_date)
print("Scheduled duration:", scheduled_days, "days")
print(
    "ML predicted duration:",
    round(predicted_eta_days, 2),
    "days"
)



def create_stage_schedule(predicted_days):
    """
    Allocate the ML-predicted total duration
    across assumed shipment stages.

    Stage duration shares and buffers are
    illustrative modelling assumptions.
    """

    total_hours = predicted_days * 24

    schedule = []

    for stage in STAGES:

        planned_hours = (
            total_hours * stage["duration_share"]
        )

        schedule.append({
            "Stage": stage["stage"],
            "Planned_Duration_Hours": planned_hours,
            "Buffer_Hours": stage["buffer_hours"]
        })

    return pd.DataFrame(schedule)



def simulate_propagation(
    schedule,
    disruptions,
    scenario_name
):
    """
    Simulate delay movement through consecutive stages.

    incoming_delay:
        Delay received from the previous stage.

    absorbed_delay:
        Part of incoming delay absorbed by the
        current stage's available buffer.

    local_disruption:
        New delay introduced at the current stage.

    outgoing_delay:
        Delay passed to the next stage.

    Assumption:
        Each stage's buffer is available to absorb
        incoming delay, not its own local disruption.
    """

    incoming_delay = 0.0

    results = []

    for _, stage in schedule.iterrows():

        stage_name = stage["Stage"]

        buffer_hours = float(
            stage["Buffer_Hours"]
        )

        planned_hours = float(
            stage["Planned_Duration_Hours"]
        )

        local_disruption = float(
            disruptions.get(stage_name, 0)
        )

        

        absorbed_delay = min(
            incoming_delay,
            buffer_hours
        )

        residual_incoming_delay = (
            incoming_delay - absorbed_delay
        )

       

        outgoing_delay = (
            residual_incoming_delay
            + local_disruption
        )

        results.append({
            "Scenario": scenario_name,
            "Stage": stage_name,
            "Planned_Duration_Hours": round(
                planned_hours, 2
            ),
            "Buffer_Hours": round(
                buffer_hours, 2
            ),
            "Incoming_Delay_Hours": round(
                incoming_delay, 2
            ),
            "Absorbed_Delay_Hours": round(
                absorbed_delay, 2
            ),
            "Residual_Upstream_Delay_Hours": round(
                residual_incoming_delay, 2
            ),
            "Local_Disruption_Hours": round(
                local_disruption, 2
            ),
            "Outgoing_Delay_Hours": round(
                outgoing_delay, 2
            )
        })

        incoming_delay = outgoing_delay

    return pd.DataFrame(results)




schedule = create_stage_schedule(
    predicted_eta_days
)

all_stage_results = []
scenario_summaries = []

for scenario_name, disruptions in SCENARIOS.items():

    result = simulate_propagation(
        schedule,
        disruptions,
        scenario_name
    )

    all_stage_results.append(result)

    final_delay_hours = float(
        result.iloc[-1]["Outgoing_Delay_Hours"]
    )

    revised_eta_days = (
        predicted_eta_days
        + final_delay_hours / 24
    )

    scenario_summaries.append({
        "Order_Id": order_id,
        "Scenario": scenario_name,
        "ML_Predicted_ETA_Days": round(
            predicted_eta_days, 2
        ),
        "Additional_Delay_Hours": round(
            final_delay_hours, 2
        ),
        "Revised_ETA_Days": round(
            revised_eta_days, 2
        ),
        "Revised_ETA_Hours": round(
            revised_eta_days * 24, 2
        )
    })

stage_results_df = pd.concat(
    all_stage_results,
    ignore_index=True
)

summary_df = pd.DataFrame(
    scenario_summaries
)



stage_results_df.to_csv(
    REPORT_DIR / "stage_propagation_details.csv",
    index=False
)

summary_df.to_csv(
    REPORT_DIR / "propagation_scenario_summary.csv",
    index=False
)



print("\n========== SCENARIO COMPARISON ==========")

print(
    summary_df[
        [
            "Scenario",
            "ML_Predicted_ETA_Days",
            "Additional_Delay_Hours",
            "Revised_ETA_Days"
        ]
    ].to_string(index=False)
)




print("\n========== STAGE-LEVEL EXPLANATION ==========")

for scenario_name in SCENARIOS:

    print(f"\nScenario: {scenario_name}")

    scenario_data = stage_results_df[
        stage_results_df["Scenario"] == scenario_name
    ]

    print(
        scenario_data[
            [
                "Stage",
                "Incoming_Delay_Hours",
                "Absorbed_Delay_Hours",
                "Local_Disruption_Hours",
                "Outgoing_Delay_Hours"
            ]
        ].to_string(index=False)
    )




plot_df = summary_df[
    summary_df["Scenario"] != "Baseline"
].copy()

plot_df = plot_df.sort_values(
    "Additional_Delay_Hours"
)

plt.figure(figsize=(11, 6))

plt.barh(
    plot_df["Scenario"],
    plot_df["Additional_Delay_Hours"]
)

plt.xlabel("Additional Final Delay (Hours)")

plt.title(
    "ShipPulse AI - Delay Propagation by Scenario"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_delay_propagation_scenarios.png",
    dpi=300
)

plt.close()




selected_scenario = "Multiple Disruptions"

selected_data = stage_results_df[
    stage_results_df["Scenario"] == selected_scenario
].copy()

plt.figure(figsize=(11, 6))

plt.plot(
    selected_data["Stage"],
    selected_data["Incoming_Delay_Hours"],
    marker="o",
    label="Incoming Delay"
)

plt.plot(
    selected_data["Stage"],
    selected_data["Outgoing_Delay_Hours"],
    marker="o",
    label="Outgoing Delay"
)

plt.xticks(rotation=25, ha="right")

plt.ylabel("Delay (Hours)")

plt.title(
    "Stage-wise Delay Propagation - Multiple Disruptions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_stage_delay_propagation.png",
    dpi=300
)

plt.close()



print("\n==================================")
print("DELAY PROPAGATION COMPLETED")
print("==================================")

print("Sample Order:", order_id)

print(
    "Predicted ETA:",
    round(predicted_eta_days, 2),
    "days"
)

print("\nGenerated files:")

print("1. reports/stage_propagation_details.csv")
print("2. reports/propagation_scenario_summary.csv")
print("3. reports/figures/07_delay_propagation_scenarios.png")
print("4. reports/figures/08_stage_delay_propagation.png")

print(
    "\nNote: Stage durations, buffers and disruptions "
    "are simulated assumptions, not observed DataCo values."
)

print("\nShipPulse AI propagation engine completed!")
