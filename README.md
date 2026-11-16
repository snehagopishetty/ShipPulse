# ShipPulse AI
### AI-Powered Shipment ETA Prediction & Delay Propagation

**ShipPulse AI** is a machine-learning-based shipment intelligence prototype that predicts shipment delivery duration, identifies potential delivery delays, and simulates how operational disruptions propagate across different stages of a logistics network.

The project combines historical supply-chain data, predictive modelling, explainability techniques, and a stage-wise delay propagation simulator to provide an understandable view of shipment performance and potential delivery risks.

---

## 1. Problem Statement

Shipment delays can originate at different points in a logistics network, including order processing, warehouse handling, carrier transit, destination handling, and final-mile delivery.

A disruption at one stage can affect subsequent stages and increase the final delivery time.

ShipPulse AI addresses three objectives:

- **ETA Prediction:** Estimate shipment delivery duration using historical order and shipping characteristics.
- **Delay Prediction:** Classify whether a shipment is likely to exceed its scheduled delivery duration.
- **Delay Propagation:** Simulate how disruptions at individual logistics stages affect downstream operations and the final delivery estimate.

---

## 2. Key Features

| Feature | Description |
|---|---|
| ETA Prediction | Predicts actual shipment duration using a Random Forest Regressor |
| Delay Classification | Estimates the likelihood of a shipment being delayed |
| Feature Importance | Uses permutation importance to understand influential model features |
| Delay Propagation | Simulates the movement of delays across five logistics stages |
| Buffer Absorption | Models how operational buffers absorb incoming delays |
| Scenario Analysis | Compares baseline delivery estimates against disruption scenarios |
| Interactive Dashboard | Allows users to select shipments and experiment with operational disruptions |

---

## 3. System Architecture

```text
                 DataCo Supply Chain Dataset
                            |
                            v
                  Data Audit & Validation
                            |
                            v
                  Exploratory Data Analysis
                            |
                            v
                   Order-Level Aggregation
                            |
                            v
                    Feature Engineering
                            |
                 +----------+----------+
                 |                     |
                 v                     v
           ETA Regression       Delay Classification
                 |                     |
                 v                     v
           ETA Prediction       Delay Probability
                 |                     |
                 +----------+----------+
                            |
                            v
                   Explainability Analysis
                            |
                            v
                 Delay Propagation Engine
                            |
                            v
                  Scenario Simulation
                            |
                            v
                  Streamlit Dashboard
```

---

## 4. Dataset

**Dataset:** DataCo Smart Supply Chain Dataset

**Source:** [Kaggle – DataCo Smart Supply Chain](https://www.kaggle.com/datasets/jolenechen/datacosupplychaindataset)

The dataset contains historical supply-chain and delivery information, including:

- Order and shipping dates
- Shipping modes
- Scheduled and actual shipping durations
- Delivery status
- Destination information
- Customer segments
- Order quantities and values

### Dataset preparation

| Description | Records |
|---|---:|
| Original order-item records | 180,519 |
| Unique orders | 65,752 |
| Cancelled orders excluded | 2,855 |
| Non-cancelled orders | 62,897 |

The original item-level records were aggregated into an order-level dataset to avoid treating multiple items belonging to the same order as independent shipments.

Personal customer information and irrelevant identifiers were excluded from model features.

---

## 5. Machine Learning Approach

### A. ETA Regression

**Objective:** Predict actual shipping duration in days.

**Algorithm:** Random Forest Regressor

The model uses order-level features such as:

- Shipping mode
- Scheduled delivery duration
- Market and destination information
- Customer segment
- Order placement date features
- Total order quantity
- Total order value
- Number of order items

**Evaluation metrics:**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### B. Delay Classification

**Objective:** Predict whether a shipment will exceed its scheduled delivery duration.

The target is defined as:

```python
Is_Delayed = (
    Actual_Delivery_Days > Scheduled_Days
).astype(int)
```

A Random Forest Classifier is used to estimate delay probabilities.

A classification threshold of **0.3424** was selected using validation data to achieve at least 75% recall, choosing the highest precision among qualifying thresholds.

The selected threshold was then evaluated on the held-out test dataset.

---

## 6. Model Evaluation

### ETA Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline – Median | 1.3981 | 1.6967 | -0.0939 |
| Random Forest | **0.9972** | **1.2727** | **0.3845** |

The Random Forest model outperformed the median baseline across all three regression metrics.

### Delay Classification Results

| Metric | Final Test Result |
|---|---:|
| Accuracy | 62.61% |
| Precision | 63.52% |
| Recall | 82.24% |
| F1 Score | 71.68% |
| F2 Score | 77.67% |
| ROC-AUC | 74.63% |

The selected classification threshold prioritizes identifying delayed shipments, resulting in higher recall.

The model identifies approximately 82% of delayed shipments in the held-out test set, with a corresponding trade-off in precision.

---

## 7. Model Explainability

Permutation importance was used to evaluate how much model performance changes when individual features are shuffled.

### ETA Regression

The most influential features were:

1. Scheduled delivery duration
2. Shipping mode

Scheduled delivery duration had the largest measured importance, followed by shipping mode.

### Delay Classification

Shipping mode was the most influential feature, followed by scheduled delivery duration and destination country.

These findings represent predictive associations in the dataset and should not be interpreted as proof of causal relationships.

---

## 8. Delay Propagation Engine

A separate simulation engine demonstrates how operational disruptions can propagate through five logistics stages.

### Logistics stages

| Stage | Illustrative buffer |
|---|---:|
| Order Processing | 2 hours |
| Warehouse Handling | 3 hours |
| Carrier Transit | 4 hours |
| Destination Handling | 2 hours |
| Final-Mile Delivery | 1 hour |

### Propagation logic

For each stage:

1. Receive the delay from the previous stage.
2. Absorb available incoming delay using the configured buffer.
3. Calculate the remaining upstream delay.
4. Add any new disruption introduced at the current stage.
5. Pass the resulting delay to the next stage.

The calculation is:

```text
Residual Delay = Incoming Delay - Absorbed Delay

Outgoing Delay = Residual Delay + Local Disruption
```

### Example: Multiple Disruptions

| Stage | Incoming | Absorbed | New disruption | Outgoing |
|---|---:|---:|---:|---:|
| Order Processing | 0h | 0h | 0h | 0h |
| Warehouse Handling | 0h | 0h | 12h | 12h |
| Carrier Transit | 12h | 4h | 24h | 32h |
| Destination Handling | 32h | 2h | 12h | 42h |
| Final-Mile Delivery | 42h | 1h | 0h | 41h |

In this illustrative scenario, 48 hours of local disruptions are introduced. Downstream buffers absorb 7 hours, leaving a final propagated delay of 41 hours.

The simulation demonstrates how delays originating at different stages can influence the final shipment ETA.

**Important:** The DataCo dataset does not contain observed intermediate-stage timestamps or actual durations for these five logistics stages. Stage allocations, buffers, and disruptions are explicitly defined simulation assumptions, not historical observations or outputs learned by the ML model.

---

## 9. Interactive Dashboard

The Streamlit dashboard provides an interactive interface for exploring shipment predictions and disruption scenarios.

### Dashboard capabilities

- Select a shipment from the held-out test dataset.
- View its predicted shipping duration and scheduled duration.
- Introduce hypothetical disruptions at individual stages.
- Observe incoming and outgoing delay at every stage.
- Understand how operational buffers absorb delays.
- Compare baseline and revised delivery estimates.
- Inspect model information and simulation assumptions.

## 10. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Explainability | Permutation Importance |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Development Environment | VS Code |

---

## 11. Project Structure

```text
ShipPulse/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── DataCoSupplyChainDataset.csv
│   │
│   └── processed/
│       ├── orders.csv
│       ├── modeling_dataset.csv
│       └── shipping_analysis_cleaned.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── data_audit.py
│   │   ├── eda.py
│   │   ├── data_validation.py
│   │   ├── shipping_analysis.py
│   │   └── prepare_model_data.py
│   │
│   ├── models/
│   │   └── artifacts/
│   │
│   ├── propagation/
│   │   └── delay_propagation.py
│   │
│   └── explainability/
│
├── reports/
│   ├── figures/
│   ├── screenshots/
│   ├── feature_documentation.csv
│   ├── eta_feature_importance.csv
│   ├── delay_feature_importance.csv
│   ├── stage_propagation_details.csv
│   └── propagation_scenario_summary.csv
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 12. Installation and Execution

### Prerequisites

- Python 3.9 or higher
- pip
- Git

### Step 1: Clone the repository


### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

Download the DataCo dataset from Kaggle and place it under:

```text
data/raw/DataCoSupplyChainDataset.csv
```

### Step 5: Execute preprocessing

Run the preprocessing scripts from the project root in the required order:

```bash
python src/preprocessing/data_audit.py

python src/preprocessing/eda.py

python src/preprocessing/data_validation.py

python src/preprocessing/shipping_analysis.py

python src/preprocessing/prepare_model_data.py
```

### Step 6: Train and save the models

Execute the model-training scripts included in the repository to generate the trained model artifacts.

### Step 7: Run the propagation engine

```bash
python src/propagation/delay_propagation.py
```

### Step 8: Launch the dashboard

```bash
streamlit run app/app.py
```

---

## 13. Limitations and Future Improvements

### Current limitations

- Historical delivery data does not provide complete intermediate-stage shipment timestamps.
- The propagation engine uses explicitly configured stage buffers and hypothetical disruptions.
- The ETA model predicts shipping duration rather than a fully operational, real-time delivery timestamp.
- Model performance is dependent on the available historical dataset and its features.
- The dataset contains historical records and does not provide live carrier or weather information.

### Future improvements

- Integrate real-time carrier tracking APIs.
- Incorporate weather, port congestion, and route disruption data.
- Develop stage-specific ETA prediction models using actual stage timestamps.
- Learn operational buffer estimates from historical stage-level observations.
- Introduce continuous monitoring and model retraining.
- Add alerting for shipments with high predicted delay probability.

---

## 14. Conclusion

ShipPulse AI demonstrates how machine learning and transparent operational simulation can be combined to improve shipment visibility.

The project brings together ETA regression, delay classification, feature importance analysis, and a multi-stage disruption propagation engine in an interactive dashboard.

By separating historical ML predictions from simulated logistics assumptions, the prototype provides an explainable foundation for exploring shipment delays and their downstream impact.

---

