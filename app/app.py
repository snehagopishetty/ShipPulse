import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data/processed/modeling_dataset.csv"
MODEL_PATH = BASE_DIR / "src/models/artifacts/eta_regression_model.joblib"

STAGES = [
    {"stage": "Order Processing", "share": 0.10, "buffer": 2},
    {"stage": "Warehouse Handling", "share": 0.20, "buffer": 3},
    {"stage": "Carrier Transit", "share": 0.40, "buffer": 4},
    {"stage": "Destination Handling", "share": 0.15, "buffer": 2},
    {"stage": "Final-Mile Delivery", "share": 0.15, "buffer": 1},
]

STAGE_ORDER = [s["stage"] for s in STAGES]

st.set_page_config(
    page_title="ShipPulse AI | Control Tower",
    page_icon="🚢",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

.stApp {
    background: #f4f7fb;
    color: #172033;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2.5rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #0b1728;
}

[data-testid="stSidebar"] * {
    color: #e7eef8 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #b8c7da !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #e7eef8 !important;
    font-weight: 600;
}

[data-testid="stSidebar"] hr {
    border-color: #263a52;
}

.brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(
        115deg,
        #0b1728,
        #12304c 65%,
        #155e75
    );
    padding: 23px 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 28px #0f233c18;
}

.brand-title {
    font-family: Manrope, sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.brand-sub {
    font-size: 13px;
    color: #c7d7e8;
    margin-top: 4px;
}

.pill {
    border: 1px solid #ffffff55;
    background: #ffffff12;
    border-radius: 30px;
    padding: 8px 13px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .6px;
    color: #dff7fa;
}

.kpi {
    background: white;
    border: 1px solid #e7edf4;
    border-radius: 15px;
    padding: 16px 18px;
    min-height: 112px;
    box-shadow: 0 4px 14px #11233a09;
}

.kpi-label {
    color: #718096;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.kpi-value {
    font-family: Manrope, sans-serif;
    color: #15243a;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
    line-height: 1.2;
}

.kpi-note {
    color: #8a97a8;
    font-size: 11px;
    margin-top: 5px;
}

.kicker {
    color: #0e7490;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 4px 0 5px;
}

.section-title {
    font-family: Manrope, sans-serif;
    color: #132238;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -.5px;
    margin: 0 0 4px;
}

.section-sub {
    color: #718096;
    font-size: 13px;
    margin-bottom: 15px;
}

.insight {
    background: #edf8fa;
    border-left: 4px solid #0e9aa7;
    border-radius: 10px;
    padding: 13px 15px;
    color: #24465a;
    font-size: 13px;
}

.warning {
    background: #fff8e8;
    border-left: 4px solid #e6a23c;
    border-radius: 10px;
    padding: 13px 15px;
    color: #74511c;
    font-size: 12px;
}

.stage-card {
    border: 1px solid #e7edf4;
    background: white;
    border-radius: 12px;
    padding: 12px;
}

.stage-name {
    color: #25364d;
    font-size: 12px;
    font-weight: 700;
}

.stage-value {
    color: #0e7490;
    font-family: Manrope, sans-serif;
    font-size: 21px;
    font-weight: 800;
    margin-top: 4px;
}

.muted {
    color: #8290a2;
    font-size: 11px;
}

.footer {
    color: #8995a5;
    font-size: 11px;
    text-align: center;
    padding: 20px 0 0;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #087f8c;
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA AND MODEL


@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data["Order_Date"] = pd.to_datetime(
        data["Order_Date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Order_Date"]
    )

    return (
        data.sort_values("Order_Date")
        .reset_index(drop=True)
    )


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists() or not MODEL_PATH.exists():

    st.error(
        "Required dataset or ETA model is missing. "
        "Run preprocessing and model training first."
    )

    st.stop()


df = load_data()
model = load_model()


features = [
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


test_start = int(len(df) * 0.80)

test_orders = df.iloc[test_start:].copy()

if test_orders.empty:

    st.error("No records found in the held-out ETA test window.")
    st.stop()


test_orders["Order_Label"] = (
    test_orders["Order_Id"].astype(str)
    + "  ·  "
    + test_orders["Shipping_Mode"].astype(str)
    + "  ·  "
    + test_orders["Order_Country"].astype(str)
)


def simulate(disruptions):

    incoming = 0.0
    rows = []

    for item in STAGES:

        name = item["stage"]

        local = float(
            disruptions.get(name, 0)
        )

        absorbed = min(
            incoming,
            float(item["buffer"])
        )

        residual = incoming - absorbed

        outgoing = residual + local

        rows.append({

            "Stage": name,

            "Buffer (h)": item["buffer"],

            "Incoming delay (h)": round(
                incoming, 1
            ),

            "Absorbed (h)": round(
                absorbed, 1
            ),

            "New disruption (h)": round(
                local, 1
            ),

            "Outgoing delay (h)": round(
                outgoing, 1
            )
        })

        incoming = outgoing

    return pd.DataFrame(rows)

header_html = (
    '<div class="brand">'
    '<div>'
    '<div class="brand-title">'
    '🚢 ShipPulse <span style="color:#69d5df">AI</span>'
    '</div>'
    '<div class="brand-sub">'
    'Shipment intelligence · ETA forecasting · Delay propagation'
    '</div>'
    '</div>'
    '<div class="pill">'
    'LOGISTICS CONTROL TOWER &nbsp; • &nbsp; PROTOTYPE'
    '</div>'
    '</div>'
)

st.markdown(header_html, unsafe_allow_html=True)

with st.sidebar:

    st.markdown("## 🚢 ShipPulse AI")

    st.caption(
        "CONTROL PANEL · Configure a shipment what-if scenario"
    )

    st.divider()

    st.markdown("### 01 · Shipment")

    selected_label = st.selectbox(
        "Select shipment",
        test_orders["Order_Label"].tolist()
    )

    selected = test_orders.loc[
        test_orders["Order_Label"] == selected_label
    ].iloc[0]

    st.divider()

    st.markdown("### 02 · Operational disruptions")

    st.caption(
        "Additional delay introduced at each stage (hours)."
    )

    disruptions = {

        "Order Processing": st.slider(
            "Order processing",
            0, 48, 0, 1
        ),

        "Warehouse Handling": st.slider(
            "Warehouse handling",
            0, 48, 0, 1
        ),

        "Carrier Transit": st.slider(
            "Carrier transit",
            0, 72, 0, 1
        ),

        "Destination Handling": st.slider(
            "Destination handling",
            0, 48, 0, 1
        ),

        "Final-Mile Delivery": st.slider(
            "Final-mile delivery",
            0, 48, 0, 1
        )
    }

    st.divider()

    st.markdown("### Simulation assumptions")

    st.caption("Buffers: 2h · 3h · 4h · 2h · 1h")

    st.caption(
        "Stage shares: 10% · 20% · 40% · 15% · 15%"
    )

    st.caption(
        "Stage values are illustrative, not observed."
    )

x = selected[features].to_frame().T

predicted_days = max(
    0.0,
    float(model.predict(x)[0])
)

scheduled_days = float(
    selected["Scheduled_Days"]
)

order_date = pd.Timestamp(
    selected["Order_Date"]
)

stage_df = simulate(disruptions)

extra_hours = float(
    stage_df.iloc[-1]["Outgoing delay (h)"]
)

revised_days = (
    predicted_days + extra_hours / 24
)

baseline_date = (
    order_date
    + pd.to_timedelta(predicted_days, unit="D")
)

revised_date = (
    order_date
    + pd.to_timedelta(revised_days, unit="D")
)

total_local = float(
    sum(disruptions.values())
)

total_absorbed = float(
    stage_df["Absorbed (h)"].sum()
)

active_count = sum(
    value > 0 for value in disruptions.values()
)
st.markdown(
    '<div class="kicker">LIVE SHIPMENT VIEW</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Shipment overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="section-sub">'
    f'Order <b>{selected["Order_Id"]}</b> · '
    f'{selected["Shipping_Mode"]} · '
    f'{selected["Order_Country"]} · '
    f'Placed {order_date.strftime("%d %b %Y")}'
    f'</div>',
    unsafe_allow_html=True
)

cols = st.columns(4)

cards = [

    (
        "ML predicted duration",
        f"{predicted_days:.2f} days",
        "Random Forest ETA model"
    ),

    (
        "Scheduled duration",
        f"{scheduled_days:.0f} days",
        "Original delivery commitment"
    ),

    (
        "Propagated delay",
        f"{extra_hours:.1f} hours",
        f"{active_count} active disruption(s)"
    ),

    (
        "Revised duration",
        f"{revised_days:.2f} days",
        f"ETA shift: +{extra_hours/24:.2f} days"
    )
]


for col, (label, value, note) in zip(cols, cards):

    with col:

        st.markdown(
            f'<div class="kpi">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


tabs = st.tabs([
    "Overview",
    "Delay propagation",
    "Model & assumptions"
])


with tabs[0]:

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown("#### ETA impact")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(

                x=[
                    "Baseline ETA",
                    "Revised ETA"
                ],

                y=[
                    predicted_days,
                    revised_days
                ],

                marker_color=[
                    "#2d75d7",
                    "#11a6a0"
                ],

                text=[
                    f"{predicted_days:.2f} days",
                    f"{revised_days:.2f} days"
                ],

                textposition="outside",

                hovertemplate=(
                    "%{x}<br>%{y:.2f} days<extra></extra>"
                )
            )
        )

        fig.update_layout(

            height=330,

            margin=dict(
                l=10,
                r=10,
                t=25,
                b=15
            ),

            yaxis_title="Shipping duration (days)",

            showlegend=False,

            paper_bgcolor="white",
            plot_bgcolor="white",

            font=dict(color="#344054"),

            yaxis=dict(
                gridcolor="#edf1f6",
                rangemode="tozero"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown(

            f'<div class="insight">'
            f'<b>Delivery date estimate</b><br>'
            f'Baseline: {baseline_date:%d %b %Y} '
            f'→ Revised: {revised_date:%d %b %Y}'
            f'<br>'
            f'<span class="muted">'
            f'Derived from order date + predicted duration; '
            f'working days, holidays and time zones are not modelled.'
            f'</span>'
            f'</div>',

            unsafe_allow_html=True
        )

    with right:

        st.markdown("#### Disruption snapshot")

        active = [
            (name, hours)
            for name, hours in disruptions.items()
            if hours > 0
        ]

        if not active:

            st.info(
                "No additional disruptions configured. "
                "Shipment remains at its baseline predicted duration."
            )

        for name, hours in active:

            st.markdown(

                f'<div style="display:flex;'
                f'justify-content:space-between;'
                f'padding:9px 0;'
                f'border-bottom:1px solid #edf1f6;">'

                f'<span style="font-size:13px;color:#344054">'
                f'{name}</span>'

                f'<b style="color:#c66b25">{hours}h</b>'

                f'</div>',

                unsafe_allow_html=True
            )

        st.metric(
            "Total local disruption",
            f"{total_local:.1f} h"
        )

        st.metric(
            "Delay absorbed by buffers",
            f"{total_absorbed:.1f} h"
        )

        st.metric(
            "Delay reaching final delivery",
            f"{extra_hours:.1f} h"
        )

    st.markdown(

        '<div class="warning">'
        '<b>Interpretation:</b> The ETA model predicts historical '
        'shipping duration from order-level features. '
        'Disruption controls run a separate what-if simulation; '
        'they do not retrain or alter the ML model.'
        '</div>',

        unsafe_allow_html=True
    )


with tabs[1]:

    st.markdown(
        '<div class="kicker">OPERATIONAL EXPLANATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'How delay moves through the shipment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Each stage absorbs available upstream delay using its '
        'configured buffer, then adds its local disruption.'
        '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=stage_df["Stage"],

            y=stage_df["Incoming delay (h)"],

            mode="lines+markers",

            name="Incoming delay",

            line=dict(
                color="#2d75d7",
                width=3
            ),

            marker=dict(size=9)
        )
    )

    fig.add_trace(
        go.Scatter(

            x=stage_df["Stage"],

            y=stage_df["Outgoing delay (h)"],

            mode="lines+markers",

            name="Outgoing delay",

            line=dict(
                color="#11a6a0",
                width=3
            ),

            marker=dict(size=9)
        )
    )

    fig.update_layout(

        height=370,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=80
        ),

        yaxis_title="Delay (hours)",

        paper_bgcolor="white",
        plot_bgcolor="white",

        font=dict(color="#344054"),

        xaxis=dict(
            categoryorder="array",
            categoryarray=STAGE_ORDER
        ),

        yaxis=dict(
            gridcolor="#edf1f6",
            rangemode="tozero"
        ),

        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("#### Stage-by-stage breakdown")

    st.dataframe(
        stage_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Stage impact")

    stage_cols = st.columns(5)

    for col, (_, row) in zip(
        stage_cols,
        stage_df.iterrows()
    ):

        with col:

            st.markdown(

                f'<div class="stage-card">'
                f'<div class="stage-name">{row["Stage"]}</div>'
                f'<div class="stage-value">'
                f'{row["Outgoing delay (h)"]:.1f}h'
                f'</div>'
                f'<div class="muted">outgoing delay</div>'
                f'<div class="muted" style="margin-top:6px">'
                f'Absorbed: {row["Absorbed (h)"]:.1f}h'
                f'</div>'
                f'</div>',

                unsafe_allow_html=True
            )


with tabs[2]:

    st.markdown(
        '<div class="kicker">MODEL TRANSPARENCY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'What powers this estimate?'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Observed dataset information is separated from the '
        'assumptions used by the propagation simulator.'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        st.markdown("#### Historical ML model")

        st.markdown("""
        - **Algorithm:** Random Forest Regressor
        - **Prediction:** Actual shipping duration in days
        - **Dataset:** DataCo Smart Supply Chain
        - **Evaluation:** Chronological 80/20 split
        - **Test MAE:** approximately 1.00 day
        - **Test RMSE:** approximately 1.27 days
        - **Test R²:** approximately 0.384
        """)

    with b:

        st.markdown("#### Propagation assumptions")

        assumptions = pd.DataFrame([

            {
                "Stage": s["stage"],
                "Allocated share": f'{s["share"]:.0%}',
                "Buffer": f'{s["buffer"]}h'
            }

            for s in STAGES
        ])

        st.dataframe(
            assumptions,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(

        '<div class="warning">'
        '<b>Data limitation:</b> DataCo does not provide observed '
        'timestamps or actual durations for the five intermediate '
        'stages. Stage shares, buffers and disruptions are explicit '
        'simulation assumptions. This is a transparent what-if '
        'mechanism, not a stage-level model trained on observed events.'
        '</div>',

        unsafe_allow_html=True
    )

    st.markdown("#### Feature signals")

    st.write(
        "Permutation importance found scheduled duration and "
        "shipping mode to be the dominant ETA signals. "
        "These are predictive associations in this dataset, "
        "not causal explanations."
    )


st.markdown(

    '<div class="footer">'
    'ShipPulse AI · Research prototype · '
    'Historical predictions and simulated operational scenarios '
    'are clearly separated.'
    '</div>',

    unsafe_allow_html=True
)
