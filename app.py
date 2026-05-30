from pathlib import Path
from io import BytesIO
import heapq

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SupplyShield",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RISK_ORDER = ["Critical", "High", "Medium", "Low"]

LOCATION_COORDS = {
    "Depot North": (39.7392, -104.9903),
    "Depot South": (32.7767, -96.7970),
    "Depot East": (38.9072, -77.0369),
    "Depot West": (34.0522, -118.2437),
    "Base Alpha": (36.1699, -115.1398),
    "Base Bravo": (33.4484, -112.0740),
    "Base Echo": (35.0844, -106.6504),
    "Base Delta": (29.7604, -95.3698),
}


def load_csv(uploaded_file, fallback_path: Path) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(fallback_path)


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        st.error(f"{dataset_name} is missing required column(s): {', '.join(missing)}")
        st.stop()


def load_data():
    st.sidebar.header("Data Source")
    st.sidebar.caption("Upload custom CSV files or keep using the included demo dataset.")

    uploaded_inventory = st.sidebar.file_uploader(
        "Upload inventory.csv", type=["csv"], key="inventory_upload"
    )
    uploaded_shipments = st.sidebar.file_uploader(
        "Upload shipments.csv", type=["csv"], key="shipments_upload"
    )
    uploaded_requests = st.sidebar.file_uploader(
        "Upload delivery_requests.csv", type=["csv"], key="requests_upload"
    )

    inventory = load_csv(uploaded_inventory, DATA_DIR / "inventory.csv")
    shipments = load_csv(uploaded_shipments, DATA_DIR / "shipments.csv")
    requests = load_csv(uploaded_requests, DATA_DIR / "delivery_requests.csv")

    validate_columns(
        inventory,
        ["location", "item", "current_stock", "daily_usage", "minimum_safe_stock"],
        "Inventory CSV",
    )
    validate_columns(
        shipments,
        [
            "shipment_id",
            "item",
            "origin",
            "destination",
            "requested_destination",
            "quantity",
            "typical_quantity",
            "expected_days",
            "actual_days",
            "route_changes",
        ],
        "Shipments CSV",
    )
    validate_columns(
        requests,
        [
            "request_id",
            "location",
            "item",
            "requested_quantity",
            "mission_importance",
            "people_affected",
            "incoming_shipment_delayed",
        ],
        "Delivery requests CSV",
    )

    if "shipment_date" not in shipments.columns:
        shipments["shipment_date"] = pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=len(shipments),
            freq="D",
        ).strftime("%Y-%m-%d")

    source_label = "Demo data"
    if uploaded_inventory or uploaded_shipments or uploaded_requests:
        source_label = "Uploaded CSV data"

    st.sidebar.success(f"Current source: {source_label}")
    return inventory, shipments, requests


def classify_inventory_risk(days_until_unsafe: float) -> str:
    if pd.isna(days_until_unsafe):
        return "Low"
    if days_until_unsafe <= 1:
        return "Critical"
    if days_until_unsafe <= 3:
        return "High"
    if days_until_unsafe <= 7:
        return "Medium"
    return "Low"


def calculate_inventory_risk(inventory: pd.DataFrame) -> pd.DataFrame:
    df = inventory.copy()
    safe_usage = df["daily_usage"].replace(0, pd.NA)
    df["days_until_stockout"] = (df["current_stock"] / safe_usage).round(1)
    df["days_until_unsafe"] = (
        (df["current_stock"] - df["minimum_safe_stock"]) / safe_usage
    ).round(1)
    df["inventory_risk"] = df["days_until_unsafe"].apply(classify_inventory_risk)
    return df


def classify_shipment_risk(score: int) -> str:
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def score_single_shipment(row: pd.Series) -> dict:
    delay_days = max(int(row["actual_days"]) - int(row["expected_days"]), 0)
    quantity_ratio = round(float(row["quantity"]) / float(row["typical_quantity"]), 2)

    score = 0
    reasons = []

    if delay_days > 3:
        score += 30
        reasons.append("Major delay")
    elif delay_days > 0:
        score += 10
        reasons.append("Minor delay")

    if quantity_ratio > 2:
        score += 20
        reasons.append("Quantity exceeds 2x typical amount")
    elif quantity_ratio > 1.5:
        score += 10
        reasons.append("Quantity exceeds 1.5x typical amount")

    if int(row["route_changes"]) > 0:
        route_points = min(int(row["route_changes"]) * 15, 45)
        score += route_points
        reasons.append(f"{int(row['route_changes'])} route change(s)")

    if str(row["destination"]).strip().lower() != str(row["requested_destination"]).strip().lower():
        score += 40
        reasons.append("Destination mismatch")

    score = min(score, 100)
    return {
        "delay_days": delay_days,
        "quantity_ratio": quantity_ratio,
        "shipment_risk_score": score,
        "shipment_risk": classify_shipment_risk(score),
        "flag_reason": "; ".join(reasons) if reasons else "No unusual activity detected",
    }


def score_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    df = shipments.copy()
    scored = df.apply(lambda row: pd.Series(score_single_shipment(row)), axis=1)
    return pd.concat([df, scored], axis=1)


def urgency_points(days_until_unsafe: float) -> int:
    if days_until_unsafe <= 1:
        return 50
    if days_until_unsafe <= 3:
        return 35
    if days_until_unsafe <= 7:
        return 20
    return 5


def prioritize_deliveries(
    requests: pd.DataFrame,
    inventory_risk: pd.DataFrame,
) -> pd.DataFrame:
    merged = requests.merge(
        inventory_risk[
            ["location", "item", "days_until_unsafe", "days_until_stockout", "inventory_risk"]
        ],
        on=["location", "item"],
        how="left",
    )

    recommendations = []
    priority_queue = []

    for _, row in merged.iterrows():
        days_until_unsafe = float(row["days_until_unsafe"])
        score = urgency_points(days_until_unsafe)
        score += int(row["mission_importance"]) * 4
        score += min(int(row["people_affected"]) // 50, 15)

        reasons = [
            f"{days_until_unsafe:.1f} day(s) until safety buffer is crossed",
            f"mission importance {int(row['mission_importance'])}/10",
            f"{int(row['people_affected'])} people affected",
        ]

        delayed_raw = row["incoming_shipment_delayed"]
        delayed = (
            str(delayed_raw).strip().lower() in {"true", "1", "yes", "y"}
            if not isinstance(delayed_raw, bool)
            else delayed_raw
        )

        if delayed:
            score += 15
            reasons.append("incoming shipment is delayed")

        heapq.heappush(
            priority_queue,
            (
                -score,
                row["request_id"],
                {
                    "request_id": row["request_id"],
                    "location": row["location"],
                    "item": row["item"],
                    "requested_quantity": row["requested_quantity"],
                    "priority_score": score,
                    "inventory_risk": row["inventory_risk"],
                    "days_until_unsafe": days_until_unsafe,
                    "reason": "; ".join(reasons),
                },
            ),
        )

    rank = 1
    while priority_queue:
        _, _, recommendation = heapq.heappop(priority_queue)
        recommendation["priority_rank"] = rank
        recommendations.append(recommendation)
        rank += 1

    return pd.DataFrame(recommendations)[
        [
            "priority_rank",
            "request_id",
            "location",
            "item",
            "requested_quantity",
            "priority_score",
            "inventory_risk",
            "days_until_unsafe",
            "reason",
        ]
    ]


def calculate_origin_reliability(shipment_risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, group in shipment_risk.groupby("origin"):
        shipment_count = len(group)
        delayed_rate = (group["delay_days"] > 0).mean()
        anomaly_rate = group["shipment_risk"].isin(["Critical", "High"]).mean()
        avg_risk_score = group["shipment_risk_score"].mean()

        reliability_score = round(
            max(0, 100 - delayed_rate * 30 - anomaly_rate * 45 - avg_risk_score * 0.25),
            1,
        )

        if reliability_score >= 80:
            rating = "Reliable"
        elif reliability_score >= 60:
            rating = "Monitor"
        else:
            rating = "High Risk"

        rows.append(
            {
                "origin": origin,
                "shipments": shipment_count,
                "delayed_rate_percent": round(delayed_rate * 100, 1),
                "high_risk_rate_percent": round(anomaly_rate * 100, 1),
                "average_risk_score": round(avg_risk_score, 1),
                "reliability_score": reliability_score,
                "reliability_rating": rating,
            }
        )

    return pd.DataFrame(rows).sort_values("reliability_score")


def build_alert_report(
    inventory_risk: pd.DataFrame,
    shipment_risk: pd.DataFrame,
    delivery_priorities: pd.DataFrame,
    origin_reliability: pd.DataFrame,
) -> bytes:
    inventory_alerts = inventory_risk[
        inventory_risk["inventory_risk"].isin(["Critical", "High"])
    ].copy()
    shipment_alerts = shipment_risk[
        shipment_risk["shipment_risk"].isin(["Critical", "High"])
    ].copy()

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inventory_alerts.to_excel(writer, sheet_name="Inventory Alerts", index=False)
        shipment_alerts.to_excel(writer, sheet_name="Shipment Alerts", index=False)
        delivery_priorities.to_excel(writer, sheet_name="Delivery Priorities", index=False)
        origin_reliability.to_excel(writer, sheet_name="Origin Reliability", index=False)

    return output.getvalue()


def show_command_center(
    inventory_risk: pd.DataFrame,
    shipment_risk: pd.DataFrame,
    delivery_priorities: pd.DataFrame,
    origin_reliability: pd.DataFrame,
):
    st.title("🛡️ SupplyShield")
    st.caption("Mission supply risk monitoring, anomaly detection, and emergency response")

    critical_inventory = int(inventory_risk["inventory_risk"].isin(["Critical", "High"]).sum())
    suspicious_shipments = int(shipment_risk["shipment_risk"].isin(["Critical", "High"]).sum())
    delayed_shipments = int((shipment_risk["delay_days"] > 0).sum())
    locations_at_risk = int(
        inventory_risk.loc[
            inventory_risk["inventory_risk"].isin(["Critical", "High"]), "location"
        ].nunique()
    )

    top_delivery = delivery_priorities.iloc[0]
    top_delivery_text = f"{top_delivery['item']} → {top_delivery['location']}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Critical / High Inventory Alerts", critical_inventory)
    col2.metric("Suspicious Shipments", suspicious_shipments)
    col3.metric("Delayed Shipments", delayed_shipments)
    col4.metric("Locations at Risk", locations_at_risk)

    st.info(f"Highest-priority delivery: **{top_delivery_text}**")

    left, right = st.columns(2)

    with left:
        st.subheader("Supplies Closest to Unsafe Levels")
        chart_data = (
            inventory_risk.sort_values("days_until_unsafe")
            .head(8)
            .assign(supply=lambda x: x["location"] + " — " + x["item"])
            .set_index("supply")[["days_until_unsafe"]]
        )
        st.bar_chart(chart_data)

    with right:
        st.subheader("Shipment Risk Distribution")
        risk_counts = (
            shipment_risk["shipment_risk"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .rename_axis("shipment_risk")
            .to_frame("shipments")
        )
        st.bar_chart(risk_counts)

    st.subheader("Top Emergency Delivery Recommendations")
    st.dataframe(delivery_priorities.head(5), use_container_width=True, hide_index=True)

    report_bytes = build_alert_report(
        inventory_risk, shipment_risk, delivery_priorities, origin_reliability
    )
    st.download_button(
        label="Download Alert Report",
        data=report_bytes,
        file_name="supplyshield_alert_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def show_inventory_monitor(inventory_risk: pd.DataFrame):
    st.header("Inventory Risk Monitor")
    st.write("Estimate when each location will fall below its minimum safe stock level.")

    locations = sorted(inventory_risk["location"].unique().tolist())
    items = sorted(inventory_risk["item"].unique().tolist())

    selected_locations = st.multiselect("Locations", locations, default=locations)
    selected_items = st.multiselect("Items", items, default=items)
    selected_risks = st.multiselect("Risk levels", RISK_ORDER, default=RISK_ORDER)

    filtered = inventory_risk[
        inventory_risk["location"].isin(selected_locations)
        & inventory_risk["item"].isin(selected_items)
        & inventory_risk["inventory_risk"].isin(selected_risks)
    ].copy()

    risk_order = pd.CategoricalDtype(RISK_ORDER, ordered=True)
    filtered["inventory_risk"] = filtered["inventory_risk"].astype(risk_order)
    filtered = filtered.sort_values(["inventory_risk", "days_until_unsafe"])

    st.dataframe(filtered, use_container_width=True, hide_index=True)


def show_shipment_detector(shipment_risk: pd.DataFrame):
    st.header("Shipment Anomaly Detector")
    st.write("Flag shipments with delays, unusual quantities, route changes, or destination mismatches.")

    selected_risk = st.selectbox("Filter by shipment risk", ["All"] + RISK_ORDER)
    filtered = shipment_risk.copy()

    if selected_risk != "All":
        filtered = filtered[filtered["shipment_risk"] == selected_risk]

    filtered = filtered.sort_values("shipment_risk_score", ascending=False)
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Explain a Shipment Flag")
    shipment_options = filtered["shipment_id"].tolist()
    if shipment_options:
        selected_id = st.selectbox("Choose a shipment", shipment_options)
        selected = filtered[filtered["shipment_id"] == selected_id].iloc[0]
        st.write(f"**Risk score:** {selected['shipment_risk_score']}/100")
        st.write(f"**Risk level:** {selected['shipment_risk']}")
        st.write(f"**Explanation:** {selected['flag_reason']}")
    else:
        st.info("No shipments match the selected filter.")


def show_delivery_prioritizer(delivery_priorities: pd.DataFrame):
    st.header("Emergency Delivery Prioritizer")
    st.write("Rank requests using a priority queue. A higher score means the delivery should be handled sooner.")

    st.dataframe(delivery_priorities, use_container_width=True, hide_index=True)

    highest_priority = delivery_priorities.iloc[0]
    st.success(
        "Recommended next action: "
        f"send **{highest_priority['requested_quantity']} units of "
        f"{highest_priority['item']}** to **{highest_priority['location']}**."
    )


def show_route_intelligence(shipment_risk: pd.DataFrame):
    st.header("Route Intelligence Map")
    st.write("Visualize shipment endpoints and inspect suspicious routes.")

    selected_id = st.selectbox("Choose a shipment", shipment_risk["shipment_id"].tolist())
    selected = shipment_risk[shipment_risk["shipment_id"] == selected_id].iloc[0]

    origin = str(selected["origin"])
    destination = str(selected["destination"])
    requested_destination = str(selected["requested_destination"])

    map_rows = []
    if origin in LOCATION_COORDS:
        lat, lon = LOCATION_COORDS[origin]
        map_rows.append({"lat": lat, "lon": lon, "location": origin, "type": "Origin"})
    if destination in LOCATION_COORDS:
        lat, lon = LOCATION_COORDS[destination]
        map_rows.append({"lat": lat, "lon": lon, "location": destination, "type": "Destination"})
    if requested_destination != destination and requested_destination in LOCATION_COORDS:
        lat, lon = LOCATION_COORDS[requested_destination]
        map_rows.append(
            {
                "lat": lat,
                "lon": lon,
                "location": requested_destination,
                "type": "Requested destination",
            }
        )

    if map_rows:
        map_df = pd.DataFrame(map_rows)
        st.map(map_df[["lat", "lon"]], use_container_width=True)
        st.dataframe(map_df, use_container_width=True, hide_index=True)
    else:
        st.warning("This shipment uses locations that are not in the demo coordinate map.")

    left, right = st.columns(2)
    with left:
        st.metric("Risk Score", f"{selected['shipment_risk_score']}/100")
        st.write(f"**Origin:** {origin}")
        st.write(f"**Destination:** {destination}")
        st.write(f"**Requested destination:** {requested_destination}")
    with right:
        st.metric("Route Changes", int(selected["route_changes"]))
        st.write(f"**Delay:** {int(selected['delay_days'])} day(s)")
        st.write(f"**Explanation:** {selected['flag_reason']}")


def show_reliability_monitor(origin_reliability: pd.DataFrame):
    st.header("Origin Reliability Monitor")
    st.write("Compare shipping origins based on delays, anomaly rates, and average risk scores.")

    chart = origin_reliability.set_index("origin")[["reliability_score"]]
    st.bar_chart(chart)
    st.dataframe(origin_reliability, use_container_width=True, hide_index=True)

    riskiest = origin_reliability.iloc[0]
    st.warning(
        f"Highest-risk origin: **{riskiest['origin']}** "
        f"with a reliability score of **{riskiest['reliability_score']}/100**."
    )


def show_historical_trends(shipment_risk: pd.DataFrame):
    st.header("Historical Risk Trends")
    st.write("Track how shipment anomaly scores change over time.")

    trend = shipment_risk.copy()
    trend["shipment_date"] = pd.to_datetime(trend["shipment_date"], errors="coerce")
    trend = trend.dropna(subset=["shipment_date"]).sort_values("shipment_date")

    if trend.empty:
        st.info("No valid shipment dates are available.")
        return

    daily = trend.groupby("shipment_date", as_index=True).agg(
        average_risk_score=("shipment_risk_score", "mean"),
        delayed_shipments=("delay_days", lambda values: int((values > 0).sum())),
    )

    st.subheader("Average Shipment Risk Score")
    st.line_chart(daily[["average_risk_score"]])

    st.subheader("Delayed Shipments Over Time")
    st.bar_chart(daily[["delayed_shipments"]])


def show_cyber_simulator(shipment_risk: pd.DataFrame):
    st.header("Cyber Manipulation Simulator")
    st.write(
        "Simulate a tampered logistics record and compare the shipment's original risk score "
        "against its modified score."
    )

    selected_id = st.selectbox("Choose a shipment to modify", shipment_risk["shipment_id"].tolist())
    original = shipment_risk[shipment_risk["shipment_id"] == selected_id].iloc[0].copy()

    st.subheader("Simulated Record Changes")

    destination_options = sorted(
        set(shipment_risk["destination"].tolist() + shipment_risk["requested_destination"].tolist())
    )

    new_destination = st.selectbox(
        "Destination",
        destination_options,
        index=destination_options.index(str(original["destination"])),
    )
    quantity_multiplier = st.slider("Quantity multiplier", 0.5, 4.0, 1.0, 0.1)
    added_route_changes = st.slider("Additional route changes", 0, 4, 0)
    added_delay_days = st.slider("Additional delay days", 0, 7, 0)

    simulated = original.copy()
    simulated["destination"] = new_destination
    simulated["quantity"] = round(float(original["quantity"]) * quantity_multiplier, 1)
    simulated["route_changes"] = int(original["route_changes"]) + added_route_changes
    simulated["actual_days"] = int(original["actual_days"]) + added_delay_days

    simulated_result = score_single_shipment(simulated)
    original_score = int(original["shipment_risk_score"])
    simulated_score = int(simulated_result["shipment_risk_score"])
    score_change = simulated_score - original_score

    left, right, third = st.columns(3)
    left.metric("Original Risk Score", f"{original_score}/100")
    right.metric("Simulated Risk Score", f"{simulated_score}/100", delta=score_change)
    third.metric("Simulated Risk Level", simulated_result["shipment_risk"])

    st.write(f"**Original explanation:** {original['flag_reason']}")
    st.write(f"**Simulated explanation:** {simulated_result['flag_reason']}")

    if simulated_score > original_score:
        st.error("Potential manipulation detected: the modified record increases operational risk.")
    elif simulated_score == original_score:
        st.info("The simulated changes do not alter the calculated risk score.")
    else:
        st.success("The simulated changes reduce the calculated risk score.")


def main():
    inventory, shipments, requests = load_data()
    inventory_risk = calculate_inventory_risk(inventory)
    shipment_risk = score_shipments(shipments)
    delivery_priorities = prioritize_deliveries(requests, inventory_risk)
    origin_reliability = calculate_origin_reliability(shipment_risk)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Command Center",
            "Inventory Risk Monitor",
            "Shipment Anomaly Detector",
            "Emergency Delivery Prioritizer",
            "Route Intelligence Map",
            "Origin Reliability Monitor",
            "Historical Risk Trends",
            "Cyber Manipulation Simulator",
        ],
    )

    if page == "Command Center":
        show_command_center(
            inventory_risk, shipment_risk, delivery_priorities, origin_reliability
        )
    elif page == "Inventory Risk Monitor":
        show_inventory_monitor(inventory_risk)
    elif page == "Shipment Anomaly Detector":
        show_shipment_detector(shipment_risk)
    elif page == "Emergency Delivery Prioritizer":
        show_delivery_prioritizer(delivery_priorities)
    elif page == "Route Intelligence Map":
        show_route_intelligence(shipment_risk)
    elif page == "Origin Reliability Monitor":
        show_reliability_monitor(origin_reliability)
    elif page == "Historical Risk Trends":
        show_historical_trends(shipment_risk)
    else:
        show_cyber_simulator(shipment_risk)


if __name__ == "__main__":
    main()
