from pathlib import Path
from io import BytesIO
import heapq
import hmac
import os

import pandas as pd
import streamlit as st
import pydeck as pdk

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


MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB per CSV
MAX_ROWS_PER_DATASET = 5_000
MAX_TEXT_LENGTH = 100

AUTH_PASSWORD_KEY = "SUPPLYSHIELD_PASSWORD"
AUTH_ALLOWED_EMAILS_KEY = "SUPPLYSHIELD_ALLOWED_EMAILS"


def get_secret_value(name: str, default: str = "") -> str:
    """Read a value from environment variables first, then Streamlit secrets."""
    env_value = os.environ.get(name, "").strip()
    if env_value:
        return env_value

    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    if value is None:
        return default

    return str(value).strip()


def get_allowed_emails() -> set[str]:
    """Return the optional email allowlist from secrets or environment variables."""
    raw_emails = get_secret_value(AUTH_ALLOWED_EMAILS_KEY)
    return {
        email.strip().lower()
        for email in raw_emails.split(",")
        if email.strip()
    }


def require_authentication():
    """
    Stop the app unless the user enters the correct dashboard password.

    If SUPPLYSHIELD_ALLOWED_EMAILS is set, the entered email must also be
    included in that comma-separated allowlist.
    """
    st.sidebar.header("Access Control")

    if st.session_state.get("authenticated"):
        user_email = st.session_state.get("user_email", "Authorized user")
        st.sidebar.success(f"Signed in as {user_email}")

        if st.sidebar.button("Log out"):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = ""
            st.rerun()

        return

    configured_password = get_secret_value(AUTH_PASSWORD_KEY)
    allowed_emails = get_allowed_emails()

    if not configured_password:
        st.title("🛡️ SupplyShield")
        st.error("Access control is not configured.")
        st.info(
            "Set SUPPLYSHIELD_PASSWORD before running or deploying this app. "
            "The dashboard will stay locked until a password is configured."
        )

        with st.expander("Local setup example"):
            st.code(
                """
mkdir -p .streamlit

cat > .streamlit/secrets.toml <<EOF
SUPPLYSHIELD_PASSWORD = "replace-this-with-a-strong-password"
SUPPLYSHIELD_ALLOWED_EMAILS = "your.email@example.com, teammate@example.com"
EOF
                """.strip(),
                language="bash",
            )

        st.stop()

    st.title("🛡️ SupplyShield")
    st.subheader("Secure Dashboard Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if not submitted:
        st.stop()

    normalized_email = email.strip().lower()

    if not normalized_email:
        st.error("Enter your email address.")
        st.stop()

    if allowed_emails and normalized_email not in allowed_emails:
        st.error("This email is not authorized to access SupplyShield.")
        st.stop()

    if not hmac.compare_digest(password, configured_password):
        st.error("Incorrect password.")
        st.stop()

    st.session_state["authenticated"] = True
    st.session_state["user_email"] = normalized_email
    st.rerun()


def validation_error(message: str):
    """Show a safe, user-friendly validation error and stop the current run."""
    st.error(message)
    st.stop()


def load_csv(uploaded_file, fallback_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load an uploaded CSV or the bundled demo CSV with basic resource limits."""
    source = uploaded_file if uploaded_file is not None else fallback_path

    if uploaded_file is not None and uploaded_file.size > MAX_UPLOAD_BYTES:
        validation_error(
            f"{dataset_name} is too large. Upload a CSV smaller than "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        validation_error(f"{dataset_name} is empty. Upload a CSV with headers and at least one row.")
    except pd.errors.ParserError:
        validation_error(f"{dataset_name} could not be parsed as a valid CSV file.")
    except UnicodeDecodeError:
        validation_error(f"{dataset_name} must be saved as a UTF-8 CSV file.")
    except Exception:
        validation_error(f"{dataset_name} could not be read. Check the file format and try again.")

    if df.empty:
        validation_error(f"{dataset_name} must contain at least one data row.")

    if len(df) > MAX_ROWS_PER_DATASET:
        validation_error(
            f"{dataset_name} has {len(df):,} rows. The maximum allowed is "
            f"{MAX_ROWS_PER_DATASET:,}."
        )

    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        validation_error(f"{dataset_name} is missing required column(s): {', '.join(missing)}")


def clean_text_columns(df: pd.DataFrame, columns: list[str], dataset_name: str):
    for column in columns:
        if df[column].isna().any():
            validation_error(f"{dataset_name}: '{column}' cannot contain blank values.")

        df[column] = df[column].astype(str).str.strip()

        if (df[column] == "").any():
            validation_error(f"{dataset_name}: '{column}' cannot contain blank values.")

        if (df[column].str.len() > MAX_TEXT_LENGTH).any():
            validation_error(
                f"{dataset_name}: values in '{column}' must be {MAX_TEXT_LENGTH} characters or fewer."
            )


def clean_numeric_column(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer_only: bool = False,
):
    converted = pd.to_numeric(df[column], errors="coerce")

    if converted.isna().any():
        validation_error(f"{dataset_name}: '{column}' must contain only numeric values.")

    if minimum is not None and (converted < minimum).any():
        validation_error(f"{dataset_name}: '{column}' must be at least {minimum}.")

    if maximum is not None and (converted > maximum).any():
        validation_error(f"{dataset_name}: '{column}' must be no greater than {maximum}.")

    if integer_only and (converted % 1 != 0).any():
        validation_error(f"{dataset_name}: '{column}' must contain whole numbers only.")

    df[column] = converted.astype("int64") if integer_only else converted.astype("float64")


def validate_unique(df: pd.DataFrame, columns: list[str], dataset_name: str):
    if df.duplicated(subset=columns).any():
        validation_error(
            f"{dataset_name}: duplicate value(s) found for {', '.join(columns)}. "
            "Each record must be unique."
        )


def validate_boolean_column(df: pd.DataFrame, column: str, dataset_name: str):
    allowed = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    normalized = df[column].astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(allowed)

    if invalid.any():
        validation_error(
            f"{dataset_name}: '{column}' must use True/False, Yes/No, or 1/0 values only."
        )

    df[column] = normalized.map(allowed)


def validate_inventory(inventory: pd.DataFrame):
    dataset_name = "Inventory CSV"
    clean_text_columns(inventory, ["location", "item"], dataset_name)
    clean_numeric_column(inventory, "current_stock", dataset_name, minimum=0)
    clean_numeric_column(inventory, "daily_usage", dataset_name, minimum=0.000001)
    clean_numeric_column(inventory, "minimum_safe_stock", dataset_name, minimum=0)
    validate_unique(inventory, ["location", "item"], dataset_name)


def validate_shipments(shipments: pd.DataFrame, inventory: pd.DataFrame):
    dataset_name = "Shipments CSV"

    clean_text_columns(
        shipments,
        ["shipment_id", "item", "origin", "destination", "requested_destination"],
        dataset_name,
    )

    clean_numeric_column(shipments, "quantity", dataset_name, minimum=0.000001)
    clean_numeric_column(shipments, "typical_quantity", dataset_name, minimum=0.000001)
    clean_numeric_column(shipments, "expected_days", dataset_name, minimum=0, integer_only=True)
    clean_numeric_column(shipments, "actual_days", dataset_name, minimum=0, integer_only=True)
    clean_numeric_column(shipments, "route_changes", dataset_name, minimum=0, integer_only=True)

    validate_unique(shipments, ["shipment_id"], dataset_name)

    valid_items = set(inventory["item"])
    unknown_items = sorted(set(shipments["item"]) - valid_items)

    if unknown_items:
        validation_error(
            f"{dataset_name}: unknown item(s) not found in inventory.csv: {', '.join(unknown_items)}"
        )

    if "shipment_date" in shipments.columns:
        parsed_dates = pd.to_datetime(shipments["shipment_date"], errors="coerce")

        if parsed_dates.isna().any():
            validation_error(f"{dataset_name}: 'shipment_date' contains an invalid date.")

        shipments["shipment_date"] = parsed_dates.dt.strftime("%Y-%m-%d")


def validate_requests(requests: pd.DataFrame, inventory: pd.DataFrame):
    dataset_name = "Delivery requests CSV"

    clean_text_columns(requests, ["request_id", "location", "item"], dataset_name)
    clean_numeric_column(requests, "requested_quantity", dataset_name, minimum=0.000001)
    clean_numeric_column(requests, "mission_importance", dataset_name, minimum=1, maximum=10, integer_only=True)
    clean_numeric_column(requests, "people_affected", dataset_name, minimum=0, integer_only=True)
    validate_boolean_column(requests, "incoming_shipment_delayed", dataset_name)
    validate_unique(requests, ["request_id"], dataset_name)

    inventory_pairs = set(zip(inventory["location"], inventory["item"]))
    request_pairs = set(zip(requests["location"], requests["item"]))
    unknown_pairs = sorted(request_pairs - inventory_pairs)

    if unknown_pairs:
        formatted = ", ".join(f"{location} / {item}" for location, item in unknown_pairs)
        validation_error(
            f"{dataset_name}: request location/item pair(s) not found in inventory.csv: {formatted}"
        )


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

    inventory = load_csv(uploaded_inventory, DATA_DIR / "inventory.csv", "Inventory CSV")
    shipments = load_csv(uploaded_shipments, DATA_DIR / "shipments.csv", "Shipments CSV")
    requests = load_csv(uploaded_requests, DATA_DIR / "delivery_requests.csv", "Delivery requests CSV")

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

    validate_inventory(inventory)
    validate_shipments(shipments, inventory)
    validate_requests(requests, inventory)

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
            inventory_risk["inventory_risk"].isin(["Critical", "High"]),
            "location",
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
        inventory_risk,
        shipment_risk,
        delivery_priorities,
        origin_reliability,
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
    st.write("Visualize shipment paths and inspect suspicious routes.")

    selected_id = st.selectbox(
        "Choose a shipment",
        shipment_risk["shipment_id"].tolist(),
    )

    selected = shipment_risk[
        shipment_risk["shipment_id"] == selected_id
    ].iloc[0]

    origin = str(selected["origin"]).strip()
    destination = str(selected["destination"]).strip()
    requested_destination = str(selected["requested_destination"]).strip()

    missing_locations = []

    if origin not in LOCATION_COORDS:
        missing_locations.append(origin)

    if destination not in LOCATION_COORDS:
        missing_locations.append(destination)

    if requested_destination not in LOCATION_COORDS:
        missing_locations.append(requested_destination)

    if missing_locations:
        st.warning(
            "These location names are missing from LOCATION_COORDS: "
            + ", ".join(sorted(set(missing_locations)))
        )

        st.write("Supported locations:", list(LOCATION_COORDS.keys()))
        return

    origin_lat, origin_lon = LOCATION_COORDS[origin]
    destination_lat, destination_lon = LOCATION_COORDS[destination]
    requested_lat, requested_lon = LOCATION_COORDS[requested_destination]

    map_rows = [
        {
            "lat": origin_lat,
            "lon": origin_lon,
            "location": origin,
            "type": "Origin",
        },
        {
            "lat": destination_lat,
            "lon": destination_lon,
            "location": destination,
            "type": "Actual destination",
        },
    ]

    if requested_destination != destination:
        map_rows.append(
            {
                "lat": requested_lat,
                "lon": requested_lon,
                "location": requested_destination,
                "type": "Originally requested destination",
            }
        )

    path_rows = [
        {
            "path_type": "Actual shipment route",
            "path": [
                [origin_lon, origin_lat],
                [destination_lon, destination_lat],
            ],
            "color": [255, 70, 70],
        }
    ]

    if requested_destination != destination:
        path_rows.append(
            {
                "path_type": "Originally requested route",
                "path": [
                    [origin_lon, origin_lat],
                    [requested_lon, requested_lat],
                ],
                "color": [70, 170, 255],
            }
        )

    map_df = pd.DataFrame(map_rows)
    path_df = pd.DataFrame(path_rows)

    layers = [
        pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_color="color",
            get_width=8,
            width_min_pixels=6,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=70000,
            get_fill_color=[255, 165, 0],
            pickable=True,
        ),
    ]

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(map_df["lat"].mean()),
                longitude=float(map_df["lon"].mean()),
                zoom=3.5,
                pitch=0,
            ),
            layers=layers,
            tooltip={
                "html": "<b>{location}</b><br />{type}<br />{path_type}"
            },
        ),
        use_container_width=True,
    )

    st.caption("Red line: actual route. Blue line: originally requested route.")

    st.dataframe(
        map_df,
        use_container_width=True,
        hide_index=True,
    )

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
    require_authentication()

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
            inventory_risk,
            shipment_risk,
            delivery_priorities,
            origin_reliability,
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