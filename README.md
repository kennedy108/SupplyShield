# SupplyShield MVP

SupplyShield is a Streamlit dashboard for mission supply-chain monitoring.

The MVP has four pages:

1. **Command Center** — summary metrics, charts, and top recommendations
2. **Inventory Risk Monitor** — predicts when supplies will cross minimum safe-stock thresholds
3. **Shipment Anomaly Detector** — flags delays, unusual quantities, route changes, and destination mismatches
4. **Emergency Delivery Prioritizer** — ranks requests using Python's `heapq` priority queue

## Setup

Open a terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Files

```text
supplyshield_mvp/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── inventory.csv
    ├── shipments.csv
    └── delivery_requests.csv
```

## Next Improvements

After the MVP runs, useful stretch features include CSV uploads, supplier reliability scores, a shipment route map, and a simulated cyberattack that manipulates shipment records.
