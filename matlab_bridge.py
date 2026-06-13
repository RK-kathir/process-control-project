# ════════════════════════════════════════════════════════════════════════
#  matlab_bridge.py
#  Python companion script that connects Simulink/MATLAB to TUNING BOT
#  via Socket.IO. Handles:
#    1. Autonomous tuning requests (FOPDT / SOPDT / 3rd-order)
#    2. Live telemetry streaming to the dashboard
#    3. Disturbance-sweep logging for ANFIS training (Kp/Ki vs Disturbance)
#
#  Run:  python matlab_bridge.py
#  Requires: pip install "python-socketio[client]"
# ════════════════════════════════════════════════════════════════════════
import socketio
import time
import json
import os

# ── CONFIG ─────────────────────────────────────────────────────────────
SERVER_URL = "https://process-control-project.onrender.com"   # ← change if needed
GAINS_FILE = "latest_gains.json"   # written every time new Kc/Ti arrive

sio = socketio.Client()
latest_gains = {"kc": 1.0, "ti": 10.0, "rule": None}


# ── EVENT HANDLERS ─────────────────────────────────────────────────────
@sio.on("connect")
def on_connect():
    print("[Bridge] Connected to Tuning Bot server.")


@sio.on("disconnect")
def on_disconnect():
    print("[Bridge] Disconnected from server.")


@sio.on("status")
def on_status(data):
    print(f"[Bridge] Status: {data.get('message')}")


@sio.on("tune_response")
def on_tune_response(data):
    if data.get("status") == "error":
        print(f"[Bridge] Tune ERROR: {data.get('message')}")
        return

    latest_gains["kc"]   = data["kc"]
    latest_gains["ti"]   = data["ti"]
    latest_gains["rule"] = data["rule"]

    print(f"[Bridge] New gains → Kc={data['kc']}  Ti={data['ti']}  Rule={data['rule']}")
    print(f"[Bridge]   Operator: {data.get('operator_decision','')}")
    if "reduction_note" in data:
        print(f"[Bridge]   Reduction: {data['reduction_note']}")

    if "anfis_row" in data:
        row = data["anfis_row"]
        print(f"[Bridge]   ANFIS point logged: D={row['disturbance']} "
              f"Kp={row['kp']} Ki={row['ki']} (total={data.get('anfis_total')})")

    # Write gains to a local JSON file MATLAB can read with jsondecode()
    with open(GAINS_FILE, "w") as f:
        json.dump(latest_gains, f)


@sio.on("anfis_data_update")
def on_anfis_update(data):
    if data.get("reset"):
        print("[Bridge] ANFIS dataset was reset on the server.")
    elif data.get("row"):
        print(f"[Bridge] ANFIS dataset now has {data['total_points']} points.")


# ── TUNE REQUEST HELPERS ───────────────────────────────────────────────
def tune_fopdt(km, tm, taum, disturbance=None):
    """First-order model. Pass `disturbance` to log a (D, Kp, Ki) point."""
    payload = {"order": 1, "km": km, "tm": tm, "taum": taum}
    if disturbance is not None:
        payload["disturbance"] = disturbance
    sio.emit("tune_request", payload)


def tune_sopdt(km, tm1, tm2, taum, zeta=1.0, disturbance=None):
    """Second-order model — server applies Skogestad's Half-Rule."""
    payload = {"order": 2, "km": km, "tm1": tm1, "tm2": tm2,
               "taum": taum, "zeta": zeta}
    if disturbance is not None:
        payload["disturbance"] = disturbance
    sio.emit("tune_request", payload)


def tune_third_order(km, tm1, tm2, tm3, taum, disturbance=None):
    """Third-order model — server applies Skogestad's Half-Rule."""
    payload = {"order": 3, "km": km, "tm1": tm1, "tm2": tm2,
               "tm3": tm3, "taum": taum}
    if disturbance is not None:
        payload["disturbance"] = disturbance
    sio.emit("tune_request", payload)


def stream_telemetry(t, pv, sp, mv):
    """Push one (or a batch of) live points to the dashboard's PV graph."""
    def _wrap(x):
        return x if isinstance(x, list) else [x]
    sio.emit("telemetry", {
        "t":  _wrap(t),  "pv": _wrap(pv),
        "sp": _wrap(sp), "mv": _wrap(mv)
    })


def download_anfis_csv(filename="anfis_training_data.csv"):
    """Pull the full ANFIS dataset (D, Kp, Ki, ...) from the server as CSV."""
    import urllib.request
    url = f"{SERVER_URL}/api/anfis-data.csv"
    urllib.request.urlretrieve(url, filename)
    print(f"[Bridge] ANFIS dataset saved to {filename}")
    return filename


def reset_anfis_dataset():
    """Clear the server-side ANFIS dataset before a fresh sweep."""
    import urllib.request
    req = urllib.request.Request(f"{SERVER_URL}/api/anfis-reset", method="POST")
    urllib.request.urlopen(req)
    print("[Bridge] ANFIS dataset reset.")


# ── MAIN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sio.connect(SERVER_URL, transports=["websocket"])

    # ── Example 1: single FOPDT tune ────────────────────────────────────
    tune_fopdt(km=2.0, tm=10.0, taum=2.0)
    time.sleep(1)

    # ── Example 2: SOPDT tune (auto-reduced via Half-Rule) ──────────────
    tune_sopdt(km=2.0, tm1=10.0, tm2=3.0, taum=1.0, zeta=1.0)
    time.sleep(1)

    # ── Example 3: disturbance sweep → builds ANFIS dataset ─────────────
    #  Replace `identify_tf_at_disturbance` with your real Simulink/online
    #  identification routine. Each call logs one (D, Kp, Ki) row.
    print("\n[Bridge] Running disturbance sweep for ANFIS dataset...")

    def identify_tf_at_disturbance(D):
        """
        Placeholder identification — replace with your real model.
        Returns (km, tm, taum) for the plant at disturbance level D.
        """
        km   = 2.0 - 0.5 * D          # gain drops as disturbance rises
        tm   = 10.0 + 2.0 * D         # process slows slightly
        taum = 1.0 + 1.5 * D          # dead time grows
        return km, tm, taum

    for D in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        km, tm, taum = identify_tf_at_disturbance(D)
        tune_fopdt(km, tm, taum, disturbance=D)
        time.sleep(0.4)   # let the server respond + broadcast before next emit

    time.sleep(1)
    print("\n[Bridge] Sweep complete.")

    # ── Download the resulting dataset for MATLAB ───────────────────────
    download_anfis_csv("anfis_training_data.csv")

    # ── Example 4: continuous telemetry stream (uncomment to use) ───────
    # for i in range(200):
    #     t  = i * 0.1
    #     pv = 1 - 0.9 ** i
    #     sp = 1.0
    #     mv = latest_gains["kc"] * (sp - pv)
    #     stream_telemetry(t, pv, sp, mv)
    #     time.sleep(0.05)

    print(f"\n[Bridge] Latest gains: Kc={latest_gains['kc']}  Ti={latest_gains['ti']}  "
          f"Rule={latest_gains['rule']}")
    print(f"[Bridge] Written to {GAINS_FILE} — read this from MATLAB with jsondecode(fileread(...))")

    sio.disconnect()
