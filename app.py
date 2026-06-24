import eventlet
eventlet.monkey_patch()
import os, re, json, pickle, math, csv, traceback
from datetime import datetime, timezone
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from tf_parser import parse_transfer_function, TFParseError

# 1. Initialize the Flask App FIRST
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'tuningbot-secret')

# 2. Add CORS so GitHub can talk to Render
CORS(app)

# 3. Initialize Socket.IO with relaxed timeouts to survive MATLAB calculations
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    ping_timeout=60,    # Give MATLAB 60 seconds before assuming it crashed
    ping_interval=25    # Check connection every 25 seconds
)
socketio_app = socketio  # alias for gunicorn

# ── Gemini ─────────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# ── Load assets ────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_dir, 'tuning_rules.json')) as f:
        rules_db = json.load(f)
    # Filter out comment keys
    rules_db = {k: v for k, v in rules_db.items() if not k.startswith('_')}
    with open(os.path.join(current_dir, 'ai_brain.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    print(f"SUCCESS: {len(rules_db)} rules loaded, AI Brain ready.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}


# ══════════════════════════════════════════════════════════════════════════
#  ANFIS TRAINING-DATA LOG  (Disturbance → Kp, Ki)
# ══════════════════════════════════════════════════════════════════════════
ANFIS_DATA_PATH = os.path.join(current_dir, 'anfis_training_data.csv')
ANFIS_FIELDS = ['timestamp', 'disturbance', 'km', 'tm', 'taum',
                'kc', 'ti', 'kp', 'ki', 'rule', 'order']

anfis_data = []  # in-memory mirror of the CSV, list of dicts

def _load_anfis_data():
    global anfis_data
    anfis_data = []
    if os.path.exists(ANFIS_DATA_PATH):
        try:
            with open(ANFIS_DATA_PATH, newline='') as f:
                for row in csv.DictReader(f):
                    anfis_data.append(row)
        except Exception as e:
            print(f"ANFIS data load error: {e}")

def _append_anfis_row(row):
    global anfis_data
    anfis_data.append(row)
    write_header = not os.path.exists(ANFIS_DATA_PATH)
    try:
        with open(ANFIS_DATA_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ANFIS_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"ANFIS data write error: {e}")

def _reset_anfis_data():
    global anfis_data
    anfis_data = []
    if os.path.exists(ANFIS_DATA_PATH):
        try: os.remove(ANFIS_DATA_PATH)
        except Exception as e: print(f"ANFIS reset error: {e}")

_load_anfis_data()
print(f"ANFIS dataset: {len(anfis_data)} existing rows loaded.")


# ══════════════════════════════════════════════════════════════════════════
#  TRANSFER-FUNCTION PARSE HISTORY  (advanced feature)
# ══════════════════════════════════════════════════════════════════════════
TF_HISTORY_MAX = 100
tf_history = []

def _log_tf_history(source, tf_text, result, fopdt):
    tf_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "tf_text": tf_text,
        "order": result["order"],
        "km": round(result["km"], 6),
        "taum_raw": round(result["taum"], 6),
        "time_constants": [round(t, 6) for t in result["time_constants"]],
        "warnings": result["warnings"],
        "fopdt": {"km": round(fopdt[0], 6), "tm": round(fopdt[1], 6), "taum": round(fopdt[2], 6)},
    })
    while len(tf_history) > TF_HISTORY_MAX:
        tf_history.pop(0)


# ══════════════════════════════════════════════════════════════════════════
#  HALF-RULE REDUCER  (Skogestad's method — O'Dwyer 3rd Ed, Appendix)
# ══════════════════════════════════════════════════════════════════════════
class HalfRuleReducer:
    @staticmethod
    def from_fopdt(km, tm, taum, zeta=1.0):
        return km, tm, taum, zeta

    @staticmethod
    def from_sopdt(km, tm1, tm2, taum, zeta=1.0):
        tcs  = sorted([tm1, tm2], reverse=True)
        tm_r = tcs[0]
        tau_r = taum + tcs[1] / 2.0
        return km, tm_r, tau_r, zeta

    @staticmethod
    def from_third_order(km, tm1, tm2, tm3, taum, zeta=1.0):
        tcs  = sorted([tm1, tm2, tm3], reverse=True)
        tm_r = tcs[0]
        tau_r = taum + tcs[1] / 2.0 + tcs[2]
        return km, tm_r, tau_r, zeta

    @staticmethod
    def reduce(order, km, time_constants, taum, zeta=1.0):
        tcs = sorted(time_constants, reverse=True)
        if not tcs:
            return km, max(taum, 1e-6), taum, zeta
        if len(tcs) == 1:
            return km, tcs[0], taum, zeta
        tm_r  = tcs[0]
        tau_r = taum + tcs[1] / 2.0 + sum(tcs[2:])
        return km, tm_r, tau_r, zeta


# ══════════════════════════════════════════════════════════════════════════
#  AUTONOMOUS OPERATOR  (MATLAB mode — no human interview)
# ══════════════════════════════════════════════════════════════════════════
class AutonomousOperator:
    HIGH_RATIO   = 0.5
    SPIKE_RATIO  = 0.7
    GAIN_DRIFT   = 0.30
    RATIO_SPIKE  = 0.20

    def __init__(self):
        self.prev_km    = None
        self.prev_ratio = None

    def decide(self, km, tm, taum):
        ratio   = taum / max(tm, 1e-6)
        reasons = []

        ratio_spiked = (self.prev_ratio is not None and
                        (ratio - self.prev_ratio) > self.RATIO_SPIKE)
        gain_shifted = (self.prev_km is not None and
                        abs(km - self.prev_km) / max(abs(self.prev_km), 1e-6) > self.GAIN_DRIFT)

        if ratio >= self.SPIKE_RATIO or ratio_spiked:
            mode, overshoot, robust, metric = 0, 0, 1, 1
            reasons.append(f"HIGH tau/T={ratio:.2f} or spike → ultra-safe robust mode (IAE, no OS)")
        elif gain_shifted:
            pct = round(abs(km - self.prev_km) / max(abs(self.prev_km), 1e-6) * 100)
            mode, overshoot, robust, metric = 0, 1, 1, 1
            reasons.append(f"Gain shifted {pct}% → robust re-tune (IAE, low OS)")
        elif ratio >= self.HIGH_RATIO:
            mode, overshoot, robust, metric = 0, 1, 0, 3
            reasons.append(f"Moderate tau/T={ratio:.2f} → ITAE regulatory")
        else:
            mode, overshoot, robust, metric = 0, 2, 0, 2
            reasons.append(f"Low tau/T={ratio:.2f} → aggressive ISE regulatory")

        self.prev_km    = km
        self.prev_ratio = ratio
        return {"mode": mode, "overshoot": overshoot, "robust": robust,
                "metric": metric, "reason": " | ".join(reasons)}


# ══════════════════════════════════════════════════════════════════════════
#  NLP INTENT MODEL
# ══════════════════════════════════════════════════════════════════════════
nlp_training_data = [
    ("hi","greeting"),("hello","greeting"),("hey","greeting"),
    ("good morning","greeting"),("howdy","greeting"),
    ("who are you","identity"),("what can you do","identity"),
    ("what is tuning bot","identity"),("tell me about yourself","identity"),
    ("what are you","identity"),("how do you work","identity"),
    ("what tuning rules do you have","rules"),("show me the rules","rules"),
    ("list rules","rules"),("what rules are available","rules"),
    ("show all rules","rules"),("how many rules","rules"),
    ("what is pid","pid_explain"),("explain pid","pid_explain"),
    ("what does pid mean","pid_explain"),("how does pid work","pid_explain"),
    ("what is km","param_explain"),("what is gain","param_explain"),
    ("what is tm","param_explain"),("what is fopdt","param_explain"),
    ("explain fopdt","param_explain"),("what is tau","param_explain"),
    ("what is sopdt","sopdt_explain"),("explain sopdt","sopdt_explain"),
    ("second order","sopdt_explain"),("what is second order","sopdt_explain"),
    ("what is ziegler nichols","rule_explain"),("explain cohen coon","rule_explain"),
    ("what is imc","rule_explain"),("what is lambda tuning","rule_explain"),
    ("what is skogestad","rule_explain"),("what is simc","rule_explain"),
    ("what is iae","metric_explain"),("what is ise","metric_explain"),
    ("what is itae","metric_explain"),("explain performance metrics","metric_explain"),
    ("help","help"),("how do i use this","help"),("guide me","help"),
    ("what should i do","help"),("where do i start","help"),
    ("matlab","matlab_help"),("how do i connect matlab","matlab_help"),
    ("simulink","matlab_help"),("how do i use matlab","matlab_help"),
    ("what is anfis","anfis_explain"),("explain anfis","anfis_explain"),
    ("how does the anfis training work","anfis_explain"),
    ("anfis data","anfis_explain"),("kp ki dataset","anfis_explain"),
    ("what is a transfer function","tf_explain"),("explain transfer function","tf_explain"),
    ("how do i enter a transfer function","tf_explain"),
    ("can you read a transfer function","tf_explain"),
    ("4th order transfer function","tf_explain"),("higher order transfer function","tf_explain"),
]
texts, labels = zip(*nlp_training_data)
intent_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_nlp = intent_vectorizer.fit_transform(texts)
intent_model = LinearSVC()
intent_model.fit(X_nlp, labels)

KNOWLEDGE_BASE = {
    "identity": (
        "<strong>I am TUNING BOT</strong> — an AI-powered PID controller optimization engine.<br><br>"
        "I use a <strong>Random Forest AI model</strong> trained on O'Dwyer's Handbook of PI and PID "
        "Controller Tuning Rules (3rd Edition).<br><br>"
        "Give me your process parameters, answer 4 guided questions, and I will prescribe the optimal "
        "PID settings and simulate the closed-loop response.<br><br>"
        "I also support <strong>Simulink Telemetry Mode</strong> — connect MATLAB directly via "
        "WebSocket/Socket.IO for autonomous real-time tuning of your ANFIS controller. "
        "Type <em>matlab help</em> for the MATLAB integration guide.<br><br>"
        "Don't have Km/Tm/Tau handy? Just paste your <strong>transfer function</strong> (any order, "
        "including 4th-order+) — type <em>what is a transfer function</em> to learn how."
    ),
    "pid_explain": (
        "<strong>PID = Proportional + Integral + Derivative</strong><br><br>"
        "- <strong>Kc (Proportional):</strong> Reacts to current error — larger gap = harder correction<br>"
        "- <strong>Ti (Integral):</strong> Corrects accumulated past errors — eliminates offsets<br>"
        "- <strong>Td (Derivative):</strong> Predicts future error — dampens oscillations<br><br>"
        "For PI control (no derivative), we tune Kc and Ti only, which covers the vast majority of "
        "industrial loops. This is the focus of all 55+ rules in my database."
    ),
    "param_explain": (
        "<strong>FOPDT = First Order Plus Dead Time</strong><br><br>"
        "The standard model for most industrial processes. Requires 3 values:<br><br>"
        "- <strong>Km (Process Gain):</strong> Output change per unit of input. "
        "Open valve 10% → temperature rises 5°C → Km = 0.5<br>"
        "- <strong>Tm (Time Constant):</strong> Time to reach ~63% of final value after a step<br>"
        "- <strong>Tau / taum (Dead Time):</strong> Pure delay before any response begins<br><br>"
        "Obtain from a step test. I also accept 2nd, 3rd, 4th (and higher) order models and "
        "auto-reduce them via Skogestad's Half-Rule — or just paste the transfer function directly."
    ),
    "sopdt_explain": (
        "<strong>SOPDT = Second Order Plus Dead Time</strong><br><br>"
        "G(s) = Km / ((T1·s+1)(T2·s+1)) · e^(-θ·s)<br><br>"
        "Requires 4 values: Km, T1 (dominant lag), T2 (secondary lag), theta (dead time).<br>"
        "Optionally provide <strong>zeta</strong> (damping ratio, default=1.0 for critically damped).<br><br>"
        "My SOPDT rules (from O'Dwyer 3rd Ed) include:<br>"
        "- Skogestad SIMC (servo + regulator)<br>"
        "- Tyreus-Luyben robust<br>"
        "- Wang & Jones min-IAE<br>"
        "- Ho et al. min-IAE<br>"
        "- Rivera IMC, Astrom IMC-based<br><br>"
        "Format: <code>order=2, Km=2, T1=10, T2=3, Tau=2</code>"
    ),
    "tf_explain": (
        "<strong>Transfer Function Input (any order)</strong><br><br>"
        "Instead of typing Km/Tm/Tau by hand, you can paste the process transfer function "
        "directly and I'll extract Km, the time constants, and the dead time myself — "
        "for 1st, 2nd, 3rd, 4th order or higher.<br><br>"
        "Use <code>s</code> as the Laplace variable, <code>*</code> for multiplication, "
        "<code>^</code> or <code>**</code> for powers, and <code>exp(-theta*s)</code> for the delay.<br><br>"
        "Examples:<br>"
        "<code>tf: 5/(10*s+1)</code> — 1st order<br>"
        "<code>tf: 5*exp(-2*s)/(10*s+1)</code> — 1st order + delay<br>"
        "<code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code> — 4th order + delay<br><br>"
        "Higher-order models are automatically reduced to an equivalent FOPDT form using "
        "<strong>Skogestad's Half-Rule</strong> before tuning, and I'll show you the time "
        "constants I found plus any stability warnings (e.g. integrators, unstable poles)."
    ),
    "rule_explain": (
        "I have <strong>55+ tuning rules</strong> from O'Dwyer's Handbook (3rd Ed):<br><br>"
        "<strong>FOPDT rules:</strong><br>"
        "Ziegler-Nichols, Cohen-Coon, Chien (0%/20% OS), Murrill, Rovira (IAE/ITAE),<br>"
        "Zhuang & Atherton (ISE/ISTSE/ISTES), Miluse (0–30% exact OS), Hang (GM=1.5–4),<br>"
        "Rivera IMC, Chien IMC, Haalman DS, Smith & Corripio, Skogestad, Chen Ms-series...<br><br>"
        "<strong>SOPDT rules (new):</strong><br>"
        "Skogestad SIMC, Tyreus-Luyben, Wang & Jones min-IAE, Ho et al. min-IAE,<br>"
        "Rivera IMC extended, Astrom IMC-based, Haalman DS adapted<br><br>"
        "The Random Forest AI selects the optimal rule for your specific process and objectives."
    ),
    "metric_explain": (
        "Performance metrics measure accumulated error — lower = better:<br><br>"
        "- <strong>IAE:</strong> Integral of |e(t)| — equal weight on all errors → smooth, moderate<br>"
        "- <strong>ISE:</strong> Integral of e(t)² — penalises large errors → fast, aggressive, more OS<br>"
        "- <strong>ITAE:</strong> Integral of t·|e(t)| — penalises late errors → best long-term settling<br>"
        "- <strong>ISTSE:</strong> Integral of t²·e(t)² — heavily penalises very late errors<br>"
        "- <strong>ISTES:</strong> Integral of t·e(t)² — composite time-error balance<br><br>"
        "Not sure which to choose? Answer the guided questions and I will select automatically."
    ),
    "matlab_help": (
        "<strong>MATLAB / Simulink Integration</strong><br><br>"
        "Switch to <strong>Simulink Telemetry Mode</strong> (button in the sidebar) to open the live dashboard.<br><br>"
        "Click <strong>MATLAB Guide</strong> in the dashboard header for the full step-by-step integration "
        "instructions with connection diagrams.<br><br>"
        "Quick summary:<br>"
        "1. Install: <code>pip install python-socketio[client] eventlet</code><br>"
        "2. Connect to: <code>ws://your-server/socket.io</code><br>"
        "3. Emit <code>tune_request</code> with your transfer function parameters "
        "(or a raw <code>tf</code> string for any order)<br>"
        "4. Listen for <code>tune_response</code> to receive Kc and Ti<br>"
        "5. Stream live data via <code>telemetry</code> events for the dashboard<br><br>"
        "SOPDT example: <code>{order:2, km:2, tm1:10, tm2:3, taum:2, zeta:1.0}</code><br>"
        "Raw TF example: <code>{tf: \"5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)\"}</code><br><br>"
        "Type <em>what is anfis</em> to learn how Kp/Ki data is logged for ANFIS training."
    ),
    "anfis_explain": (
        "<strong>ANFIS Training Data Pipeline</strong><br><br>"
        "Add a <code>disturbance</code> field to your <code>tune_request</code>:<br>"
        "<code>{order:1, km:2, tm:10, taum:2, disturbance: 0.4}</code><br><br>"
        "For every request that includes <code>disturbance</code>, the server:<br>"
        "1. Computes <strong>Kc (=Kp)</strong> and <strong>Ti</strong> as usual<br>"
        "2. Derives <strong>Ki = Kc / Ti</strong><br>"
        "3. Appends a row <code>(disturbance, Kp, Ki, ...)</code> to "
        "<code>anfis_training_data.csv</code> on the server<br>"
        "4. Broadcasts the new point live to the 3D dashboard chart<br><br>"
        "Download the dataset anytime:<br>"
        "<code>GET /api/anfis-data.csv</code> (CSV) or <code>GET /api/anfis-data</code> (JSON)<br><br>"
        "Your MATLAB ANFIS script loads this file as <code>D_values</code>, "
        "<code>Kp_values</code>, <code>Ki_values</code> and trains "
        "<code>Kp_Data.fis</code> / <code>Ki_Data.fis</code> directly. "
        "See the MATLAB Guide → section 8 for the exact code."
    ),
    "help": (
        "<strong>Getting started — two modes:</strong><br><br>"
        "<strong>Human Chat Mode:</strong><br>"
        "1. Provide: <code>Km=2, Tm=10, Tau=2</code> (FOPDT)<br>"
        "   Or: <code>order=2, Km=2, T1=10, T2=3, Tau=2</code> (SOPDT)<br>"
        "   Or paste a <strong>transfer function</strong> directly (any order): "
        "<code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code><br>"
        "2. Answer 4 plain-language questions<br>"
        "3. Receive optimal Kc/Ti and a step-response simulation<br><br>"
        "<strong>Simulink Telemetry Mode:</strong><br>"
        "Click the amber button in the sidebar to open the live dashboard.<br>"
        "Connect MATLAB via Socket.IO for autonomous real-time tuning.<br>"
        "Type <em>matlab help</em> for the full guide."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════════
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "metric": None, "robust": None, "overshoot": None,
    "interview_stage": 0, "allows_overshoot": False, "overshoot_answer": None,
    "order": 1, "tm2": None, "tm3": None, "tm4": None, "tm5": None, "tm6": None,
    "zeta": 1.0
}

auto_operator = AutonomousOperator()

def _reset_memory():
    global bot_memory
    bot_memory = {
        "km": None, "tm": None, "taum": None, "tau_c": None,
        "mode": None, "metric": None, "robust": None, "overshoot": None,
        "interview_stage": 0, "allows_overshoot": False, "overshoot_answer": None,
        "order": 1, "tm2": None, "tm3": None, "tm4": None, "tm5": None, "tm6": None,
        "zeta": 1.0
    }

# ══════════════════════════════════════════════════════════════════════════
#  TRANSFER-FUNCTION TEXT DETECTION
# ══════════════════════════════════════════════════════════════════════════
TF_TRIGGER_RE  = re.compile(r'(transfer function|tf\s*[:=]|g\(s\)|gp\(s\))', re.IGNORECASE)
TF_STRIP_RE    = re.compile(r'(?i)^.*?(transfer function|tf|g\(s\)|gp\(s\))\s*[:=]?\s*')
ORDER_SUFFIX   = {1: "st", 2: "nd", 3: "rd"}

def _order_suffix(n):
    return ORDER_SUFFIX.get(n, "th")


# ══════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════════
def simulate_step(kc, ti, km, tm, taum):
    t  = np.linspace(0, (tm + taum) * 8, 600)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum = 0
    d_steps = max(1, int(taum / dt))
    ti_val  = max(ti, 0.001)

    for i in range(1, len(t)):
        err      = 1.0 - pv[i-1]
        err_sum += err * dt
        mv       = max(0, min(100, kc * (err + (1/ti_val) * err_sum)))
        mv_hist[i] = mv
        d_idx    = i - d_steps
        d_mv     = mv_hist[d_idx] if d_idx >= 0 else 0
        dpv      = ((km * d_mv) - pv[i-1]) / tm
        pv[i]    = pv[i-1] + dpv * dt

    os_val = 0.0
    try:
        os_val = round(max(0, (np.max(pv) - 1.0) * 100), 1)
        if math.isnan(os_val) or math.isinf(os_val): os_val = 0.0
    except: pass

    settling_time = round(t[-1], 1)
    for i in range(len(pv)-1, 0, -1):
        if abs(pv[i] - 1.0) > 0.02:
            settling_time = round(t[i], 1)
            break

    graph_data = {
        "data": [
            {"x": t.tolist(), "y": pv.tolist(), "type": "scatter", "name": "Process Output",
             "line": {"color": "#00d4ff", "width": 2.5},
             "fill": "tozeroy", "fillcolor": "rgba(0,212,255,0.04)"},
            {"x": [t[0], t[-1]], "y": [1.0, 1.0], "type": "scatter", "name": "Setpoint",
             "line": {"color": "#ff4d6d", "dash": "dash", "width": 1.5}},
            {"x": [t[0], t[-1]], "y": [1.02, 1.02], "type": "scatter", "name": "+2% Band",
             "line": {"color": "rgba(255,255,255,0.12)", "dash": "dot", "width": 1},
             "showlegend": False},
            {"x": [t[0], t[-1]], "y": [0.98, 0.98], "type": "scatter", "name": "-2% Band",
             "line": {"color": "rgba(255,255,255,0.12)", "dash": "dot", "width": 1},
             "showlegend": False},
        ],
        "layout": {
            "title": {"text": "Closed-Loop Step Response", "font": {"size": 14}},
            "xaxis": {"title": "Time (s)", "gridcolor": "#1e2a3a", "zerolinecolor": "#333"},
            "yaxis": {"title": "Normalised PV", "gridcolor": "#1e2a3a", "zerolinecolor": "#333",
                      "range": [-0.05, max(1.5, os_val/100 + 1.1)]},
            "paper_bgcolor": "transparent", "plot_bgcolor": "rgba(10,14,20,0.6)",
            "font": {"color": "#a0aec0"},
            "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(255,255,255,0.1)", "borderwidth": 1},
            "margin": {"l": 50, "r": 20, "t": 45, "b": 45},
            "hovermode": "x unified"
        }
    }
    return json.dumps(graph_data), os_val, settling_time


# ══════════════════════════════════════════════════════════════════════════
#  CORE TUNING ENGINE  (shared by REST and WebSocket)
# ══════════════════════════════════════════════════════════════════════════
def run_tuning(km, tm, taum, mode, overshoot, robust, metric,
               overshoot_answer=None, zeta=1.0, order=1, D=0.0):
    OVERSHOOT_OVERRIDE = {
        "os_5": "miluse_5os", "os_10": "miluse_10os",
        "os_20": "miluse_20os", "os_30": "miluse_30os"
    }

    ratio    = taum / max(tm, 1e-6)
    features = np.array([[km, tm, taum, ratio, mode, overshoot, robust, metric]])

    fopdt_keys  = [k for k, v in rules_db.items() if v.get('order', 1) == 1 and v.get('kc_math') != 'SPECIAL_LOOKUP']
    sopdt_keys  = [k for k, v in rules_db.items() if v.get('order', 1) == 2 and v.get('kc_math') != 'SPECIAL_LOOKUP']

    if order >= 2 and sopdt_keys:
        candidate_rules = sopdt_keys + fopdt_keys
    else:
        candidate_rules = fopdt_keys

    # ── DISTURBANCE-DRIVEN SMOOTH RULE SELECTION ──
    best_rule = "ziegler_nichols"
    is_disturbance = (mode == 0)

    if is_disturbance:
        if D >= 1.0:
            best_rule = "tyreus_luyben" if "tyreus_luyben" in rules_db else "skogestad"
        elif D >= 0.2:
            best_rule = "chien_smooth" if "chien_smooth" in rules_db else "rovira_iae"
        else:
            best_rule = "skogestad" if "skogestad" in rules_db else "cohen_coon"
    else:
        if rf_model:
            try: best_rule = rf_model.predict(features)[0]
            except: pass
        if best_rule not in rules_db:
            best_rule = "skogestad" if "skogestad" in rules_db else "ziegler_nichols"

    if best_rule not in rules_db: 
        best_rule = "ziegler_nichols"

    if overshoot_answer and overshoot_answer in OVERSHOOT_OVERRIDE:
        ok = OVERSHOOT_OVERRIDE[overshoot_answer]
        if ok in rules_db: best_rule = ok

    if order >= 2 and sopdt_keys:
        if robust: sopdt_pref = "skogestad_sopdt_reg" if mode == 0 else "skogestad_sopdt_servo"
        elif metric == 2: sopdt_pref = "ho_sopdt_iae"
        elif metric == 1: sopdt_pref = "wang_jones_lopdt_sopdt"
        else: sopdt_pref = "tyreus_luyben_sopdt"
        if sopdt_pref in rules_db: best_rule = sopdt_pref

    r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

    tau_c_val = max(0.1, taum)
    safe_env = {
        "km": km, "tm": tm, "taum": taum, "tau_c": tau_c_val, "zeta": max(zeta, 0.1),
        "min": min, "max": max, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "np": np, "math": math
    }

    try:
        if r.get('kc_math') == 'SPECIAL_LOOKUP':
            kc = (0.85 * tm) / (km * taum)
            ti = 2.4 * taum
        else:
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
            if not ti or ti <= 0 or math.isnan(ti): ti = 2.0 * taum
            if not kc or kc <= 0 or math.isnan(kc): kc = (1.2 * tm) / (km * taum)
    except Exception as e:
        print(f"Math eval error for rule {best_rule}: {e}")
        kc = (1.2 * tm) / (km * taum)
        ti = 2.0 * taum
        r  = rules_db.get("ziegler_nichols", {"name": "Ziegler-Nichols"})

    chart, os_est, settling = simulate_step(kc, ti, km, tm, taum)
    return (kc, ti, best_rule, r.get('name', best_rule), r.get('unique_feature', ''), chart, os_est, settling)


def quick_pi_estimate(km, tm, taum, lam=None):
    km = km if abs(km) > 1e-9 else 1e-9
    if lam is None:
        lam = max(taum, 0.5 * tm)
    lam = max(lam, 1e-6)
    kc = tm / (km * (lam + taum))
    ti = min(tm, 4 * (lam + taum))
    return round(kc, 4), round(ti, 4), round(lam, 4)


# ══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET EVENTS  (MATLAB/Simulink interface)
# ══════════════════════════════════════════════════════════════════════════
last_valid_tune = None

@socketio.on('connect')
def handle_connect():
    print(f"[WS] Connected: {request.sid}")
    emit('status', {'message': 'TUNING BOT connected. Ready for telemetry.'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WS] Disconnected: {request.sid}")


@socketio.on('telemetry')
def handle_telemetry(data):
    try:
        socketio.emit('telemetry_update', {
            "t":  data.get("t",  []),
            "pv": data.get("pv", []),
            "sp": data.get("sp", []),
            "mv": data.get("mv", []),
        })
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('tune_request')
def handle_tune_request(data):
    global last_valid_tune
    try:
        warnings_list = []
        tf_text = data.get("tf")

        if tf_text:
            try:
                tf_result = parse_transfer_function(tf_text)
            except TFParseError as e:
                emit('tune_response', {'status': 'error', 'message': f'TF parse error: {e}'})
                return
            order = max(1, tf_result["order"])
            km    = tf_result["km"]
            taum  = tf_result["taum"]
            tcs   = tf_result["time_constants"] or [1.0]
            warnings_list = tf_result["warnings"]
        else:
            order = int(data.get("order", 1))
            km    = float(data.get("km",   1.0))
            taum  = float(data.get("taum", 1.0))

            if order == 1:
                tcs = [float(data.get("tm", 10.0))]
            elif order == 2:
                tcs = [float(data.get("tm1", 10.0)), float(data.get("tm2", 3.0))]
            elif order == 3:
                tcs = [float(data.get("tm1", 10.0)), float(data.get("tm2",  3.0)), float(data.get("tm3",  1.0))]
            else:
                tcs = [float(data.get(f"tm{i}", 10.0 / i)) for i in range(1, order + 1)]

        zeta = float(data.get("zeta", 1.0))
        km_r, tm_r, taum_r, zeta_r = HalfRuleReducer.reduce(order, km, tcs, taum, zeta)

        if order > 1:
            tc_str = ", ".join(f"τ{i+1}={round(t,4)}" for i, t in enumerate(sorted(tcs, reverse=True)))
            note = (f"Order-{order} TF ({tc_str}) reduced via Skogestad Half-Rule → "
                    f"FOPDT: Km={round(km_r,4)}, Tm={round(tm_r,4)}, Tau={round(taum_r,4)}")
        else:
            note = f"FOPDT: Km={km_r}, Tm={tm_r}, Tau={taum_r}"

        if tf_text: _log_tf_history("websocket", tf_text, tf_result, (km_r, tm_r, taum_r))

        disturbance_raw = data.get("disturbance")
        D_val = abs(float(disturbance_raw)) if disturbance_raw is not None else 0.0

        decision = auto_operator.decide(km_r, tm_r, taum_r)

        kc, ti, rule_key, rule_name, rule_desc, chart, os_est, settling = run_tuning(
            km_r, tm_r, taum_r,
            decision["mode"], decision["overshoot"],
            decision["robust"], decision["metric"],
            zeta=zeta_r, order=order, D=D_val
        )

      # 🔥 DISTURBANCE SCALING (CRITICALLY DAMPED / NO OSCILLATION) 🔥
        if D_val > 0:
            # 1. CRANK THE GAIN MASSIVELY
            # You want high gain to stop the drop instantly. We use a massive linear multiplier.
            empirical_boost = 1.0 + (150.0 * D_val)
            kc = kc * empirical_boost

            # 2. KILL THE OSCILLATION (CRITICAL DAMPING)
            # Ringing happens when Ti is too small. To force a smooth, first-order
            # exponential curve, Ti must be large enough to prevent integral windup.
            # We override the base rule and tie Ti directly to your plant's time constant.
            ti = max(tm_r * 0.8, 3.5)

            # 3. RAISE THE CEILING
            # Allow the bot to output massive Kp to fight the drop
            kc = min(kc, 80000.0)

            rule_name = f"Smooth First-Order Recovery (Boost: {round(empirical_boost, 1)}x)"

        qkc, qti, qlam = quick_pi_estimate(km_r, tm_r, taum_r)

        response = {
            "status":            "ok",
            "reduction_note":    note,
            "operator_decision": decision["reason"],
            "rule_key":          rule_key,
            "rule":              rule_name,
            "kc":                round(kc, 6),
            "ti":                round(ti, 6),
            "os_predicted":      os_est,
            "settling_time":     settling,
            "fopdt": { "km": round(km_r, 4), "tm": round(tm_r, 4), "taum": round(taum_r, 4) },
            "quick_estimate": {"kc": qkc, "ti": qti, "lambda": qlam},
            "warnings": warnings_list,
            "chart": chart
        }

        if disturbance_raw is not None:
            ki_val = kc / ti if ti else 0.0
            row = {
                'timestamp':   datetime.now(timezone.utc).isoformat(),
                'disturbance': round(float(disturbance_raw), 6),
                'km':   round(km_r,  4), 'tm':   round(tm_r,  4), 'taum': round(taum_r, 4),
                'kc':   round(kc, 6),    'ti':   round(ti, 6),
                'kp':   round(kc, 6),    'ki':   round(ki_val, 6),
                'rule': rule_key,        'order': order
            }
            _append_anfis_row(row)
            response["anfis_row"]   = row
            response["anfis_total"] = len(anfis_data)
            
            socketio.emit('anfis_data_update', {
                "row": row, "total_points": len(anfis_data), "reset": False
            })

        last_valid_tune = response
        socketio.emit('tune_response', response)
        print(f"[WS] Tuned: rule={rule_key} Kc={kc:.4f} Ti={ti:.4f}")

    except Exception as e:
        error_msg = str(e)
        print(f"CRASH: {error_msg}")
        print(traceback.format_exc())
        socketio.emit('tune_response', {'status': 'error', 'message': error_msg})


# ══════════════════════════════════════════════════════════════════════════
#  REST API — INTERVIEW FLOW  (human chat)
# ══════════════════════════════════════════════════════════════════════════
INTERVIEW = {
    1: {
        "text": (
            "Parameters confirmed. Let me understand your control objective.<br><br>"
            "<strong>Q1 of 4 — What is this control loop doing?</strong>"
        ),
        "options": [
            {"label": "Following a target — setpoint tracking (servo)", "val": "servo",
             "hint": "e.g. robot arm, temperature profile"},
            {"label": "Holding steady — rejecting disturbances (regulator)", "val": "regulator",
             "hint": "e.g. pressure vessel, boiler level"}
        ],
        "map": {"servo": {"mode": 1}, "regulator": {"mode": 0}}
    },
    2: {
        "text": "<strong>Q2 of 4 — Desired response behaviour?</strong>",
        "options": [
            {"label": "Maximum speed — overshoot acceptable", "val": "fast",    "hint": "ISE criterion"},
            {"label": "Smooth approach — no overshoot",       "val": "smooth",  "hint": "IAE criterion"},
            {"label": "Balanced — reasonably fast, clean",    "val": "balanced","hint": "ITAE criterion"}
        ],
        "map": {
            "fast":     {"metric": 2, "overshoot": 2, "robust": 0},
            "smooth":   {"metric": 1, "overshoot": 0, "robust": 1},
            "balanced": {"metric": 3, "overshoot": 1, "robust": 0}
        }
    },
    3: {
        "text": "<strong>Q3 of 4 — Confidence in your process model?</strong>",
        "options": [
            {"label": "Very confident — careful step test performed", "val": "confident", "hint": "Full performance"},
            {"label": "Approximately correct — estimated from data",  "val": "estimated", "hint": "Moderate detuning"},
            {"label": "Uncertain — rough estimate, process varies",   "val": "uncertain", "hint": "Robust safety margin"}
        ],
        "map": {"confident": {"robust": 0}, "estimated": {"robust": 0}, "uncertain": {"robust": 1}}
    },
    4: {
        "text": "<strong>Q4 of 4 — Maximum acceptable overshoot?</strong>",
        "options": [
            {"label": "Up to 5%",   "val": "os_5",  "hint": "Miluse 5%"},
            {"label": "Up to 10%",  "val": "os_10", "hint": "Miluse 10%"},
            {"label": "Up to 20%",  "val": "os_20", "hint": "Standard industrial"},
            {"label": "30% or more","val": "os_30", "hint": "Max speed"}
        ],
        "map": {
            "os_5": {"overshoot": 1}, "os_10": {"overshoot": 1},
            "os_20": {"overshoot": 2}, "os_30": {"overshoot": 3}
        }
    }
}

ANSWER_PATTERNS = {
    1: {"servo":     ["servo","setpoint","track","follow","target","position","profile"],
        "regulator": ["regulator","disturbance","hold","steady","reject","fixed","constant"]},
    2: {"fast":     ["fast","quick","aggressive","speed","ise","overshoot"],
        "smooth":   ["smooth","safe","slow","no overshoot","gentle","iae","soft"],
        "balanced": ["balance","balanced","standard","normal","itae","moderate","typical"]},
    3: {"confident": ["confident","accurate","careful","step test","measured","certain"],
        "estimated": ["estimated","roughly","approximate","historical","quick"],
        "uncertain": ["uncertain","unsure","rough","guess","varies","not sure"]},
    4: {"os_5":  ["5","five","5%","very controlled"],
        "os_10": ["10","ten","10%","moderate"],
        "os_20": ["20","twenty","20%","standard","industrial","typical"],
        "os_30": ["30","thirty","30%","maximum","aggressive","fast"]}
}

def _parse_answer(stage, msg_lower):
    for ans, keywords in ANSWER_PATTERNS.get(stage, {}).items():
        if any(kw in msg_lower for kw in keywords):
            return ans
    return None

def _extract_params_regex(msg):
    ext = {"km": None, "tm": None, "taum": None, "order": None, "zeta": None}
    ml  = msg.lower()
    km_m  = re.search(r'(km|gain)\s*(=|:)?\s*([+-]?\d+\.?\d*)', ml)
    tm_m  = re.search(r'\b(tm|t1|lag|time\s*constant)\s*(=|:)?\s*([+-]?\d+\.?\d*)', ml)
    tau_m = re.search(r'(tau|dead\s*time|delay|theta)\s*(=|:)?\s*([+-]?\d+\.?\d*)', ml)
    ord_m = re.search(r'order\s*(=|:)?\s*([1-6])', ml)
    zeta_m= re.search(r'zeta\s*(=|:)?\s*([+-]?\d+\.?\d*)', ml)
    if km_m:   ext["km"]    = float(km_m.group(3))
    if tm_m:   ext["tm"]    = float(tm_m.group(3))
    if tau_m:  ext["taum"]  = float(tau_m.group(3))
    if ord_m:  ext["order"] = int(ord_m.group(2))
    if zeta_m: ext["zeta"]  = float(zeta_m.group(2))

    for n in range(2, 7):
        m = re.search(rf'\b(tm{n}|t{n})\s*(=|:)?\s*([+-]?\d+\.?\d*)', ml)
        if m:
            ext[f"tm{n}"] = float(m.group(3))
    return ext

def _build_rules_response():
    fopdt_reg, fopdt_srv, sopdt_r, sopdt_s = [], [], [], []
    for k, v in rules_db.items():
        ord_ = v.get('order', 1)
        mode = v.get('mode', 'regulator')
        nm   = v.get('name', k)
        if ord_ == 1:
            (fopdt_srv if mode in ('servo', 1) else fopdt_reg).append(nm)
        else:
            (sopdt_s if mode in ('servo', 1) else sopdt_r).append(nm)

    reply  = f"<strong>Tuning database: {len(rules_db)} rules</strong> from O'Dwyer's Handbook 3rd Ed.<br><br>"
    n = 1
    for cat, lst in [("FOPDT — Regulatory",fopdt_reg),("FOPDT — Servo",fopdt_srv),
                     ("SOPDT — Regulatory",sopdt_r),("SOPDT — Servo",sopdt_s)]:
        if lst:
            reply += f"<strong>{cat}</strong><br>"
            for nm in lst: reply += f"{n}. {nm}<br>"; n += 1
            reply += "<br>"
    reply += "<em>AI selects the optimal rule for your process type, objectives and model confidence.</em>"
    return jsonify({"reply": reply, "options": [], "chart": None})

def _build_tcs_from_memory():
    order = bot_memory.get('order', 1)
    tm    = bot_memory.get('tm')
    tcs   = [tm]
    for n in range(2, order + 1):
        val = bot_memory.get(f'tm{n}')
        if val is not None:
            tcs.append(val)
    return tcs

def _run_rest_tuning():
    global bot_memory
    km   = bot_memory['km'];   tm    = bot_memory['tm']
    taum = bot_memory['taum']; mode  = bot_memory.get('mode', 1)
    os_v = bot_memory.get('overshoot', 0); rob  = bot_memory.get('robust', 0)
    met  = bot_memory.get('metric', 1);    oa   = bot_memory.get('overshoot_answer')
    zeta = bot_memory.get('zeta', 1.0);    order= bot_memory.get('order', 1)

    km_r, tm_r, taum_r = km, tm, taum
    reduction_note = ""
    if order >= 2:
        tcs = _build_tcs_from_memory()
        if len(tcs) >= 2:
            km_r, tm_r, taum_r, _ = HalfRuleReducer.reduce(order, km, tcs, taum, zeta)
            tc_str = ", ".join(f"τ{i+1}={round(t,4)}" for i, t in enumerate(sorted(tcs, reverse=True)))
            reduction_note = (
                f"<br>Your order-{order} model ({tc_str}) was reduced via "
                f"<strong>Skogestad's Half-Rule</strong> "
                f"→ FOPDT: Km={round(km_r,4)}, Tm={round(tm_r,4)}, Tau={round(taum_r,4)}<br>"
            )

    kc, ti, rule_key, rule_name, rule_desc, chart, os_est, settling = run_tuning(
        km_r, tm_r, taum_r, mode, os_v, rob, met, oa, zeta=zeta, order=order
    )

    ratio      = taum_r / max(tm_r, 1e-6)
    mode_str   = "Servo (Setpoint Tracking)" if mode == 1 else "Regulatory (Disturbance Rejection)"
    metric_map = {"1": "IAE — smooth", "2": "ISE — aggressive", "3": "ITAE — balanced long-term"}
    metric_str = metric_map.get(str(met), "IAE")
    robust_str = "Robust (uncertainty-tolerant)" if rob == 1 else "Performance-optimised"
    order_str  = f"Order-{order} input" + (" (Half-Rule reduced)" if order > 1 else "") + "<br>"

    final_reply = (
        f"<strong>Optimization Complete</strong><br><br>"
        f"<strong>Process:</strong><br>"
        f"- Model: {order_str}"
        f"{reduction_note}"
        f"- Mode: {mode_str}<br>"
        f"- Objective: {metric_str}<br>"
        f"- Philosophy: {robust_str}<br>"
        f"- Dead-time ratio (tau/T): {round(ratio, 3)}<br><br>"
        f"<strong>AI Rule Selected:</strong> <strong>{rule_name}</strong><br>"
        f"<em>{rule_desc}</em><br><br>"
        f"<strong>Tuned PID Parameters:</strong><br>"
        f"- Proportional Gain <code>Kc = {round(kc, 4)}</code><br>"
        f"- Integral Time <code>Ti = {round(ti, 4)} s</code><br><br>"
        f"<strong>Predicted Closed-Loop Performance:</strong><br>"
        f"- Overshoot: <strong>{os_est}%</strong><br>"
        f"- Settling Time (~2% band): <strong>{settling} s</strong><br><br>"
        f"Step response simulation below."
    )
    _reset_memory()
    return jsonify({"reply": final_reply, "chart": chart, "options": []})


@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg  = request.json.get('message', '')
        ul        = user_msg.lower().strip()

        if ul == "reset":
            _reset_memory()
            return jsonify({
                "reply": (
                    "Session reset. Ready for a new process.<br><br>"
                    "FOPDT: <code>Km=2, Tm=10, Tau=2</code><br>"
                    "SOPDT: <code>order=2, Km=2, T1=10, T2=3, Tau=2</code><br>"
                    "Or paste a transfer function: <code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code>"
                ),
                "options": [], "chart": None
            })

        tf_note = ""
        tf_parsed = False
        if TF_TRIGGER_RE.search(user_msg):
            expr = TF_STRIP_RE.sub('', user_msg).strip()
            try:
                tf_result = parse_transfer_function(expr)
            except TFParseError as e:
                return jsonify({
                    "reply": (
                        f"⚠️ I couldn't parse that transfer function: {e}<br><br>"
                        "Examples:<br>"
                        "<code>tf: 5/(10*s+1)</code> — 1st order<br>"
                        "<code>tf: 5*exp(-2*s)/(10*s+1)</code> — 1st order + delay<br>"
                        "<code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code> — 4th order + delay"
                    ),
                    "options": [], "chart": None
                })

            tcs   = tf_result["time_constants"]
            order = max(1, tf_result["order"])
            bot_memory["km"]    = tf_result["km"]
            bot_memory["taum"]  = tf_result["taum"]
            bot_memory["order"] = order
            bot_memory["tm"]    = tcs[0] if tcs else max(tf_result["taum"], 1.0)
            for i, tau in enumerate(tcs[1:], start=2):
                bot_memory[f"tm{i}"] = tau

            km_r, tm_r, taum_r, _ = HalfRuleReducer.reduce(
                order, tf_result["km"], tcs or [bot_memory["tm"]], tf_result["taum"])
            _log_tf_history("chat", expr, tf_result, (km_r, tm_r, taum_r))

            tc_str    = ", ".join(f"τ{i+1}={round(t,4)}" for i, t in enumerate(tcs)) or "—"
            warn_html = "".join(f"⚠️ {w}<br>" for w in tf_result["warnings"])
            tf_note = (
                f"<strong>Transfer function parsed</strong> "
                f"({order}{_order_suffix(order)} order):<br>"
                f"Km={round(tf_result['km'],4)}, dead time θ={round(tf_result['taum'],4)}, "
                f"time constants: {tc_str}<br>{warn_html}"
            )
            if order > 1:
                tf_note += (
                    f"Reduced (Half-Rule) → FOPDT: Km={round(km_r,4)}, "
                    f"Tm={round(tm_r,4)}, Tau={round(taum_r,4)}<br>"
                )
            tf_note += "<br>"
            tf_parsed = True

        has_digits = any(ch.isdigit() for ch in ul)
        word_count = len(ul.split())

        if not tf_parsed and word_count < 14 and not has_digits:
            try:
                vec   = intent_vectorizer.transform([ul])
                li    = intent_model.predict(vec)[0]
                conf  = max(intent_model.decision_function(vec)[0])
            except: li, conf = "none", 0

            if li == "greeting" and conf > 0.3:
                return jsonify({
                    "reply": (
                        "Hello. I am <strong>TUNING BOT</strong>.<br><br>"
                        "Provide process parameters to begin:<br>"
                        "FOPDT: <code>Km=2, Tm=10, Tau=2</code><br>"
                        "SOPDT: <code>order=2, Km=2, T1=10, T2=3, Tau=2</code><br>"
                        "Or paste a transfer function: <code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code>"
                    ),
                    "options": [], "chart": None
                })
            if li in KNOWLEDGE_BASE and conf > 0.2:
                return jsonify({"reply": KNOWLEDGE_BASE[li], "options": [], "chart": None})
            if li == "rules" and conf > 0.2:
                return _build_rules_response()

        if not tf_parsed and any(kw in ul for kw in ["what rules","show rules","list rules","how many rules"]):
            return _build_rules_response()

        if not tf_parsed:
            regex_ext = _extract_params_regex(user_msg)
            for k in ["km", "tm", "taum", "zeta", "tm2", "tm3", "tm4", "tm5", "tm6"]:
                if regex_ext.get(k) is not None:
                    bot_memory[k] = regex_ext[k]
            if regex_ext["order"] is not None:
                bot_memory["order"] = regex_ext["order"]

            if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]) and has_digits:
                try:
                    prompt = (
                        f'Extract FOPDT/higher-order process params from: "{user_msg}". '
                        f'Current: km={bot_memory["km"]}, tm={bot_memory["tm"]}, '
                        f'taum={bot_memory["taum"]}, tm2={bot_memory["tm2"]}, '
                        f'tm3={bot_memory["tm3"]}, tm4={bot_memory["tm4"]}, order={bot_memory["order"]}. '
                        'OUTPUT JSON ONLY: {"km":float_or_null,"tm":float_or_null,"taum":float_or_null,'
                        '"tm2":float_or_null,"tm3":float_or_null,"tm4":float_or_null,"tm5":float_or_null,'
                        '"tm6":float_or_null,"order":int_or_null,"zeta":float_or_null}'
                    )
                    res = llm_model.generate_content(prompt)
                    ext = json.loads(res.text.replace('```json','').replace('```','').strip())
                    for k in ["km","tm","taum","zeta","tm2","tm3","tm4","tm5","tm6"]:
                        if ext.get(k) is not None: bot_memory[k] = ext[k]
                    if ext.get("order") is not None: bot_memory["order"] = ext["order"]
                except: pass

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        missing = []
        if not km:   missing.append("<strong>Km</strong> (process gain)")
        if not tm:   missing.append("<strong>Tm</strong> (time constant / T1 for higher-order)")
        if not taum: missing.append("<strong>Tau</strong> (dead time)")

        if missing:
            return jsonify({
                "reply": (
                    "I need: " + ", ".join(missing) + "<br><br>"
                    "FOPDT: <code>Km=2, Tm=10, Tau=2</code><br>"
                    "SOPDT: <code>order=2, Km=2, T1=10, T2=3, Tau=2</code><br>"
                    "4th order: <code>order=4, Km=2, T1=10, T2=4, T3=1.5, T4=0.5, Tau=0.5</code><br>"
                    "Or paste a transfer function: <code>tf: 5/((10*s+1)*(5*s+1)*(2*s+1)*(1*s+1))*exp(-1*s)</code><br>"
                    "Type <em>explain FOPDT</em>, <em>explain SOPDT</em>, or "
                    "<em>what is a transfer function</em> for help."
                ),
                "options": [], "chart": None
            })

        stage = bot_memory['interview_stage']

        if stage > 0:
            iq     = INTERVIEW[stage]
            answer = None
            for opt in iq["options"]:
                if ul == opt["val"]: answer = opt["val"]; break
            if not answer: answer = _parse_answer(stage, ul)

            if answer:
                bot_memory.update(iq["map"][answer])
                if stage == 2: bot_memory['allows_overshoot'] = (answer == "fast")
                if stage == 4: bot_memory['overshoot_answer'] = answer
                bot_memory['interview_stage'] += 1
                stage = bot_memory['interview_stage']

                if stage == 4 and not bot_memory.get('allows_overshoot'):
                    bot_memory['interview_stage'] = 5
                    return _run_rest_tuning()
                if stage <= 4:
                    nq = INTERVIEW[stage]
                    return jsonify({
                        "reply": nq["text"],
                        "options": [{"label": o["label"], "val": o["val"]} for o in nq["options"]],
                        "chart": None
                    })
                return _run_rest_tuning()
            else:
                return jsonify({
                    "reply": f"I didn't recognise that selection.<br><br>{iq['text']}",
                    "options": [{"label": o["label"], "val": o["val"]} for o in iq["options"]],
                    "chart": None
                })

        if stage == 0:
            bot_memory['interview_stage'] = 1
            ratio = taum / max(tm, 1e-6)
            ord_  = bot_memory.get('order', 1)
            order_note = f"<br>Model order: <strong>{ord_}{_order_suffix(ord_)}</strong>." if ord_ > 1 else ""
            if   ratio < 0.2: rnote = f" Dead-time ratio {round(ratio,2)} — low, highly controllable."
            elif ratio < 0.5: rnote = f" Dead-time ratio {round(ratio,2)} — moderate."
            else:              rnote = f" Dead-time ratio {round(ratio,2)} — high, dead-time compensation rules apply."
            q = INTERVIEW[1]
            return jsonify({
                "reply": (
                    f"{tf_note}"
                    f"Parameters confirmed: <strong>Km={km}, Tm={tm}s, Tau={taum}s</strong>.{order_note}{rnote}<br><br>"
                    f"{q['text']}"
                ),
                "options": [{"label": o["label"], "val": o["val"]} for o in q["options"]],
                "chart": None
            })

        return jsonify({"reply": "State error. Please reset.", "options": [], "chart": None})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"reply": "System error. Please reset.", "options": []})


# ══════════════════════════════════════════════════════════════════════════
#  ANFIS DATASET ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.route('/api/anfis-data', methods=['GET'])
def get_anfis_data():
    return jsonify({"data": anfis_data, "count": len(anfis_data)})


@app.route('/api/anfis-data.csv', methods=['GET'])
def get_anfis_csv():
    if not os.path.exists(ANFIS_DATA_PATH):
        empty = ','.join(ANFIS_FIELDS) + '\n'
        return app.response_class(empty, mimetype='text/csv',
                                   headers={"Content-Disposition": "attachment; filename=anfis_training_data.csv"})
    return send_file(ANFIS_DATA_PATH, mimetype='text/csv',
                      as_attachment=True, download_name='anfis_training_data.csv')


@app.route('/api/anfis-reset', methods=['POST'])
def reset_anfis_data():
    _reset_anfis_data()
    socketio.emit('anfis_data_update', {"row": None, "total_points": 0, "reset": True})
    return jsonify({"status": "ok", "count": len(anfis_data)})


# ══════════════════════════════════════════════════════════════════════════
#  TRANSFER-FUNCTION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.route('/api/parse-tf', methods=['POST'])
def parse_tf_endpoint():
    tf_text = (request.json or {}).get('tf', '')
    if not tf_text:
        return jsonify({"status": "error", "message": "Missing 'tf' field."}), 400
    try:
        result = parse_transfer_function(tf_text)
    except TFParseError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    tcs = result["time_constants"] or [1.0]
    km_r, tm_r, taum_r, _ = HalfRuleReducer.reduce(
        max(1, result["order"]), result["km"], tcs, result["taum"])
    qkc, qti, qlam = quick_pi_estimate(km_r, tm_r, taum_r)

    _log_tf_history("api/parse-tf", tf_text, result, (km_r, tm_r, taum_r))

    return jsonify({
        "status": "ok",
        "order": result["order"],
        "km": round(result["km"], 6),
        "time_constants": [round(t, 6) for t in result["time_constants"]],
        "taum": round(result["taum"], 6),
        "warnings": result["warnings"],
        "fopdt_equivalent": {"km": round(km_r, 6), "tm": round(tm_r, 6), "taum": round(taum_r, 6)},
        "quick_pi_estimate": {"kc": qkc, "ti": qti, "lambda": qlam},
    })


@app.route('/api/tf-history', methods=['GET'])
def get_tf_history():
    return jsonify({"data": tf_history, "count": len(tf_history)})

# ══════════════════════════════════════════════════════════════════════════
#  REST API FALLBACK (Bypasses MATLAB Socket.IO Thread Blocking)
# ══════════════════════════════════════════════════════════════════════════
@app.route('/api/tune', methods=['POST'])
def api_tune_fallback():
    # THESE 5 LINES WERE MISSING!
    data = request.json or {}
    D_val = float(data.get('disturbance', 0.0))
    pv_hist = data.get('pv_history', [])
    mv_hist = data.get('mv_history', [])
    
    # --- 1. REAL-TIME SYSTEM IDENTIFICATION (SysID) ---
    # FIX: Change default Km to match your plant's tiny magnitude! 
    # If Km is 0.002, 1/Km = 500, which aligns perfectly with your ANFIS 880 baseline.
    km, tm, taum = 0.002, 12.3, 2.0  
    
    if len(pv_hist) > 10 and len(mv_hist) > 10:
        delta_pv = pv_hist[-1] - pv_hist[0]
        delta_mv = mv_hist[-1] - mv_hist[0]
        
        # FIX: Lowered the threshold to 0.001 to catch tiny plant movements
        if abs(delta_mv) > 0.001 and abs(delta_pv) > 0.001:
            km = abs(delta_pv / delta_mv)
            
            # Prevent Km from hitting exact zero, which would cause infinite Kc
            if km < 0.0001:
                km = 0.002 
            
            target_pv = pv_hist[0] + (0.632 * delta_pv)
            tm_index = 0
            for i, pv in enumerate(pv_hist):
                if (delta_pv > 0 and pv >= target_pv) or (delta_pv < 0 and pv <= target_pv):
                    tm_index = i
                    break
            tm = max((tm_index * 0.3), 1.0) # Assuming delay block is set to 0.3s
        else:
            print("⚠️ [SysID] Array wave too small. Using scaled fallback plant defaults.")

    # --- 2. ADAPTIVE TUNING RULES ---
    decision = auto_operator.decide(km, tm, taum)
    kc, ti, rule_key, rule_name, rule_desc, chart, os_est, settling = run_tuning(
        km, tm, taum, decision["mode"], decision["overshoot"], 
        decision["robust"], decision["metric"], order=1, D=D_val
    )
    
    # --- 3. EXACT DISTURBANCE SCALING ---
    if D_val > 0:
        # FIX 1: Tame the Proportional Boost. 
        # Lowered the multiplier from 150.0 to 15.0
        empirical_boost = 1.0 + (15.0 * D_val) 
        kc = kc * empirical_boost
        
        # FIX 2: Relax the Integral Action.
        # Increased the shielding multiplier from 0.8 to 1.5. 
        # A larger Ti means a SMALLER Ki, which stops the violent undershoot.
        ti = max(tm * 1.5, 10.0) 
        
        kc = min(kc, 80000.0) # Lowered the safety ceiling
        rule_name = f"Adaptive Recovery (Km: {round(km,2)}, Tm: {round(tm,2)}s, Boost: {round(empirical_boost, 1)}x)"

    # Broadcast to Web Dashboard so the chart still draws
    socketio.emit('tune_response', {
        "status": "ok", "rule": rule_name, "kc": round(kc, 6), "ti": round(ti, 6),
        "os_predicted": os_est, "settling_time": settling,
        "fopdt": {"km": round(km,3), "tm": round(tm,3), "taum": round(taum,3)}, "chart": chart
    })
    
    # Return the newly identified km and tm so the Python bridge can print them!
    return jsonify({
        "kc": kc, 
        "ti": ti,
        "km": round(km, 3),
        "tm": round(tm, 3)
    })

@app.route('/api/telemetry', methods=['POST'])
def api_telemetry_fallback():
    # Forward MATLAB's HTTP telemetry directly to the WebSocket dashboard
    socketio.emit('telemetry_update', request.json or {})
    return jsonify({"status": "ok"})


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
