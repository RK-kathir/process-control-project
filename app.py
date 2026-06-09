import os, re, json, pickle, math, traceback
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'tuningbot-secret')
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ── Gemini ────────────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# ── Load assets ───────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_dir, 'tuning_rules.json')) as f:
        rules_db = json.load(f)
    with open(os.path.join(current_dir, 'ai_brain.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    print(f"SUCCESS: {len(rules_db)} rules, AI Brain loaded.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}

# ══════════════════════════════════════════════════════════════════════════════
#  HALF-RULE REDUCER  (Skogestad's method)
# ══════════════════════════════════════════════════════════════════════════════
class HalfRuleReducer:
    """
    Converts higher-order FOPDT/SOPDT to a 1st-order FOPDT approximation
    using Skogestad's Half-Rule:
      - Largest neglected time constant → half added to dead time
      - All smaller neglected time constants → added in full to dead time
    """

    @staticmethod
    def from_fopdt(km, tm, taum):
        """Already FOPDT — pass through unchanged."""
        return km, tm, taum

    @staticmethod
    def from_sopdt(km, tm1, tm2, taum):
        """
        Second-order: G(s) = km / ((tm1*s+1)(tm2*s+1)) * e^(-taum*s)
        Half-rule: keep tm1, add tm2/2 + taum to dead time.
        """
        tm_approx   = tm1
        taum_approx = taum + tm2 / 2.0
        return km, tm_approx, taum_approx

    @staticmethod
    def from_third_order(km, tm1, tm2, tm3, taum):
        """
        Third-order: G(s) = km / ((tm1*s+1)(tm2*s+1)(tm3*s+1)) * e^(-taum*s)
        Assume tm1 >= tm2 >= tm3.
        Half-rule: keep tm1, add tm2/2 + tm3 + taum to dead time.
        """
        taus = sorted([tm1, tm2, tm3], reverse=True)
        tm_approx   = taus[0]
        taum_approx = taum + taus[1] / 2.0 + taus[2]
        return km, tm_approx, taum_approx

    @staticmethod
    def reduce(order, km, time_constants, taum):
        """
        Dispatcher. time_constants is a list, largest first.
        Returns (km, tm_approx, taum_approx).
        """
        if order == 1 or len(time_constants) == 0:
            return km, time_constants[0] if time_constants else 1.0, taum
        elif order == 2:
            return HalfRuleReducer.from_sopdt(km, time_constants[0], time_constants[1], taum)
        elif order >= 3:
            return HalfRuleReducer.from_third_order(
                km,
                time_constants[0], time_constants[1], time_constants[2],
                taum
            )
        return km, time_constants[0], taum


# ══════════════════════════════════════════════════════════════════════════════
#  AUTONOMOUS OPERATOR  (replaces interview in MATLAB mode)
# ══════════════════════════════════════════════════════════════════════════════
class AutonomousOperator:
    """
    Evaluates incoming process parameters and decides tuning strategy without
    human input. Mimics a skilled process control engineer.
    """

    # Thresholds
    HIGH_RATIO_THRESHOLD  = 0.5   # tau/T above this → consider robust
    SPIKE_RATIO_THRESHOLD = 0.7   # tau/T above this → force ultra-safe
    GAIN_CHANGE_THRESHOLD = 0.30  # >30% Km change between calls → re-tune
    RATIO_SPIKE_DELTA     = 0.20  # ratio increased by this much → spike detected

    def __init__(self):
        self.prev_km    = None
        self.prev_ratio = None

    def decide(self, km, tm, taum):
        """
        Returns dict: {mode, overshoot, robust, metric, reason}
        """
        ratio   = taum / max(tm, 1e-6)
        reasons = []

        # ── Spike / instability detection ─────────────────────────────────
        ratio_spiked = (
            self.prev_ratio is not None and
            (ratio - self.prev_ratio) > self.RATIO_SPIKE_DELTA
        )
        gain_shifted = (
            self.prev_km is not None and
            abs(km - self.prev_km) / max(abs(self.prev_km), 1e-6) > self.GAIN_CHANGE_THRESHOLD
        )

        # ── Decision tree ──────────────────────────────────────────────────
        if ratio >= self.SPIKE_RATIO_THRESHOLD or ratio_spiked:
            mode, overshoot, robust, metric = 0, 0, 1, 1  # regulatory, no OS, robust, IAE
            reasons.append(f"⚠️ High dead-time ratio ({ratio:.2f}) or ratio spike detected → ultra-safe robust mode")

        elif gain_shifted:
            mode, overshoot, robust, metric = 0, 1, 1, 1  # regulatory, low OS, robust, IAE
            reasons.append(f"⚠️ Process gain shifted {round(abs(km - self.prev_km)/max(abs(self.prev_km),1e-6)*100)}% → robust re-tune")

        elif ratio >= self.HIGH_RATIO_THRESHOLD:
            mode, overshoot, robust, metric = 0, 1, 0, 3  # regulatory, low OS, normal, ITAE
            reasons.append(f"Moderate dead-time ratio ({ratio:.2f}) → ITAE regulatory tuning")

        else:
            mode, overshoot, robust, metric = 0, 2, 0, 2  # regulatory, medium OS, performance, ISE
            reasons.append(f"Low dead-time ratio ({ratio:.2f}) → aggressive ISE regulatory tuning")

        # ── Update memory ──────────────────────────────────────────────────
        self.prev_km    = km
        self.prev_ratio = ratio

        return {
            "mode": mode, "overshoot": overshoot,
            "robust": robust, "metric": metric,
            "reason": " | ".join(reasons)
        }


# ══════════════════════════════════════════════════════════════════════════════
#  NLP INTENT MODEL  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
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
    ("what is ziegler nichols","rule_explain"),("explain cohen coon","rule_explain"),
    ("what is imc","rule_explain"),("what is lambda tuning","rule_explain"),
    ("what is iae","metric_explain"),("what is ise","metric_explain"),
    ("what is itae","metric_explain"),("explain performance metrics","metric_explain"),
    ("help","help"),("how do i use this","help"),("guide me","help"),
    ("what should i do","help"),("where do i start","help"),
]
texts, labels = zip(*nlp_training_data)
intent_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_nlp = intent_vectorizer.fit_transform(texts)
intent_model = LinearSVC()
intent_model.fit(X_nlp, labels)

KNOWLEDGE_BASE = {
    "identity": (
        "<strong>I am TUNING BOT</strong> — an AI-powered PID controller optimization engine.<br><br>"
        "I use a <strong>Random Forest AI model</strong> trained on O'Dwyer's handbook. "
        "Give me your process parameters, answer a few guided questions, and I will prescribe the optimal PID settings "
        "and simulate the closed-loop response.<br><br>"
        "I also support <strong>Simulink Telemetry Mode</strong> — connect MATLAB directly via WebSocket "
        "for autonomous real-time tuning of your ANFIS controller."
    ),
    "pid_explain": (
        "<strong>PID stands for Proportional-Integral-Derivative.</strong><br><br>"
        "- <strong>Proportional (Kc):</strong> Reacts to current error<br>"
        "- <strong>Integral (Ti):</strong> Corrects accumulated past errors<br>"
        "- <strong>Derivative (Td):</strong> Predicts future error<br><br>"
        "Getting these three numbers right is what I do."
    ),
    "param_explain": (
        "FOPDT = <strong>First Order Plus Dead Time</strong>:<br><br>"
        "- <strong>Km:</strong> Process gain — output change per unit input<br>"
        "- <strong>Tm:</strong> Time constant — speed of response (~63% of final value)<br>"
        "- <strong>Tau:</strong> Dead time — pure delay before any response<br><br>"
        "Obtain these from a step test. I also accept 2nd and 3rd-order transfer functions "
        "and will auto-reduce them using Skogestad's Half-Rule."
    ),
    "rule_explain": (
        "I have <strong>50+ tuning rules</strong> from O'Dwyer's handbook:<br><br>"
        "- <strong>Ziegler-Nichols:</strong> Classic aggressive baseline<br>"
        "- <strong>Cohen-Coon:</strong> For high dead-time processes<br>"
        "- <strong>Rivera IMC / Chien IMC:</strong> Robust, safety-critical<br>"
        "- <strong>Rovira / Murrill:</strong> Minimise IAE/ISE/ITAE integrals<br>"
        "- <strong>Miluse et al.:</strong> Exact overshoot specification 0–50%<br>"
        "- <strong>Hang et al.:</strong> Explicit gain/phase margin targets"
    ),
    "metric_explain": (
        "Performance metrics measure accumulated error — lower is better:<br><br>"
        "- <strong>IAE:</strong> Equal weight on all errors — smooth, moderate<br>"
        "- <strong>ISE:</strong> Penalises large errors heavily — fast, aggressive<br>"
        "- <strong>ITAE:</strong> Penalises persistent late errors — best long-term settling"
    ),
    "help": (
        "<strong>Human Chat Mode:</strong><br>"
        "Provide <code>Km=x, Tm=x, Tau=x</code> and answer 4 guided questions.<br><br>"
        "<strong>Simulink Telemetry Mode (click the toggle):</strong><br>"
        "Connect MATLAB via WebSocket to <code>ws://your-server/socket.io</code>.<br>"
        "See MATLAB integration guide below for the exact commands."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "metric": None, "robust": None, "overshoot": None,
    "interview_stage": 0, "allows_overshoot": False, "overshoot_answer": None
}

auto_operator = AutonomousOperator()

def _reset_memory():
    global bot_memory
    bot_memory = {
        "km": None, "tm": None, "taum": None, "tau_c": None,
        "mode": None, "metric": None, "robust": None, "overshoot": None,
        "interview_stage": 0, "allows_overshoot": False, "overshoot_answer": None
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_step(kc, ti, km, tm, taum):
    t  = np.linspace(0, (tm + taum) * 8, 500)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum  = 0
    d_steps  = max(1, int(taum / dt))
    ti_val   = ti if ti > 0 else 0.001

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

    settling_time = t[-1]
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
        ],
        "layout": {
            "title": {"text": "Closed-Loop Step Response", "font": {"size": 14}},
            "xaxis": {"title": "Time (s)", "gridcolor": "#1e2a3a", "zerolinecolor": "#333"},
            "yaxis": {"title": "Normalised PV", "gridcolor": "#1e2a3a", "zerolinecolor": "#333"},
            "paper_bgcolor": "transparent", "plot_bgcolor": "rgba(10,14,20,0.6)",
            "font": {"color": "#a0aec0"},
            "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(255,255,255,0.1)", "borderwidth": 1},
            "margin": {"l": 50, "r": 20, "t": 45, "b": 45},
            "hovermode": "x unified"
        }
    }
    return json.dumps(graph_data), os_val, settling_time


# ══════════════════════════════════════════════════════════════════════════════
#  CORE TUNING ENGINE  (shared by REST and WebSocket paths)
# ══════════════════════════════════════════════════════════════════════════════
def run_tuning(km, tm, taum, mode, overshoot, robust, metric, overshoot_answer=None):
    """
    Returns (kc, ti, rule_name, description, chart_json, os_est, settling).
    """
    OVERSHOOT_RULE_OVERRIDE = {
        "os_5": "miluse_5os", "os_10": "miluse_10os",
        "os_20": "miluse_20os", "os_30": "miluse_30os"
    }

    ratio    = taum / max(tm, 1e-6)
    features = np.array([[km, tm, taum, ratio, mode, overshoot, robust, metric]])

    best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"

    if overshoot_answer and overshoot_answer in OVERSHOOT_RULE_OVERRIDE:
        override_key = OVERSHOOT_RULE_OVERRIDE[overshoot_answer]
        if override_key in rules_db:
            best_rule = override_key

    r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

    tau_c_val = max(0.1, taum)
    safe_env = {
        "km": km, "tm": tm, "taum": taum, "tau_c": tau_c_val,
        "min": min, "max": max, "exp": np.exp, "log": np.log,
        "sqrt": np.sqrt, "np": np, "math": math
    }

    try:
        if r.get('kc_math') == "SPECIAL_LOOKUP":
            kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
        else:
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
            if not ti or ti <= 0: ti = 0.001
    except Exception as e:
        print(f"Math eval error: {e}")
        kc = (1.2 * tm) / (km * taum)
        ti = 2.0 * taum
        r  = {"name": "Ziegler-Nichols (Fallback)"}

    chart, os_est, settling = simulate_step(kc, ti, km, tm, taum)
    return kc, ti, r.get('name', best_rule), r.get('unique_feature', ''), chart, os_est, settling


# ══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET EVENTS  (MATLAB/Simulink interface)
# ══════════════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")
    emit('status', {'message': 'TUNING BOT connected. Ready for telemetry.'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")


@socketio.on('telemetry')
def handle_telemetry(data):
    """
    Receive live simulation data from MATLAB for dashboard display.
    Expected payload:
        { "t": [...], "pv": [...], "sp": [...], "mv": [...] }
    Broadcasts to all connected frontend clients.
    """
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
    """
    Autonomous tuning request from MATLAB.

    FOPDT payload:
        { "order": 1, "km": 2.0, "tm": 10.0, "taum": 2.0 }

    SOPDT payload:
        { "order": 2, "km": 2.0, "tm1": 10.0, "tm2": 3.0, "taum": 1.0 }

    Third-order payload:
        { "order": 3, "km": 2.0, "tm1": 10.0, "tm2": 3.0, "tm3": 1.0, "taum": 0.5 }
    """
    try:
        order = int(data.get("order", 1))
        km    = float(data.get("km",   1.0))
        taum  = float(data.get("taum", 1.0))

        # Build time-constants list
        if order == 1:
            tcs = [float(data.get("tm", 10.0))]
        elif order == 2:
            tcs = sorted([float(data.get("tm1", 10.0)),
                          float(data.get("tm2",  3.0))], reverse=True)
        else:
            tcs = sorted([float(data.get("tm1", 10.0)),
                          float(data.get("tm2",  3.0)),
                          float(data.get("tm3",  1.0))], reverse=True)

        # ── Half-rule reduction ────────────────────────────────────────────
        km_r, tm_r, taum_r = HalfRuleReducer.reduce(order, km, tcs, taum)

        if order > 1:
            reduction_note = (
                f"Order-{order} TF reduced via Skogestad Half-Rule → "
                f"Km={round(km_r,4)}, Tm={round(tm_r,4)}, Tau={round(taum_r,4)}"
            )
        else:
            reduction_note = f"FOPDT: Km={km_r}, Tm={tm_r}, Tau={taum_r}"

        # ── Autonomous operator decision ───────────────────────────────────
        decision = auto_operator.decide(km_r, tm_r, taum_r)

        # ── Run tuning engine ──────────────────────────────────────────────
        kc, ti, rule_name, rule_desc, chart, os_est, settling = run_tuning(
            km_r, tm_r, taum_r,
            decision["mode"], decision["overshoot"],
            decision["robust"], decision["metric"]
        )

        response = {
            "status": "ok",
            "reduction_note": reduction_note,
            "operator_decision": decision["reason"],
            "rule": rule_name,
            "kc": round(kc, 6),
            "ti": round(ti, 6),
            "os_predicted": os_est,
            "settling_time": settling,
            "fopdt": {"km": round(km_r,4), "tm": round(tm_r,4), "taum": round(taum_r,4)},
        }
        emit('tune_response', response)
        print(f"[WS] Tuned: Kc={kc:.4f}, Ti={ti:.4f} | {decision['reason']}")

    except Exception as e:
        print(traceback.format_exc())
        emit('tune_response', {'status': 'error', 'message': str(e)})


MODE_MAP   = {"regulator": 0, "servo": 1}
OS_MAP     = {"none": 0, "low": 1, "medium": 2, "high": 3}
METRIC_MAP = {"none": 0, "iae": 1, "ise": 2, "itae": 3, "istse": 4, "istes": 5}

INTERVIEW = {
    1: {
        "text": (
            "Parameters confirmed. Let me understand your control objective.<br><br>"
            "<strong>Question 1 of 4 — What is this control loop doing?</strong>"
        ),
        "options": [
            {"label": "Following a target — setpoint tracking", "val": "servo",
             "hint": "e.g. robot arm, temperature profile, flow setpoint"},
            {"label": "Holding steady — rejecting external disturbances", "val": "regulator",
             "hint": "e.g. pressure vessel, boiler level, conveyor speed"}
        ],
        "map": {"servo": {"mode": 1}, "regulator": {"mode": 0}}
    },
    2: {
        "text": "<strong>Question 2 of 4 — Response behaviour?</strong>",
        "options": [
            {"label": "Maximum speed — overshoot acceptable", "val": "fast",   "hint": "ISE — fast, aggressive"},
            {"label": "Smooth approach — no overshoot",        "val": "smooth", "hint": "IAE — balanced"},
            {"label": "Balanced — fast but clean settling",    "val": "balanced","hint": "ITAE — best long-term"}
        ],
        "map": {
            "fast":     {"metric": 2, "overshoot": 2, "robust": 0},
            "smooth":   {"metric": 1, "overshoot": 0, "robust": 1},
            "balanced": {"metric": 3, "overshoot": 1, "robust": 0}
        }
    },
    3: {
        "text": "<strong>Question 3 of 4 — How confident are you in your model?</strong>",
        "options": [
            {"label": "Very confident — careful step test", "val": "confident", "hint": "Full performance"},
            {"label": "Approximately correct — estimated",  "val": "estimated", "hint": "Moderate detuning"},
            {"label": "Uncertain — rough estimate",         "val": "uncertain", "hint": "Robust, safety margin"}
        ],
        "map": {
            "confident": {"robust": 0},
            "estimated": {"robust": 0},
            "uncertain": {"robust": 1}
        }
    },
    4: {
        "text": "<strong>Question 4 of 4 — Maximum acceptable overshoot?</strong>",
        "options": [
            {"label": "Up to 5%",  "val": "os_5",  "hint": "Miluse 5%"},
            {"label": "Up to 10%", "val": "os_10", "hint": "Miluse 10%"},
            {"label": "Up to 20%", "val": "os_20", "hint": "Standard industrial"},
            {"label": "Up to 30%+","val": "os_30", "hint": "Maximum speed"}
        ],
        "map": {
            "os_5": {"overshoot": 1}, "os_10": {"overshoot": 1},
            "os_20": {"overshoot": 2}, "os_30": {"overshoot": 3}
        }
    }
}

INTERVIEW_ANSWER_PATTERNS = {
    1: {"servo": ["servo","setpoint","track","follow","target","change","position"],
        "regulator": ["regulator","disturbance","hold","steady","reject","fixed","constant"]},
    2: {"fast":     ["fast","quick","aggressive","speed","ise","overshoot"],
        "smooth":   ["smooth","safe","slow","no overshoot","gentle","iae","soft"],
        "balanced": ["balance","balanced","standard","normal","itae","moderate","typical"]},
    3: {"confident": ["confident","accurate","careful","step test","measured","certain"],
        "estimated": ["estimated","roughly","approximate","historical","quick"],
        "uncertain": ["uncertain","unsure","rough","guess","varies","not sure","unknown"]},
    4: {"os_5":  ["5","five","5%","nearly","almost none"],
        "os_10": ["10","ten","10%","moderate","well damped"],
        "os_20": ["20","twenty","20%","standard","industrial","typical"],
        "os_30": ["30","thirty","30%","maximum","aggressive","fast"]}
}

def _parse_interview_answer(stage, msg_lower):
    for ans, keywords in INTERVIEW_ANSWER_PATTERNS.get(stage, {}).items():
        if any(kw in msg_lower for kw in keywords):
            return ans
    return None

def _local_regex_extract(msg):
    ext = {"km": None, "tm": None, "taum": None}
    msg_lower = msg.lower()
    km_m  = re.search(r'(km|gain|process\s*gain)\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    tm_m  = re.search(r'(tm|lag|time\s*constant)\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    tau_m = re.search(r'(tau|dead\s*time|delay|theta)\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    if km_m:  ext["km"]   = float(km_m.group(3))
    if tm_m:  ext["tm"]   = float(tm_m.group(3))
    if tau_m: ext["taum"] = float(tau_m.group(3))
    return ext

def _build_rules_response():
    cats = {"Servo / Setpoint Tracking": [], "Regulatory / Disturbance Rejection": []}
    for k, v in rules_db.items():
        if v.get("mode") in (1, "servo"):
            cats["Servo / Setpoint Tracking"].append(v.get("name", k))
        else:
            cats["Regulatory / Disturbance Rejection"].append(v.get("name", k))
    reply = f"<strong>Tuning database: {len(rules_db)} rules</strong> from O'Dwyer's handbook:<br><br>"
    n = 1
    for cat, rl in cats.items():
        if rl:
            reply += f"<strong>{cat}</strong><br>"
            for r in rl:
                reply += f"{n}. {r}<br>"
                n += 1
            reply += "<br>"
    reply += "<em>The Random Forest AI selects the optimal rule for your process and objectives.</em>"
    return jsonify({"reply": reply, "options": [], "chart": None})

def _run_rest_tuning():
    global bot_memory
    km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
    mode = bot_memory.get('mode', 1)
    os_v = bot_memory.get('overshoot', 0)
    rob  = bot_memory.get('robust', 0)
    met  = bot_memory.get('metric', 1)
    oa   = bot_memory.get('overshoot_answer', None)

    kc, ti, rule_name, rule_desc, chart, os_est, settling = run_tuning(
        km, tm, taum, mode, os_v, rob, met, oa
    )

    ratio      = taum / max(tm, 1e-6)
    mode_str   = "Servo (Setpoint Tracking)" if mode == 1 else "Regulatory (Disturbance Rejection)"
    metric_map = {"1": "IAE — smooth", "2": "ISE — aggressive", "3": "ITAE — balanced"}
    metric_str = metric_map.get(str(met), "IAE")
    robust_str = "Robust" if rob == 1 else "Performance-optimized"

    final_reply = (
        f"<strong>Optimization Complete</strong><br><br>"
        f"<strong>Process Analysis:</strong><br>"
        f"- Mode: {mode_str}<br>"
        f"- Performance objective: {metric_str}<br>"
        f"- Tuning philosophy: {robust_str}<br>"
        f"- Dead-time ratio (tau/T): {round(ratio, 3)}<br><br>"
        f"<strong>AI Selection:</strong> <strong>{rule_name}</strong><br>"
        f"<em>{rule_desc}</em><br><br>"
        f"<strong>PID Parameters:</strong><br>"
        f"- Proportional Gain <strong>Kc = {round(kc, 4)}</strong><br>"
        f"- Integral Time <strong>Ti = {round(ti, 4)} s</strong><br><br>"
        f"<strong>Predicted Performance:</strong><br>"
        f"- Overshoot: <strong>{os_est}%</strong><br>"
        f"- Settling Time: <strong>{settling} s</strong><br><br>"
        f"Step response simulation below."
    )
    _reset_memory()
    return jsonify({"reply": final_reply, "chart": chart, "options": []})


@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg       = request.json.get('message', '')
        user_msg_lower = user_msg.lower().strip()

        if user_msg_lower == "reset":
            _reset_memory()
            return jsonify({
                "reply": (
                    "Session reset. Ready for a new process.<br><br>"
                    "Provide your FOPDT parameters: <code>Km=2, Tm=10, Tau=2</code>"
                ),
                "options": [], "chart": None
            })

        has_digits = any(ch.isdigit() for ch in user_msg_lower)
        word_count = len(user_msg_lower.split())

        # ── NLP intent fast-path ───────────────────────────────────────────
        if word_count < 12 and not has_digits:
            try:
                vec          = intent_vectorizer.transform([user_msg_lower])
                local_intent = intent_model.predict(vec)[0]
                confidence   = max(intent_model.decision_function(vec)[0])
            except:
                local_intent, confidence = "none", 0

            if local_intent == "greeting" and confidence > 0.3:
                return jsonify({
                    "reply": (
                        "Hello. I am <strong>TUNING BOT</strong>.<br><br>"
                        "Provide process parameters: <code>Km=value, Tm=value, Tau=value</code><br>"
                        "Or switch to <strong>Simulink Telemetry Mode</strong> to connect MATLAB directly."
                    ),
                    "options": [], "chart": None
                })
            if local_intent in KNOWLEDGE_BASE and confidence > 0.2:
                return jsonify({"reply": KNOWLEDGE_BASE[local_intent], "options": [], "chart": None})
            if local_intent == "rules" and confidence > 0.2:
                return _build_rules_response()

        if any(kw in user_msg_lower for kw in ["what rules","show rules","list rules","how many rules","rules do you"]):
            return _build_rules_response()

    
        regex_data = _local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]) and has_digits:
            try:
                prompt = (
                    f'Extract FOPDT params from: "{user_msg}". '
                    f'Current: km={bot_memory["km"]}, tm={bot_memory["tm"]}, taum={bot_memory["taum"]}. '
                    'OUTPUT JSON ONLY: {"km": float_or_null, "tm": float_or_null, "taum": float_or_null}'
                )
                res = llm_model.generate_content(prompt)
                ext = json.loads(res.text.replace('```json','').replace('```','').strip())
                for k in ["km", "tm", "taum"]:
                    if ext.get(k) is not None: bot_memory[k] = ext[k]
            except: pass

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        missing = []
        if not km:   missing.append("<strong>Km</strong>")
        if not tm:   missing.append("<strong>Tm</strong>")
        if not taum: missing.append("<strong>Tau</strong>")

        if missing:
            return jsonify({
                "reply": (
                    "To tune your PID controller I need: " + ", ".join(missing) + "<br><br>"
                    "Format: <code>Km=2, Tm=10, Tau=2</code>"
                ),
                "options": [], "chart": None
            })

        stage = bot_memory['interview_stage']

        if stage > 0:
            interview_q = INTERVIEW[stage]
            answer = None
            for opt in interview_q["options"]:
                if user_msg_lower == opt["val"]: answer = opt["val"]; break
            if not answer:
                answer = _parse_interview_answer(stage, user_msg_lower)

            if answer:
                bot_memory.update(interview_q["map"][answer])
                if stage == 2:
                    bot_memory['allows_overshoot'] = (answer == "fast")
                if stage == 4:
                    bot_memory['overshoot_answer'] = answer

                bot_memory['interview_stage'] += 1
                stage = bot_memory['interview_stage']

                if stage == 4 and not bot_memory.get('allows_overshoot', False):
                    bot_memory['interview_stage'] = 5
                    return _run_rest_tuning()
                if stage <= 4:
                    next_q = INTERVIEW[stage]
                    return jsonify({
                        "reply": next_q["text"],
                        "options": [{"label": o["label"], "val": o["val"]} for o in next_q["options"]],
                        "chart": None
                    })
                else:
                    return _run_rest_tuning()
            else:
                return jsonify({
                    "reply": f"I did not recognise that. {interview_q['text']}",
                    "options": [{"label": o["label"], "val": o["val"]} for o in interview_q["options"]],
                    "chart": None
                })

        if stage == 0:
            bot_memory['interview_stage'] = 1
            ratio = taum / max(tm, 1e-6)
            if   ratio < 0.2: ratio_note = f" Dead-time ratio {round(ratio,2)} — <strong>low</strong>, highly controllable."
            elif ratio < 0.5: ratio_note = f" Dead-time ratio {round(ratio,2)} — <strong>moderate</strong>."
            else:              ratio_note = f" Dead-time ratio {round(ratio,2)} — <strong>high</strong>, dead-time compensating rules will apply."

            q = INTERVIEW[1]
            return jsonify({
                "reply": (
                    f"Parameters confirmed: <strong>Km={km}, Tm={tm}s, Tau={taum}s</strong>.{ratio_note}<br><br>"
                    f"{q['text']}"
                ),
                "options": [{"label": o["label"], "val": o["val"]} for o in q["options"]],
                "chart": None
            })

        return jsonify({"reply": "State error. Please reset.", "options": [], "chart": None})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"reply": "System error. Please reset.", "options": []})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
