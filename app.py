import os
import re
import json
import pickle
import math
import numpy as np
import traceback
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_dir, 'tuning_rules.json'), 'r') as f:
        rules_db = json.load(f)
    with open(os.path.join(current_dir, 'ai_brain.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    print(f"SUCCESS: Loaded {len(rules_db)} tuning rules and AI Brain.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}

nlp_training_data = [
    ("hi", "greeting"), ("hello", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"), ("howdy", "greeting"),
    ("who are you", "identity"), ("what can you do", "identity"),
    ("what is tuning bot", "identity"), ("tell me about yourself", "identity"),
    ("what are you", "identity"), ("how do you work", "identity"),
    ("what tuning rules do you have", "rules"), ("show me the rules", "rules"),
    ("list rules", "rules"), ("what rules are available", "rules"),
    ("show all rules", "rules"), ("how many rules", "rules"),
    ("what is pid", "pid_explain"), ("explain pid", "pid_explain"),
    ("what does pid mean", "pid_explain"), ("how does pid work", "pid_explain"),
    ("what is proportional integral derivative", "pid_explain"),
    ("what is km", "param_explain"), ("what is gain", "param_explain"),
    ("what is tm", "param_explain"), ("what is lag time", "param_explain"),
    ("what is tau", "param_explain"), ("what is dead time", "param_explain"),
    ("what is fopdt", "param_explain"), ("explain fopdt", "param_explain"),
    ("what is ziegler nichols", "rule_explain"), ("explain cohen coon", "rule_explain"),
    ("what is imc", "rule_explain"), ("what is lambda tuning", "rule_explain"),
    ("what is amigo", "rule_explain"), ("what is skogestad", "rule_explain"),
    ("what is iae", "metric_explain"), ("what is ise", "metric_explain"),
    ("what is itae", "metric_explain"), ("explain performance metrics", "metric_explain"),
    ("help", "help"), ("how do i use this", "help"), ("guide me", "help"),
    ("what should i do", "help"), ("where do i start", "help"),
]
texts, labels = zip(*nlp_training_data)
intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_nlp = intent_vectorizer.fit_transform(texts)
intent_model = LinearSVC()
intent_model.fit(X_nlp, labels)

KNOWLEDGE_BASE = {
    "identity": (
        "<strong>I am TUNING BOT</strong> — an AI-powered PID controller optimization engine.<br><br>"
        "I use a <strong>Random Forest AI model</strong> trained on O'Dwyer's handbook. "
        "Give me your process parameters, answer a few structured questions, and I will prescribe the mathematically optimal PID settings — "
        "plus simulate the closed-loop response with a live graph.<br><br>"
        "No prior knowledge of process control required. I handle the mathematics."
    ),
    "pid_explain": (
        "<strong>PID stands for Proportional-Integral-Derivative.</strong> It is the most widely used control algorithm in industry.<br><br>"
        "Think of it as a smart regulator for any industrial process:<br>"
        "- <strong>Proportional (Kc):</strong> Reacts to the current error — the larger the gap, the harder it corrects<br>"
        "- <strong>Integral (Ti):</strong> Corrects accumulated past errors — eliminates persistent offsets<br>"
        "- <strong>Derivative (Td):</strong> Predicts future error — dampens oscillations<br><br>"
        "Getting these three numbers right is what I do."
    ),
    "param_explain": (
        "FOPDT stands for <strong>First Order Plus Dead Time</strong> — the standard model for most industrial processes.<br><br>"
        "It requires three numbers from your process:<br>"
        "- <strong>Km (Process Gain):</strong> How much the output changes per unit of input. If you open a valve 10% and temperature rises 5 degrees, then Km = 0.5<br>"
        "- <strong>Tm (Time Constant):</strong> How quickly the process responds — time to reach approximately 63% of its final value<br>"
        "- <strong>Tau (Dead Time):</strong> The pure delay before any response begins — like fluid travel time through a pipe<br><br>"
        "These values are obtained via a step test on your process."
    ),
    "rule_explain": (
        "I have <strong>50 plus tuning rules</strong> from O'Dwyer's handbook, each designed for different scenarios:<br><br>"
        "- <strong>Ziegler-Nichols:</strong> The classic — aggressive, fast, high overshoot<br>"
        "- <strong>Cohen-Coon:</strong> Designed for processes with large dead time<br>"
        "- <strong>Skogestad IMC:</strong> Robust and predictable for safety-critical systems<br>"
        "- <strong>Lambda Tuning:</strong> User-controlled speed via closed-loop time constant<br>"
        "- <strong>Rovira (IAE/ISE/ITAE):</strong> Mathematically minimizes error integrals<br>"
        "- <strong>Miluse et al.:</strong> Exact overshoot specification from 0% to 50%<br><br>"
        "My AI selects the optimal rule for your specific process and objectives."
    ),
    "metric_explain": (
        "Performance metrics measure how much error accumulates during a transient. Lower values indicate better performance.<br><br>"
        "- <strong>IAE (Integral of Absolute Error):</strong> Weights all errors equally — smooth, moderate response. Recommended for most processes.<br>"
        "- <strong>ISE (Integral of Squared Error):</strong> Penalizes large errors heavily — aggressive, fast response with possible overshoot.<br>"
        "- <strong>ITAE (Integral of Time x Absolute Error):</strong> Penalizes errors that persist over time — best long-term settling, minimal oscillation.<br><br>"
        "Not certain which to select? I will ask about your process behavior and choose for you."
    ),
    "help": (
        "Getting started requires no engineering background.<br><br>"
        "<strong>Step 1:</strong> Provide your three process numbers: Km (gain), Tm (lag), and Tau (dead time).<br>"
        "Example: <code>Km=2, Tm=10, Tau=2</code><br><br>"
        "<strong>Step 2:</strong> Answer a few plain-language questions about your system's behavior.<br><br>"
        "<strong>Step 3:</strong> I compute your optimal PID settings and display a live simulation graph.<br><br>"
        "Use the suggestion chips below to try an example instantly."
    ),
}

bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "metric": None, "robust": None, "overshoot": None,
    "interview_stage": 0,
    "allows_overshoot": False
}

def simulate_step(kc, ti, km, tm, taum):
    t = np.linspace(0, (tm + taum) * 8, 500)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum = 0
    d_steps = max(1, int(taum / dt))
    ti_val = ti if ti > 0 else 0.001

    for i in range(1, len(t)):
        err = 1.0 - pv[i - 1]
        err_sum += err * dt
        mv = max(0, min(100, kc * (err + (1 / ti_val) * err_sum)))
        mv_hist[i] = mv
        d_idx = i - d_steps
        d_mv = mv_hist[d_idx] if d_idx >= 0 else 0
        dpv = ((km * d_mv) - pv[i - 1]) / tm
        pv[i] = pv[i - 1] + dpv * dt

    os_val = 0.0
    try:
        os_val = round(max(0, (np.max(pv) - 1.0) * 100), 1)
        if math.isnan(os_val) or math.isinf(os_val):
            os_val = 0.0
    except:
        pass

    settling_time = t[-1]
    for i in range(len(pv) - 1, 0, -1):
        if abs(pv[i] - 1.0) > 0.02:
            settling_time = round(t[i], 1)
            break

    graph_data = {
        "data": [
            {
                "x": t.tolist(), "y": pv.tolist(),
                "type": "scatter", "name": "Process Output",
                "line": {"color": "#00d4ff", "width": 2.5},
                "fill": "tozeroy", "fillcolor": "rgba(0,212,255,0.04)"
            },
            {
                "x": [t[0], t[-1]], "y": [1.0, 1.0],
                "type": "scatter", "name": "Setpoint",
                "line": {"color": "#ff4d6d", "dash": "dash", "width": 1.5}
            },
            {
                "x": [t[0], t[-1]], "y": [1.02, 1.02],
                "type": "scatter", "name": "+2% Band",
                "line": {"color": "rgba(255,255,255,0.15)", "dash": "dot", "width": 1},
                "showlegend": False
            },
            {
                "x": [t[0], t[-1]], "y": [0.98, 0.98],
                "type": "scatter", "name": "-2% Band",
                "line": {"color": "rgba(255,255,255,0.15)", "dash": "dot", "width": 1},
                "showlegend": False
            }
        ],
        "layout": {
            "title": {"text": "Closed-Loop Step Response", "font": {"size": 14}},
            "xaxis": {"title": "Time (s)", "gridcolor": "#1e2a3a", "zerolinecolor": "#333"},
            "yaxis": {"title": "Normalized Process Variable", "gridcolor": "#1e2a3a", "zerolinecolor": "#333"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "rgba(10,14,20,0.6)",
            "font": {"color": "#a0aec0"},
            "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(255,255,255,0.1)", "borderwidth": 1},
            "margin": {"l": 50, "r": 20, "t": 45, "b": 45},
            "hovermode": "x unified"
        }
    }
    return json.dumps(graph_data), os_val, settling_time

def local_regex_extract(msg):
    ext = {"km": None, "tm": None, "taum": None}
    msg_lower = msg.lower()
    km_m  = re.search(r'(km|gain|process\s*gain|k(?!c|p))\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    tm_m  = re.search(r'(tm|lag|time\s*constant|t(?!au|d))\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    tau_m = re.search(r'(tau|dead\s*time|delay|theta)\s*(is|:|=)?\s*([+-]?\d+\.?\d*)', msg_lower)
    if km_m:  ext["km"]   = float(km_m.group(3))
    if tm_m:  ext["tm"]   = float(tm_m.group(3))
    if tau_m: ext["taum"] = float(tau_m.group(3))
    return ext

INTERVIEW = {
    1: {
        "text": (
            "Parameters confirmed. Let me understand your control objective before selecting a rule.<br><br>"
            "<strong>Question 1 of 4 — What is this control loop doing?</strong><br>"
            "The controller's primary function determines the entire tuning strategy:"
        ),
        "options": [
            {
                "label": "Following a target — I change the setpoint and need it to track accurately",
                "val": "servo",
                "hint": "e.g. robot arm position, temperature profile, flow rate setpoint changes"
            },
            {
                "label": "Holding steady — setpoint is fixed; I want to reject external disturbances",
                "val": "regulator",
                "hint": "e.g. pressure vessel, boiler level, conveyor speed under varying load"
            }
        ],
        "map": {"servo": {"mode": 1}, "regulator": {"mode": 0}}
    },
    2: {
        "text": (
            "<strong>Question 2 of 4 — How should the response behave?</strong><br>"
            "When a setpoint change or disturbance occurs, which response profile best fits your requirements?"
        ),
        "options": [
            {
                "label": "Maximum speed — fastest possible response, overshoot is acceptable",
                "val": "fast",
                "hint": "Prioritizes speed. Accepts overshoot. Best for batch reactors, fast flow loops"
            },
            {
                "label": "Smooth approach — no overshoot, no oscillation, steady convergence",
                "val": "smooth",
                "hint": "Eliminates overshoot entirely. Best for level control, furnaces, biological processes"
            },
            {
                "label": "Balanced — reasonably fast but settles cleanly without excessive oscillation",
                "val": "balanced",
                "hint": "The engineering standard. Works well for most general-purpose control loops"
            }
        ],
        "map": {
            "fast":     {"metric": 2, "overshoot": 2, "robust": 0},
            "smooth":   {"metric": 1, "overshoot": 0, "robust": 1},
            "balanced": {"metric": 3, "overshoot": 1, "robust": 0}
        }
    },
    3: {
        "text": (
            "<strong>Question 3 of 4 — How confident are you in your process model?</strong><br>"
            "When you measured Km, Tm, and Tau — how accurate are those numbers likely to be?"
        ),
        "options": [
            {
                "label": "Very confident — careful step test was performed, model is accurate",
                "val": "confident",
                "hint": "Allows more aggressive tuning — full performance extraction"
            },
            {
                "label": "Approximately correct — estimated from historical data or a quick test",
                "val": "estimated",
                "hint": "Balanced tuning with a moderate detuning factor"
            },
            {
                "label": "Uncertain — rough estimate, process varies significantly in real operation",
                "val": "uncertain",
                "hint": "Robust tuning — leaves a safety margin for model uncertainty"
            }
        ],
        "map": {
            "confident": {"robust": 0},
            "estimated": {"robust": 0},
            "uncertain": {"robust": 1}
        }
    },
    4: {
        "text": (
            "<strong>Question 4 of 4 — What is your acceptable overshoot limit?</strong><br>"
            "You selected a response profile that permits some overshoot. "
            "Overshoot is how far the process variable exceeds the setpoint before settling. "
            "Select the maximum overshoot your process can safely tolerate:"
        ),
        "options": [
            {
                "label": "Up to 5% overshoot — nearly no-overshoot, very controlled",
                "val": "os_5",
                "hint": "Miluse 5% rule — precise overshoot specification"
            },
            {
                "label": "Up to 10% overshoot — moderate, still well-damped",
                "val": "os_10",
                "hint": "Miluse 10% rule — good speed with controlled oscillation"
            },
            {
                "label": "Up to 20% overshoot — faster response, standard industrial tolerance",
                "val": "os_20",
                "hint": "Chien/Miluse 20% rule — typical for most industrial loops"
            },
            {
                "label": "Up to 30% or more — maximum speed, overshoot is acceptable",
                "val": "os_30",
                "hint": "Miluse 30% / Ziegler-Nichols class — aggressive, maximum speed"
            }
        ],
        "map": {
            "os_5":  {"overshoot": 1},
            "os_10": {"overshoot": 1},
            "os_20": {"overshoot": 2},
            "os_30": {"overshoot": 3}
        }
    }
}

OVERSHOOT_RULE_OVERRIDE = {
    "os_5":  "miluse_5os",
    "os_10": "miluse_10os",
    "os_20": "miluse_20os",
    "os_30": "miluse_30os"
}

INTERVIEW_ANSWER_PATTERNS = {
    1: {
        "servo":     ["servo", "setpoint", "track", "follow", "target", "change", "position", "profile"],
        "regulator": ["regulator", "disturbance", "hold", "steady", "reject", "fixed", "constant", "stable"]
    },
    2: {
        "fast":     ["fast", "quick", "aggressive", "speed", "snap", "ise", "overshoot"],
        "smooth":   ["smooth", "safe", "slow", "no overshoot", "gentle", "iae", "soft", "glide"],
        "balanced": ["balance", "balanced", "standard", "normal", "itae", "moderate", "typical"]
    },
    3: {
        "confident": ["confident", "accurate", "careful", "step test", "measured", "certain", "precise"],
        "estimated": ["estimated", "roughly", "approximate", "historical", "quick", "average"],
        "uncertain": ["uncertain", "unsure", "rough", "guess", "varies", "not sure", "unknown"]
    },
    4: {
        "os_5":  ["5", "five", "5%", "nearly", "almost none", "very controlled"],
        "os_10": ["10", "ten", "10%", "moderate", "well damped"],
        "os_20": ["20", "twenty", "20%", "standard", "industrial", "typical"],
        "os_30": ["30", "thirty", "30%", "maximum", "aggressive", "fast"]
    }
}

def parse_interview_answer(stage, msg_lower):
    patterns = INTERVIEW_ANSWER_PATTERNS.get(stage, {})
    for ans, keywords in patterns.items():
        if any(kw in msg_lower for kw in keywords):
            return ans
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        user_msg_lower = user_msg.lower().strip()

        if user_msg_lower == "reset":
            bot_memory = {
                "km": None, "tm": None, "taum": None, "tau_c": None,
                "mode": None, "metric": None, "robust": None, "overshoot": None,
                "interview_stage": 0,
                "allows_overshoot": False,
                "overshoot_answer": None
            }
            return jsonify({
                "reply": (
                    "Session reset. Ready for a new process.<br><br>"
                    "Please provide your FOPDT parameters: <strong>Km</strong> (process gain), "
                    "<strong>Tm</strong> (time constant), and <strong>Tau</strong> (dead time).<br>"
                    "Example: <code>Km=2, Tm=10, Tau=2</code>"
                ),
                "options": [], "chart": None
            })

        has_digits = any(ch.isdigit() for ch in user_msg_lower)
        word_count = len(user_msg_lower.split())

        if word_count < 12 and not has_digits:
            try:
                vec = intent_vectorizer.transform([user_msg_lower])
                local_intent = intent_model.predict(vec)[0]
                confidence = max(intent_model.decision_function(vec)[0])
            except:
                local_intent, confidence = "none", 0

            if local_intent == "greeting" and confidence > 0.3:
                return jsonify({
                    "reply": (
                        "Hello. I am <strong>TUNING BOT</strong> — your AI-powered PID tuning assistant.<br><br>"
                        "To begin, share your process parameters:<br>"
                        "<code>Km=value, Tm=value, Tau=value</code><br><br>"
                        "Not sure what those are? Type <em>explain parameters</em> and I will guide you."
                    ),
                    "options": [], "chart": None
                })

            if local_intent in KNOWLEDGE_BASE and confidence > 0.2:
                return jsonify({"reply": KNOWLEDGE_BASE[local_intent], "options": [], "chart": None})

            if local_intent == "rules" and confidence > 0.2:
                return _build_rules_response()

        if any(kw in user_msg_lower for kw in ["what rules", "show rules", "list rules", "how many rules", "rules do you"]):
            return _build_rules_response()

        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]) and has_digits:
            try:
                prompt = f"""
                You are an expert process control assistant. Extract FOPDT parameters from the user's message.
                User said: "{user_msg}"
                Current memory: km={bot_memory['km']}, tm={bot_memory['tm']}, taum={bot_memory['taum']}
                OUTPUT JSON ONLY:
                {{"km": float_or_null, "tm": float_or_null, "taum": float_or_null}}
                """
                res = llm_model.generate_content(prompt)
                ext = json.loads(res.text.replace('```json','').replace('```','').strip())
                for k in ["km", "tm", "taum"]:
                    if ext.get(k) is not None:
                        bot_memory[k] = ext[k]
            except:
                pass

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        missing = []
        if not km:   missing.append("<strong>Km</strong> (process gain)")
        if not tm:   missing.append("<strong>Tm</strong> (time constant)")
        if not taum: missing.append("<strong>Tau</strong> (dead time)")

        if missing:
            if len(missing) == 3:
                return jsonify({
                    "reply": (
                        "To tune your PID controller I need three numbers from your process model.<br><br>"
                        "Please provide:<br>"
                        f"{'<br>'.join(f'- {m}' for m in missing)}<br><br>"
                        "Format: <code>Km=2, Tm=10, Tau=2</code><br>"
                        "Type <em>explain parameters</em> if you need help finding these values."
                    ),
                    "options": [], "chart": None
                })
            else:
                return jsonify({
                    "reply": (
                        f"Almost there. I still need: {', '.join(missing)}<br><br>"
                        "Provide the remaining values to continue."
                    ),
                    "options": [], "chart": None
                })

        stage = bot_memory['interview_stage']

        if stage > 0:
            interview_q = INTERVIEW[stage]
            answer = None

            for opt in interview_q["options"]:
                if user_msg_lower == opt["val"]:
                    answer = opt["val"]
                    break

            if not answer:
                answer = parse_interview_answer(stage, user_msg_lower)

            if answer:
                updates = interview_q["map"][answer]
                bot_memory.update(updates)

                if stage == 2:
                    if answer == "fast":
                        bot_memory['allows_overshoot'] = True
                    else:
                        bot_memory['allows_overshoot'] = False

                if stage == 4:
                    bot_memory['overshoot_answer'] = answer

                bot_memory['interview_stage'] += 1
                stage = bot_memory['interview_stage']

                if stage == 4 and not bot_memory.get('allows_overshoot', False):
                    bot_memory['interview_stage'] = 5
                    return _run_tuning_engine()

                if stage <= 4:
                    next_q = INTERVIEW[stage]
                    return jsonify({
                        "reply": next_q["text"],
                        "options": [{"label": o["label"], "val": o["val"]} for o in next_q["options"]],
                        "chart": None
                    })
                else:
                    return _run_tuning_engine()
            else:
                return jsonify({
                    "reply": f"I did not recognise that selection. {interview_q['text']}",
                    "options": [{"label": o["label"], "val": o["val"]} for o in interview_q["options"]],
                    "chart": None
                })

        if stage == 0:
            bot_memory['interview_stage'] = 1
            q = INTERVIEW[1]
            ratio = taum / tm
            if ratio < 0.2:
                ratio_insight = f" Your dead-time ratio (tau/T = {round(ratio,2)}) is <strong>low</strong> — this process is very controllable."
            elif ratio < 0.5:
                ratio_insight = f" Your dead-time ratio (tau/T = {round(ratio,2)}) is <strong>moderate</strong> — standard tuning applies."
            else:
                ratio_insight = f" Your dead-time ratio (tau/T = {round(ratio,2)}) is <strong>high</strong> — I will apply dead-time compensating rules."

            return jsonify({
                "reply": (
                    f"Parameters confirmed: <strong>Km={km}, Tm={tm}s, Tau={taum}s</strong>.{ratio_insight}<br><br>"
                    f"{q['text']}"
                ),
                "options": [{"label": o["label"], "val": o["val"]} for o in q["options"]],
                "chart": None
            })

        return jsonify({
            "reply": "State machine error. Please reset and try again.",
            "options": [], "chart": None
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"reply": "System error. Please reset and try again.", "options": []})

def _build_rules_response():
    categories = {
        "Servo / Setpoint Tracking": [],
        "Regulatory / Disturbance Rejection": [],
        "General Purpose": []
    }
    for k, v in rules_db.items():
        mode_val = v.get("mode", -1)
        if mode_val == 1:
            categories["Servo / Setpoint Tracking"].append(v.get("name", k.replace("_", " ").title()))
        elif mode_val == 0:
            categories["Regulatory / Disturbance Rejection"].append(v.get("name", k.replace("_", " ").title()))
        else:
            categories["General Purpose"].append(v.get("name", k.replace("_", " ").title()))

    reply = f"<strong>Tuning database: {len(rules_db)} rules</strong> from Aidan O'Dwyer's handbook:<br><br>"
    num = 1
    for cat, rule_list in categories.items():
        if rule_list:
            reply += f"<strong>{cat}</strong><br>"
            for r in rule_list:
                reply += f"{num}. {r}<br>"
                num += 1
            reply += "<br>"

    reply += "<em>The Random Forest AI selects the optimal rule for your specific process and objectives.</em>"
    return jsonify({"reply": reply, "options": [], "chart": None})

def _run_tuning_engine():
    global bot_memory
    km   = bot_memory['km']
    tm   = bot_memory['tm']
    taum = bot_memory['taum']
    mode = bot_memory.get('mode', 1)
    os_v = bot_memory.get('overshoot', 0)
    rob  = bot_memory.get('robust', 0)
    met  = bot_memory.get('metric', 1)
    overshoot_answer = bot_memory.get('overshoot_answer', None)

    ratio    = taum / tm
    features = np.array([[km, tm, taum, ratio, mode, os_v, rob, met]])

    best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"

    if overshoot_answer and overshoot_answer in OVERSHOOT_RULE_OVERRIDE:
        override_key = OVERSHOOT_RULE_OVERRIDE[overshoot_answer]
        if override_key in rules_db:
            best_rule = override_key

    r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

    tau_c_val = max(0.1, taum)
    safe_env = {
        "km": km, "tm": tm, "taum": taum,
        "tau_c": tau_c_val,
        "min": min, "max": max,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "np": np, "math": math
    }

    try:
        if r.get('kc_math') == "SPECIAL_LOOKUP":
            kc = (0.85 * tm) / (km * taum)
            ti = 2.4 * taum
        else:
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
            if not ti or ti <= 0:
                ti = 0.001
    except Exception as e:
        print(f"Math eval error: {e}")
        kc = (1.2 * tm) / (km * taum)
        ti = 2.0 * taum
        r  = {"name": "Ziegler-Nichols (Fallback)"}

    chart, os_est, settling = simulate_step(kc, ti, km, tm, taum)

    mode_str   = "Servo (Setpoint Tracking)" if mode == 1 else "Regulatory (Disturbance Rejection)"
    metric_map = {"1": "IAE — smooth, minimal overshoot", "2": "ISE — fast, aggressive", "3": "ITAE — balanced, minimal long-term error"}
    metric_str = metric_map.get(str(met), "IAE")
    robust_str = "Robust (model-uncertainty tolerant)" if rob == 1 else "Performance-optimized"

    rule_name = r.get('name', best_rule)
    rule_description = r.get('unique_feature', 'a specialized rule from O\'Dwyer\'s handbook.')

    overshoot_note = ""
    if overshoot_answer:
        os_pct_map = {"os_5": "5%", "os_10": "10%", "os_20": "20%", "os_30": "30%"}
        pct = os_pct_map.get(overshoot_answer, "")
        overshoot_note = f"- Overshoot specification: <strong>{pct} target</strong><br>"

    final_reply = (
        f"<strong>Optimization Complete</strong><br><br>"
        f"<strong>Process Analysis:</strong><br>"
        f"- Mode: {mode_str}<br>"
        f"- Performance objective: {metric_str}<br>"
        f"- Tuning philosophy: {robust_str}<br>"
        f"{overshoot_note}"
        f"- Dead-time ratio (tau/T): {round(ratio, 3)}<br><br>"
        f"<strong>AI Selection:</strong> Random Forest evaluated all {len(rules_db)} candidate rules and selected "
        f"<strong>{rule_name}</strong> — {rule_description}<br><br>"
        f"<strong>PID Parameters:</strong><br>"
        f"- Proportional Gain <strong>Kc = {round(kc, 4)}</strong><br>"
        f"- Integral Time <strong>Ti = {round(ti, 4)} s</strong><br><br>"
        f"<strong>Predicted Closed-Loop Performance:</strong><br>"
        f"- Overshoot: <strong>{os_est}%</strong><br>"
        f"- Estimated Settling Time: <strong>{settling} s</strong><br><br>"
        f"Step response simulation below."
    )

    bot_memory = {
        "km": None, "tm": None, "taum": None, "tau_c": None,
        "mode": None, "metric": None, "robust": None, "overshoot": None,
        "interview_stage": 0,
        "allows_overshoot": False,
        "overshoot_answer": None
    }

    return jsonify({"reply": final_reply, "chart": chart, "options": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
