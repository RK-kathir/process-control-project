import os
import re
import json
import pickle
import numpy as np
import traceback
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    print("SUCCESS: Database and AI Brain loaded.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}

# ─────────────────────────────────────────────
# STATE MEMORY
# ─────────────────────────────────────────────
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "overshoot": None, "robust": None, "metric": None,
    "preferences_set": False,
    "stage": "greet",          # greet → params → q1 → q2 → q3 → tune
    "q_index": 0,
    "answers": {}
}

# ─────────────────────────────────────────────
# SMART QUESTION FUNNEL
# Each question has a user-friendly framing and maps answers to feature values
# ─────────────────────────────────────────────
QUESTION_FUNNEL = [
    {
        "id": "response_style",
        "text": "⚡ How fast should your system respond to a change?",
        "subtitle": "Think of it like a car's throttle — instant punch or smooth cruise?",
        "options": [
            {"label": "🚀 React instantly — speed matters most", "val": "aggressive",
             "features": {"mode": 1, "overshoot": 2, "robust": 0, "metric": 2}},
            {"label": "🎯 Settle quickly but stay controlled", "val": "balanced",
             "features": {"mode": 1, "overshoot": 1, "robust": 0, "metric": 3}},
            {"label": "🛡️ Take time, but never overshoot or oscillate", "val": "smooth",
             "features": {"mode": 0, "overshoot": 0, "robust": 1, "metric": 1}},
        ]
    },
    {
        "id": "disturbance_type",
        "text": "🌊 What kind of disruptions does your process face?",
        "subtitle": "Does something external push your system off-target, or do you mainly change the target itself?",
        "options": [
            {"label": "🔧 External disturbances hit constantly (load changes, pressure drops)", "val": "regulatory",
             "features": {"mode": 0}},
            {"label": "🎚️ I mostly change the setpoint / target value", "val": "servo",
             "features": {"mode": 1}},
            {"label": "🔄 Both happen equally", "val": "both",
             "features": {"mode": 0}},
        ]
    },
    {
        "id": "safety_priority",
        "text": "🏭 How critical is safety and stability for this process?",
        "subtitle": "Some processes (like chemical reactors) must never oscillate. Others can tolerate some wiggle.",
        "options": [
            {"label": "🔴 Critical — oscillation or overshoot is dangerous", "val": "critical",
             "features": {"robust": 1, "overshoot": 0}},
            {"label": "🟡 Moderate — some overshoot is acceptable", "val": "moderate",
             "features": {"robust": 0, "overshoot": 1}},
            {"label": "🟢 Relaxed — performance matters more than caution", "val": "relaxed",
             "features": {"robust": 0, "overshoot": 2}},
        ]
    }
]

# ─────────────────────────────────────────────
# KNOWLEDGE BASE FOR OFFLINE Q&A
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "what is pid": "A **PID controller** has three parts: **P** (Proportional) reacts to current error, **I** (Integral) eliminates steady-state offset, and **D** (Derivative) predicts future error. Together they keep your process at the desired setpoint.",
    "what is kc": "**Kc (Controller Gain)** is how aggressively the controller reacts. A high Kc = fast response but risk of oscillation. A low Kc = sluggish but stable.",
    "what is ti": "**Ti (Integral Time)** controls how fast the controller corrects steady-state error. Smaller Ti = faster correction but can cause instability.",
    "what is km": "**Km (Process Gain)** is how much the process output changes per unit of controller output. It describes your process's sensitivity.",
    "what is tm": "**Tm (Time Constant)** is how long your process takes to reach ~63% of its final value after a step change — basically its 'speed of response'.",
    "what is tau": "**Tau (Dead Time / θ)** is the delay before your process starts responding at all. High dead time makes control harder.",
    "what is fopdt": "**FOPDT (First Order Plus Dead Time)** is the standard model for most industrial processes. It captures three key behaviors: process gain (Km), lag (Tm), and delay (Tau).",
    "what is ziegler nichols": "**Ziegler-Nichols** is one of the oldest tuning rules (1942). It's aggressive and great for fast response but can overshoot significantly. Best for non-critical processes.",
    "what is cohen coon": "**Cohen-Coon** is an improvement over Z-N for processes with large dead time. It's more robust while still being relatively fast.",
    "what is imc": "**IMC (Internal Model Control)** like Skogestad's method is model-based and highly reliable. It's gentle and robust — ideal for critical processes.",
    "what is lambda tuning": "**Lambda Tuning** lets you directly set the desired closed-loop speed (Lambda). Slower Lambda = more robust. Faster Lambda = better performance.",
    "what is amigo": "**AMIGO** (Approximate M-constrained Integral Gain Optimization) is a modern robust method that balances performance and stability margins.",
    "what is iae": "**IAE (Integral Absolute Error)** minimizes the total absolute error over time. Rules optimizing for IAE tend to be moderately aggressive.",
    "what is ise": "**ISE (Integral Squared Error)** penalizes large errors more heavily, producing faster but potentially oscillatory responses.",
    "what is itae": "**ITAE (Integral Time-weighted Absolute Error)** penalizes errors that persist for a long time, producing well-damped, smooth responses.",
    "what is overshoot": "**Overshoot** is how much your process variable exceeds the setpoint before settling. High overshoot is dangerous for temperature-sensitive or pressure-critical systems.",
    "what is robustness": "**Robustness** means the controller still works even if your process model isn't perfectly accurate. Robust tuning trades some speed for reliability.",
    "what is servo": "**Servo control** focuses on following setpoint changes quickly. Used when you frequently change your target value.",
    "what is regulatory": "**Regulatory control** focuses on rejecting disturbances and keeping the process steady at a fixed setpoint despite external upsets.",
}

def find_knowledge_answer(msg):
    msg_lower = msg.lower()
    # Check for "what is X" patterns
    for key, answer in KNOWLEDGE_BASE.items():
        words = key.split()
        if all(w in msg_lower for w in words):
            return answer
    # Fuzzy fallback — check if any topic keyword appears
    topic_map = {
        "pid": "what is pid", "controller": "what is pid",
        "kc": "what is kc", "gain": "what is kc",
        "ti": "what is ti", "integral time": "what is ti",
        "km": "what is km", "process gain": "what is km",
        "tm": "what is tm", "time constant": "what is tm",
        "tau": "what is tau", "dead time": "what is tau", "delay": "what is tau",
        "fopdt": "what is fopdt", "first order": "what is fopdt",
        "ziegler": "what is ziegler nichols", "z-n": "what is ziegler nichols",
        "cohen": "what is cohen coon",
        "imc": "what is imc", "skogestad": "what is imc",
        "lambda": "what is lambda tuning",
        "amigo": "what is amigo",
        "iae": "what is iae",
        "ise": "what is ise",
        "itae": "what is itae",
        "overshoot": "what is overshoot",
        "robust": "what is robustness",
        "servo": "what is servo",
        "regulatory": "what is regulatory",
    }
    for kw, kb_key in topic_map.items():
        if kw in msg_lower:
            return KNOWLEDGE_BASE[kb_key]
    return None

# ─────────────────────────────────────────────
# ADVANCED PARAMETER EXTRACTION
# ─────────────────────────────────────────────
def local_regex_extract(msg):
    ext = {"km": None, "tm": None, "taum": None}
    msg_lower = msg.lower()
    
    # Km patterns: "km=2", "gain is 2", "k = 2.5", "process gain of 3"
    km_patterns = [
        r'\bkm\s*[=:is\s]+\s*(\d+\.?\d*)',
        r'\bk\s*[=:]\s*(\d+\.?\d*)\b',
        r'(?:process\s+gain|gain)\s+(?:is|of|=)?\s*(\d+\.?\d*)',
    ]
    for p in km_patterns:
        m = re.search(p, msg_lower)
        if m: ext["km"] = float(m.group(1)); break

    # Tm patterns: "tm=10", "lag=10", "time constant of 5"
    tm_patterns = [
        r'\btm\s*[=:is\s]+\s*(\d+\.?\d*)',
        r'(?:time\s+constant|lag)\s+(?:is|of|=)?\s*(\d+\.?\d*)',
        r'\bt\s*[=:]\s*(\d+\.?\d*)\b(?!\s*au)',
    ]
    for p in tm_patterns:
        m = re.search(p, msg_lower)
        if m: ext["tm"] = float(m.group(1)); break

    # Tau/dead time patterns: "tau=2", "dead time=3", "delay of 1.5"
    tau_patterns = [
        r'\btau[m]?\s*[=:is\s]+\s*(\d+\.?\d*)',
        r'(?:dead\s*time|delay|theta)\s+(?:is|of|=)?\s*(\d+\.?\d*)',
    ]
    for p in tau_patterns:
        m = re.search(p, msg_lower)
        if m: ext["taum"] = float(m.group(1)); break

    return ext

# ─────────────────────────────────────────────
# LOCAL PREFERENCE EXTRACTION (no LLM needed)
# ─────────────────────────────────────────────
PREFERENCE_MAP = {
    # Aggressive patterns
    "aggressive|fast|quick|rapid|speed|ise|urgent|instant": {"mode": 1, "overshoot": 2, "robust": 0, "metric": 2},
    # Smooth/safe patterns
    "smooth|safe|gentle|slow|stable|careful|conservative|iae|no overshoot": {"mode": 1, "overshoot": 0, "robust": 1, "metric": 1},
    # Balanced patterns
    "balance|standard|moderate|normal|itae|typical|medium": {"mode": 0, "overshoot": 1, "robust": 0, "metric": 3},
    # Servo patterns
    "setpoint|servo|tracking|follow": {"mode": 1},
    # Regulatory patterns
    "disturbance|regulatory|reject|load|upset": {"mode": 0},
}

def local_preference_extract(msg):
    msg_lower = msg.lower()
    merged = {}
    for pattern_group, features in PREFERENCE_MAP.items():
        for kw in pattern_group.split("|"):
            if kw in msg_lower:
                merged.update(features)
                break
    return merged

# ─────────────────────────────────────────────
# SIMULATOR (untouched)
# ─────────────────────────────────────────────
def simulate_step(kc, ti, km, tm, taum):
    t = np.linspace(0, (tm + taum) * 6, 400)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum = 0
    d_steps = int(taum / dt)
    ti_val = ti if ti > 0 else 0.001

    for i in range(1, len(t)):
        err = 1.0 - pv[i-1]
        err_sum += err * dt
        mv = max(0, min(100, kc * (err + (1/ti_val) * err_sum)))
        mv_hist[i] = mv
        d_idx = i - d_steps
        d_mv = mv_hist[d_idx] if d_idx >= 0 else 0
        dpv = ((km * d_mv) - pv[i-1]) / tm
        pv[i] = pv[i-1] + dpv * dt

    settling_time = None
    for i in range(len(pv)-1, 0, -1):
        if abs(pv[i] - 1.0) > 0.02:
            settling_time = round(t[i], 2)
            break

    graph_data = {
        "data": [
            {
                "x": t.tolist(), "y": pv.tolist(),
                "type": "scatter", "name": "Process Output",
                "line": {"color": "#00d4ff", "width": 3},
                "fill": "tozeroy", "fillcolor": "rgba(0,212,255,0.05)"
            },
            {
                "x": [t[0], t[-1]], "y": [1.0, 1.0],
                "type": "scatter", "name": "Setpoint",
                "line": {"color": "#ff6b6b", "dash": "dash", "width": 2}
            }
        ],
        "layout": {
            "title": {"text": "Closed-Loop Step Response", "font": {"size": 14}},
            "xaxis": {"title": "Time (s)", "gridcolor": "rgba(255,255,255,0.07)", "zeroline": False},
            "yaxis": {"title": "Process Variable (normalized)", "gridcolor": "rgba(255,255,255,0.07)", "zeroline": False},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "rgba(0,0,0,0.2)",
            "font": {"color": "#c9d1d9", "family": "JetBrains Mono, monospace"},
            "margin": {"l": 50, "r": 20, "t": 50, "b": 50},
            "legend": {"bgcolor": "transparent"},
            "hovermode": "x unified"
        }
    }
    os_est = round(max(0, (np.max(pv) - 1.0) * 100), 1)
    return json.dumps(graph_data), os_est, settling_time

# ─────────────────────────────────────────────
# INTENT ROUTER
# ─────────────────────────────────────────────
def detect_intent(msg):
    ml = msg.lower().strip()
    if ml in ["reset", "start over", "new session", "restart"]:
        return "reset"
    if any(k in ml for k in ["rules", "what do you have", "what rules", "list rules", "available rules"]):
        return "list_rules"
    if any(k in ml for k in ["what is", "explain", "tell me about", "what does", "how does", "define"]):
        return "knowledge"
    if any(k in ml for k in ["help", "how to use", "guide", "tutorial"]):
        return "help"
    # Check if message contains process description keywords (for LLM routing)
    process_keywords = ["tank", "reactor", "furnace", "boiler", "flow", "temperature", "pressure",
                        "level", "heat", "pump", "valve", "column", "distillation", "pH", "speed",
                        "motor", "conveyor", "humidity", "concentration"]
    if any(kw in ml for kw in process_keywords) and not any(k in ml for k in ["km", "tm", "tau", "gain"]):
        return "process_description"
    return "general"

# ─────────────────────────────────────────────
# RESET STATE
# ─────────────────────────────────────────────
def reset_memory():
    return {
        "km": None, "tm": None, "taum": None, "tau_c": None,
        "mode": None, "overshoot": None, "robust": None, "metric": None,
        "preferences_set": False, "stage": "params", "q_index": 0, "answers": {}
    }

# ─────────────────────────────────────────────
# MAIN CHAT ENDPOINT
# ─────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '').strip()
        if not user_msg:
            return jsonify({"reply": "Please type a message!", "options": [], "chart": None})
        
        user_msg_lower = user_msg.lower()
        intent = detect_intent(user_msg)

        # ── RESET ──────────────────────────────────────
        if intent == "reset":
            bot_memory = reset_memory()
            return jsonify({
                "reply": "Session reset! Let's start fresh.\n\nPlease give me your **FOPDT process parameters**:\n- **Km** (Process Gain)\n- **Tm** (Time Constant in seconds)\n- **Tau** (Dead Time in seconds)\n\nYou can write something like: `Km=2, Tm=10, Tau=2`",
                "options": [], "chart": None,
                "chips": ["Km=2, Tm=10, Tau=2", "What is Km?", "What is FOPDT?"]
            })

        # ── LIST RULES ─────────────────────────────────
        if intent == "list_rules":
            rule_names = list(rules_db.keys()) if rules_db else []
            categories = {
                "⚡ Fast Response": ["ziegler_nichols", "cohen_coon", "hazebroek_duyser"],
                "🎯 Minimum Error (IAE/ISE/ITAE)": ["rovira", "zhuang_atherton", "wang_juang_chan"],
                "🛡️ Robust & Safe": ["skogestad_imc", "amigo", "chun"],
                "🔬 Model-Based": ["lambda_tuning", "direct_synthesis"]
            }
            reply = "**My tuning rule database (O'Dwyer's Handbook):**\n\n"
            for cat, rules in categories.items():
                reply += f"**{cat}**\n"
                for r in rules:
                    if r in rules_db:
                        reply += f"  • {rules_db[r].get('name', r)}\n"
            reply += f"\n_Total: {len(rule_names)} rules available. My AI brain picks the best one for your process._"
            return jsonify({"reply": reply, "options": [], "chart": None,
                           "chips": ["Explain Ziegler-Nichols", "What is IMC?", "Start tuning"]})

        # ── KNOWLEDGE Q&A ──────────────────────────────
        if intent == "knowledge":
            answer = find_knowledge_answer(user_msg)
            if answer:
                return jsonify({"reply": answer, "options": [], "chart": None,
                               "chips": ["Start tuning", "What rules do you have?", "What is FOPDT?"]})
            # Fall through to LLM for unknown questions

        # ── HELP ───────────────────────────────────────
        if intent == "help":
            return jsonify({
                "reply": "**How to use Tuning Bot:**\n\n**Step 1:** Give me your process parameters (Km, Tm, Tau)\n**Step 2:** I'll ask you 3 simple questions about how you want your system to behave\n**Step 3:** My AI brain picks the best rule and calculates your PID gains\n**Step 4:** You get Kc and Ti values + a live simulation graph\n\n💡 You don't need to know any control theory — just answer in plain English!",
                "options": [], "chart": None,
                "chips": ["Km=2, Tm=10, Tau=2", "What tuning rules do you have?"]
            })

        # ── PROCESS DESCRIPTION (LLM extracts params from context) ────
        if intent == "process_description":
            ai_prompt = f"""
You are an expert process control engineer. A user described their process in plain language.
Extract FOPDT parameters if mentioned, and give an empathetic, helpful reply asking for what's missing.

User said: "{user_msg}"
Current memory: {json.dumps({k: bot_memory[k] for k in ['km','tm','taum']})}

Rules:
- If user mentions "water tank" with no numbers, Km is typically 1-3, Tm 20-60s, Tau 2-10s — hint at typical ranges
- Never fabricate exact numbers unless the user stated them
- Be friendly and non-technical

OUTPUT ONLY JSON: {{"km": float_or_null, "tm": float_or_null, "taum": float_or_null, "reply": "helpful string"}}
"""
            try:
                res = llm_model.generate_content(ai_prompt)
                ext = json.loads(res.text.replace('```json','').replace('```','').strip())
                for k in ["km", "tm", "taum"]:
                    if ext.get(k) is not None:
                        bot_memory[k] = ext[k]
                reply_text = ext.get('reply', "I'd need Km, Tm, and Tau to proceed. Can you share those values?")
                return jsonify({"reply": reply_text, "options": [], "chart": None})
            except:
                pass

        # ── EXTRACT PARAMETERS (always try regex first) ────────────────
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        # ── EXTRACT PREFERENCES (regex first) ──────────────────────────
        pref = local_preference_extract(user_msg)
        if pref:
            bot_memory.update(pref)
            if all(k in pref for k in ["mode", "overshoot", "robust", "metric"]):
                bot_memory["preferences_set"] = True

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # ── IF WE HAVE PARAMS: CHECK STAGE ────────────────────────────
        if not all([km, tm, taum]):
            # Need to collect parameters
            missing = [k.upper() for k in ['km', 'tm', 'taum'] if not bot_memory[k]]
            
            # Try LLM for complex parameter extraction
            if any(c.isdigit() for c in user_msg):
                ai_prompt = f"""
Extract FOPDT parameters from user message. Only extract values explicitly stated.
User said: "{user_msg}"
OUTPUT ONLY JSON: {{"km": float_or_null, "tm": float_or_null, "taum": float_or_null, "reply": "string asking for what's still missing"}}
Missing parameters: {missing}
"""
                try:
                    res = llm_model.generate_content(ai_prompt)
                    ext = json.loads(res.text.replace('```json','').replace('```','').strip())
                    for k in ["km", "tm", "taum"]:
                        if ext.get(k) is not None:
                            bot_memory[k] = ext[k]
                    km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
                    if not all([km, tm, taum]):
                        return jsonify({"reply": ext.get('reply', f"Still need: {', '.join(missing)}"), "options": [], "chart": None})
                except:
                    pass

            if not all([km, tm, taum]):
                still_missing = [k.upper() for k in ['km', 'tm', 'taum'] if not bot_memory[k]]
                hints = {
                    "KM": "Process Gain — how much does the output change per unit input? (e.g., 2)",
                    "TM": "Time Constant — how many seconds to reach ~63% of final value? (e.g., 10)",
                    "TAUM": "Dead Time — how many seconds before the process starts responding? (e.g., 2)"
                }
                reply = "I need a few numbers to tune your system. Here's what's still missing:\n\n"
                for param in still_missing:
                    reply += f"**{param}** — {hints[param]}\n"
                reply += "\nExample: `Km=2, Tm=10, Tau=2`"
                return jsonify({"reply": reply, "options": [], "chart": None,
                               "chips": ["Km=2, Tm=10, Tau=2", "What is Km?", "What is dead time?"]})

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        ratio = taum / tm

        # ── PARAMETER CONFIRMED — START QUESTION FUNNEL ───────────────
        if all([km, tm, taum]) and not bot_memory['preferences_set']:
            q_idx = bot_memory.get('q_index', 0)

            # Check if user answered a funnel question via option button
            for q in QUESTION_FUNNEL:
                for opt in q["options"]:
                    if user_msg_lower == opt["val"] or user_msg_lower == opt["label"].lower():
                        bot_memory["answers"][q["id"]] = opt["val"]
                        bot_memory.update(opt["features"])
                        # advance question
                        current_q_ids = [qq["id"] for qq in QUESTION_FUNNEL]
                        if q["id"] in current_q_ids:
                            q_idx = current_q_ids.index(q["id"]) + 1
                            bot_memory["q_index"] = q_idx

            # Check if all 3 questions answered
            answered = len(bot_memory["answers"])
            
            if answered >= len(QUESTION_FUNNEL):
                # Merge all feature answers
                for q in QUESTION_FUNNEL:
                    if q["id"] in bot_memory["answers"]:
                        val = bot_memory["answers"][q["id"]]
                        for opt in q["options"]:
                            if opt["val"] == val:
                                bot_memory.update(opt["features"])
                # Set defaults for any missing features
                if bot_memory.get("mode") is None: bot_memory["mode"] = 1
                if bot_memory.get("overshoot") is None: bot_memory["overshoot"] = 1
                if bot_memory.get("robust") is None: bot_memory["robust"] = 0
                if bot_memory.get("metric") is None: bot_memory["metric"] = 1
                bot_memory["preferences_set"] = True
            else:
                # Ask next unanswered question
                unanswered = [q for q in QUESTION_FUNNEL if q["id"] not in bot_memory["answers"]]
                if unanswered:
                    next_q = unanswered[0]
                    q_num = QUESTION_FUNNEL.index(next_q) + 1
                    opts = [{"label": o["label"], "val": o["val"]} for o in next_q["options"]]
                    
                    prefix = ""
                    if q_num == 1:
                        prefix = f"Parameters locked ✓ (Km={km}, Tm={tm}s, Tau={taum}s)\n\nDead time ratio: **{round(ratio,2)}** {'⚠️ — high dead time, robustness matters!' if ratio > 0.5 else '✓ — good controllability'}\n\n"
                    else:
                        prefix = f"Got it! ({answered}/{len(QUESTION_FUNNEL)} questions done)\n\n"
                    
                    return jsonify({
                        "reply": f"{prefix}**Question {q_num} of {len(QUESTION_FUNNEL)}:** {next_q['text']}\n_{next_q['subtitle']}_",
                        "options": opts,
                        "chart": None,
                        "progress": answered / len(QUESTION_FUNNEL)
                    })

        # ── PREFERENCES SET — RUN AI BRAIN AND TUNE ───────────────────
        if all([km, tm, taum]) and bot_memory['preferences_set']:
            mode = bot_memory.get('mode', 1)
            os_val = bot_memory.get('overshoot', 0)
            rob = bot_memory.get('robust', 0)
            met = bot_memory.get('metric', 1)

            features = np.array([[km, tm, taum, ratio, mode, os_val, rob, met]])

            best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"
            r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

            safe_env = {"km": km, "tm": tm, "taum": taum, "tau_c": max(0.1*tm, taum), "min": min, "max": max}
            try:
                if r.get('kc_math') == "SPECIAL_LOOKUP":
                    kc = (0.85 * tm) / (km * taum)
                    ti = 2.4 * taum
                else:
                    kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
                    ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
            except Exception as e:
                kc = (0.9 / km) * (tm / taum)
                ti = 3.33 * taum

            chart, os_est, settle = simulate_step(kc, ti, km, tm, taum)
            
            rule_name = r.get('name', best_rule)
            
            # Generate contextual interpretation
            os_comment = "✅ No significant overshoot" if os_est < 2 else f"⚠️ Overshoot: {os_est}%"
            settle_comment = f"📉 Settling time: ~{settle}s" if settle else ""
            
            final_reply = (
                f"🧠 **AI Brain Selected:** {rule_name}\n\n"
                f"**Your PID Parameters:**\n"
                f"• **Kc (Proportional Gain):** `{round(kc, 4)}`\n"
                f"• **Ti (Integral Time):** `{round(ti, 4)} s`\n\n"
                f"**Simulation Results:**\n"
                f"• {os_comment}\n"
                f"• {settle_comment}\n\n"
                f"_Rule picked based on your process ratio τ/T = {round(ratio,2)} and performance preferences._"
            )
            
            # Reset for next session
            bot_memory = reset_memory()
            
            return jsonify({
                "reply": final_reply,
                "chart": chart,
                "options": [],
                "chips": ["Tune another process", "Explain this rule", "What is overshoot?"]
            })

        return jsonify({
            "reply": "Something went wrong in my reasoning. Let's start over!",
            "options": [],
            "chart": None,
            "chips": ["Start over"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"An internal error occurred. Please reset and try again.", "options": [], "chart": None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
