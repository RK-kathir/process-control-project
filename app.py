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

# 1. CLOUD NLP ENGINE (Gemini API for extraction)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})

# 2. LOAD DECISION ENGINE (Random Forest & Database)
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_dir, 'tuning_rules.json'), 'r') as f:
        rules_db = json.load(f)
    with open(os.path.join(current_dir, 'ai_brain.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    print(f"SUCCESS: {len(rules_db)} rules and AI Brain loaded.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}

# 3. EDGE NLP ENGINE (Local Intent Classifier)
nlp_training_data = [
    ("hi", "greeting"), ("hello", "greeting"), ("hey", "greeting"),
    ("who are you", "identity"), ("what can you do", "identity"),
    ("what tuning rules do you have", "rules"), ("show me the rules", "rules"), ("list rules", "rules")
]
texts, labels = zip(*nlp_training_data)
intent_vectorizer = TfidfVectorizer()
X_nlp = intent_vectorizer.fit_transform(texts)
intent_model = LinearSVC()
intent_model.fit(X_nlp, labels)

# Intelligent State Memory
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "metric": None, "robust": None, "overshoot": None
}

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

    os_val = 0.0
    try:
        os_val = round(max(0, (np.max(pv) - 1.0) * 100), 1)
        if math.isnan(os_val) or math.isinf(os_val): os_val = 0.0
    except: pass

    graph_data = {
        "data": [
            {"x": t.tolist(), "y": pv.tolist(), "type": "scatter", "name": "Process Output", "line": {"color": "#0ea5e9", "width": 3}},
            {"x": [t[0], t[-1]], "y": [1.0, 1.0], "type": "scatter", "name": "Setpoint", "line": {"color": "#ef4444", "dash": "dash"}}
        ],
        "layout": {
            "title": "Closed-Loop Step Response", "xaxis": {"title": "Time (s)", "gridcolor": "#333"}, "yaxis": {"title": "Process Variable", "gridcolor": "#333"}, 
            "paper_bgcolor": "transparent", "plot_bgcolor": "transparent", "font": {"color": "#e0e0e0"}, "margin": {"l": 40, "r": 20, "t": 40, "b": 40}
        }
    }
    return json.dumps(graph_data), os_val

def local_regex_extract(msg):
    ext = {"km": None, "tm": None, "taum": None}
    msg_lower = msg.lower()
    km_m = re.search(r'(km|gain|k)\s*(is|:|=)?\s*(\d+\.?\d*)', msg_lower)
    tm_m = re.search(r'(tm|lag|t)\s*(is|:|=)?\s*(\d+\.?\d*)', msg_lower)
    tau_m = re.search(r'(tau|dead|delay)\s*(is|:|=)?\s*(\d+\.?\d*)', msg_lower)
    if km_m: ext["km"] = float(km_m.group(3))
    if tm_m: ext["tm"] = float(tm_m.group(3))
    if tau_m: ext["taum"] = float(tau_m.group(3))
    return ext

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        user_msg_lower = user_msg.lower().strip()

        # 1. Reset
        if user_msg_lower == "reset":
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "metric": None, "robust": None, "overshoot": None}
            return jsonify({"reply": "Session reset. Please provide your process parameters (Km, Tm, Tau).", "options": [], "chart": None})

        # 2. EDGE NLP ROUTING (Only runs if we aren't in the middle of an interview)
        if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]):
            if len(user_msg_lower.split()) < 8 and not any(char.isdigit() for char in user_msg_lower):
                try:
                    vec = intent_vectorizer.transform([user_msg_lower])
                    local_intent = intent_model.predict(vec)[0]
                except: local_intent = "none"
                
                if local_intent == "greeting":
                    return jsonify({"reply": "Hello! Please provide your system parameters (Km, Tm, Tau) to begin.", "options": [], "chart": None})
                
                if local_intent == "rules" or "rules" in user_msg_lower:
                    categories = {"Servo": [], "Regulator": [], "General/Hybrid": []}
                    for k, v in rules_db.items():
                        mode_val = v.get("mode", -1)
                        mode_str = "Servo" if mode_val == 1 else "Regulator" if mode_val == 0 else "General/Hybrid"
                        categories[mode_str].append(v.get("name", k.replace("_", " ").title()))
                    
                    # STRICT 1. 2. 3. FORMATTING (No Wall of Text)
                    reply_text = f"<strong>I have {len(rules_db)} tuning rules loaded:</strong><br><br>"
                    rule_num = 1
                    for mode_category, rules_list in categories.items():
                        if rules_list:
                            reply_text += f"<strong>{mode_category} Rules:</strong><br>"
                            for r in rules_list:
                                reply_text += f"{rule_num}. {r}<br>"
                                rule_num += 1
                            reply_text += "<br>"
                    return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. Extract Numbers Locally
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None: bot_memory[k] = regex_data[k]

        # 4. Request Parameters if Missing
        if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]):
            return jsonify({"reply": "I need complete process data to tune. Please ensure you provide Gain (Km), Lag (Tm), and Dead Time (Tau).", "options": [], "chart": None})

        # =====================================================================
        # 5. STRICT 3-STEP INTERVIEW (Cannot loop, filters all 40+ rules)
        # =====================================================================
        
        # STEP 1: Ask Mode
        if bot_memory['mode'] is None:
            if "servo" in user_msg_lower or "setpoint" in user_msg_lower: bot_memory['mode'] = 1
            elif "regulator" in user_msg_lower or "disturbance" in user_msg_lower: bot_memory['mode'] = 0
            else:
                return jsonify({
                    "reply": "Parameters locked. <strong>Question 1 of 3:</strong> Are we optimizing for Setpoint Tracking (Servo) or Disturbance Rejection (Regulator)?",
                    "options": [{"label": "Servo (Setpoint)", "val": "servo"}, {"label": "Regulator (Disturbance)", "val": "regulator"}],
                    "chart": None
                })

        # STEP 2: Ask Metric
        if bot_memory['metric'] is None:
            if "iae" in user_msg_lower or "smooth" in user_msg_lower: bot_memory['metric'] = 1
            elif "ise" in user_msg_lower or "aggressive" in user_msg_lower: bot_memory['metric'] = 2
            elif "itae" in user_msg_lower or "balance" in user_msg_lower: bot_memory['metric'] = 3
            else:
                return jsonify({
                    "reply": "Got it. <strong>Question 2 of 3:</strong> Which performance metric should I minimize?",
                    "options": [{"label": "IAE (Smooth)", "val": "iae"}, {"label": "ISE (Aggressive)", "val": "ise"}, {"label": "ITAE (Balanced)", "val": "itae"}],
                    "chart": None
                })

        # STEP 3: Ask Robustness
        if bot_memory['robust'] is None:
            if "robust" in user_msg_lower or "safe" in user_msg_lower:
                bot_memory['robust'] = 1
                bot_memory['overshoot'] = 0
            elif "fast" in user_msg_lower or "allow" in user_msg_lower:
                bot_memory['robust'] = 0
                bot_memory['overshoot'] = 2
            else:
                return jsonify({
                    "reply": "Almost done. <strong>Question 3 of 3:</strong> Do you need a Robust/Safe response (0% Overshoot) or a Fast response (Allows Overshoot)?",
                    "options": [{"label": "Robust & Safe", "val": "robust"}, {"label": "Fast (Allow Overshoot)", "val": "fast"}],
                    "chart": None
                })

        # =====================================================================
        # 6. EXECUTE DECISION ENGINE & GRAPH
        # =====================================================================
        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        mode = bot_memory['mode']
        os_val = bot_memory['overshoot']
        rob = bot_memory['robust']
        met = bot_memory['metric']
        
        ratio = taum / tm
        features = np.array([[km, tm, taum, ratio, mode, os_val, rob, met]])
        
        best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"
        r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

        safe_env = {
            "km": km, "tm": tm, "taum": taum, "tau_c": bot_memory.get('tau_c', max(0.1, taum)), 
            "min": min, "max": max, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "np": np, "math": math
        }
        
        try:
            if r.get('kc_math') == "SPECIAL_LOOKUP":
                kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
            else:
                kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
                ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
                if ti <= 0: ti = 0.001
        except Exception as math_error:
            print(f"Math Error: {math_error}")
            kc, ti = (1.2 * tm) / (km * taum), 2.0 * taum
            r = {"name": f"Ziegler-Nichols (Fallback from {best_rule})"}

        chart, os_est = simulate_step(kc, ti, km, tm, taum)
        
        mode_str = "Servo" if mode == 1 else "Regulator"
        metric_str = "IAE" if met == 1 else "ISE" if met == 2 else "ITAE"
        
        final_reply = f"<strong>System Optimization Complete.</strong><br><br>"
        final_reply += f"<strong>Analysis:</strong> You requested a <strong>{mode_str}</strong> response minimizing <strong>{metric_str}</strong>, tailored for your specific robustness needs.<br><br>"
        final_reply += f"<strong>Decision:</strong> My Random Forest engine evaluated all rules and selected: <strong>{r.get('name')}</strong>.<br><br>"
        final_reply += f"<strong>Kc:</strong> {round(kc,4)}<br><strong>Ti:</strong> {round(ti,4)}s<br><strong>Overshoot:</strong> {os_est}%<br><br>Here is your step response:"
        
        # Reset Memory
        bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "metric": None, "robust": None, "overshoot": None}
        
        return jsonify({"reply": final_reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": "System Error. Please try again.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
