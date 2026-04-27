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

# 1. CLOUD NLP ENGINE (Gemini)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})

# 2. LOAD DECISION ENGINE (Random Forest)
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

# 3. EDGE NLP ENGINE (Local Intent Classifier - Zero API Cost)
nlp_training_data = [
    ("hi", "greeting"), ("hello there", "greeting"), ("hey", "greeting"), ("good morning", "greeting"),
    ("who are you", "identity"), ("what can you do", "identity"), ("how do you work", "identity"),
    ("what tuning rules do you have", "rules"), ("show me the rules", "rules"), ("list all rules", "rules"),
    ("km is 50", "parameters"), ("tm=10 tau=2", "parameters"), ("i have a water tank", "parameters")
]
texts, labels = zip(*nlp_training_data)
intent_vectorizer = TfidfVectorizer()
X_nlp = intent_vectorizer.fit_transform(texts)
intent_model = LinearSVC()
intent_model.fit(X_nlp, labels)

def get_local_intent(text):
    vec = intent_vectorizer.transform([text.lower()])
    return intent_model.predict(vec)[0]

# Intelligent State Memory
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "overshoot": None, "robust": None, "metric": None,
    "ready_to_tune": False
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
    except:
        pass

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
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "ready_to_tune": False}
            return jsonify({"reply": "Session reset. I am your AI Process Control Engineer. Please provide your process parameters (Km, Tm, Tau) and the type of system you are running.", "options": [], "chart": None})

        # 2. EDGE NLP ROUTING (Handles rules and greetings locally)
        if len(user_msg_lower.split()) < 10 and not any(char.isdigit() for char in user_msg_lower):
            local_intent = get_local_intent(user_msg_lower)
            
            if local_intent == "greeting":
                return jsonify({"reply": "Hello! Provide your system parameters (Km, Tm, Tau) to begin.", "options": [], "chart": None})
            
            if local_intent == "rules" or "rules" in user_msg_lower:
                categories = {"Servo": [], "Regulator": [], "General/Hybrid": []}
                for k, v in rules_db.items():
                    mode_val = v.get("mode", -1)
                    mode_str = "Servo" if mode_val == 1 else "Regulator" if mode_val == 0 else "General/Hybrid"
                    rule_name = v.get("name", k.replace("_", " ").title())
                    categories[mode_str].append(rule_name)
                
                reply_text = f"**I have {len(rules_db)} tuning rules loaded into my database.** Here is the structured list:<br><br>"
                rule_count = 1
                for mode_category, rules_list in categories.items():
                    if rules_list:
                        reply_text += f"**{mode_category} Rules:**<br>"
                        for r in rules_list:
                            reply_text += f"{rule_count}. {r}<br>"
                            rule_count += 1
                        reply_text += "<br>"
                return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. Extract Numbers Locally
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None: bot_memory[k] = regex_data[k]

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # 4. EDGE INTERCEPTOR: Breaks the infinite loop if user provides preference keywords
        if all([km, tm, taum]) and not bot_memory['ready_to_tune']:
            if any(w in user_msg_lower for w in ["smooth", "safe", "iae"]):
                bot_memory.update({"mode": 1, "overshoot": 0, "robust": 1, "metric": 1, "ready_to_tune": True})
            elif any(w in user_msg_lower for w in ["fast", "aggressive", "ise", "quick"]):
                bot_memory.update({"mode": 1, "overshoot": 2, "robust": 0, "metric": 2, "ready_to_tune": True})
            elif any(w in user_msg_lower for w in ["standard", "balance", "itae", "normal"]):
                bot_memory.update({"mode": 0, "overshoot": 1, "robust": 0, "metric": 3, "ready_to_tune": True})
            
            # If the user answered the question, skip the LLM and go straight to tuning
            if not bot_memory['ready_to_tune']:
                # CLOUD NLP ENGINE (Gemini Multi-Turn Interview)
                ai_prompt = f"""
                You are a Senior Industrial Process Control Engineer. 
                User says: "{user_msg}"
                Memory: {json.dumps({k:v for k,v in bot_memory.items() if v is not None})}
                
                Ask ONLY ONE logical question to determine if they want a fast response (which may overshoot) or a smooth/safe response. Provide 2-3 clickable options.
                OUTPUT JSON ONLY: {{"reply": "Your question here.", "options": [{{"label": "Fast", "val": "fast"}}, {{"label": "Smooth", "val": "smooth"}}]}}
                """
                try:
                    res = llm_model.generate_content(ai_prompt)
                    ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                    reply_text = ext.get("reply", "Understood.")
                    options = ext.get("options", [])
                except Exception as e:
                    reply_text = "I have your parameters. Do you want to tune this system for a Fast response, or a Smooth/Safe response?"
                    options = [{"label": "Fast & Aggressive", "val": "fast"}, {"label": "Smooth & Safe", "val": "smooth"}]

                return jsonify({"reply": reply_text, "options": options, "chart": None})

        # 5. Missing Parameters
        if not all([km, tm, taum]):
            return jsonify({"reply": "I received some parameters, but I am still missing complete data. Please ensure you provide Km, Tm, and Tau.", "options": [], "chart": None})

        # 6. EXECUTE DECISION ENGINE & GRAPH
        if all([km, tm, taum]) and bot_memory['ready_to_tune']:
            mode = bot_memory.get('mode', 1)
            os_val = bot_memory.get('overshoot', 0)
            rob = bot_memory.get('robust', 0)
            met = bot_memory.get('metric', 1)
            
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
                print(f"Math Error on rule {best_rule}: {math_error}")
                kc = (1.2 * tm) / (km * taum) 
                ti = 2.0 * taum
                r = {"name": f"Ziegler-Nichols (Fallback from {best_rule})"}

            chart, os_est = simulate_step(kc, ti, km, tm, taum)
            
            # PROFESSIONAL ANALYSIS EXPLANATION
            mode_str = "Setpoint Tracking (Servo)" if mode == 1 else "Disturbance Rejection (Regulatory)"
            os_str = "Minimal/No" if os_val == 0 else "Low" if os_val == 1 else "Medium" if os_val == 2 else "High"
            rob_str = "High Robustness/Stability" if rob == 1 else "Aggressive Performance"
            metric_str = "IAE (Smooth Error)" if met == 1 else "ISE (Aggressive Error)" if met == 2 else "ITAE (Balanced Error)" if met == 3 else "Standard"
            
            final_reply = f"**System Optimization Complete.**<br><br>"
            final_reply += f"**Scenario Analysis:** You requested a **{mode_str}** response prioritizing **{metric_str}** minimization, with a need for **{rob_str}** and **{os_str} Overshoot**.<br><br>"
            final_reply += f"**Rule Selection:** Based on these exact constraints and your parameters (Km={km}, Tm={tm}, Tau={taum}), my AI Decision Engine evaluated the entire database and determined that **{r.get('name')}** is the absolute optimal choice for your scenario.<br><br>"
            final_reply += f"**Proportional Gain (Kc):** {round(kc,4)}<br>**Integral Time (Ti):** {round(ti,4)}s<br>**Estimated Overshoot:** {os_est}%<br><br>Here is your closed-loop step response:"
            
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "ready_to_tune": False}
            
            return jsonify({"reply": final_reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": "An unexpected system error occurred. Please try entering your parameters again.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
