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
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})

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

# State Memory
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": None, "overshoot": None, "robust": None, "metric": None,
    "preferences_set": False
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
    return json.dumps(graph_data), round(max(0, (np.max(pv) - 1.0) * 100), 1)

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

        # 1. Reset Command
        if user_msg_lower == "reset":
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "preferences_set": False}
            return jsonify({"reply": "Session reset. Please give me your process parameters (Km, Tm, Tau).", "options": [], "chart": None})

        # 2. LOCAL NLP ROUTER: Offline Intents (Saves Gemini Tokens)
        if "rules" in user_msg_lower or "what do you have" in user_msg_lower:
            reply_text = "**Here is my tuning database (Aidan O'Dwyer):**\n\n"
            reply_text += "**1. Servo & Regulatory:** Ziegler-Nichols, Cohen-Coon, Hazebroek\n"
            reply_text += "**2. Minimum Error (IAE/ISE):** Rovira, Zhuang & Atherton, Wang-Juang-Chan\n"
            reply_text += "**3. Robust Tuning:** Skogestad IMC, AMIGO, Chun et al.\n"
            reply_text += "**4. Direct Synthesis:** Lambda Tuning"
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. LOCAL NLP ROUTER: Hardcoded Preferences (Stops the Looping Bug)
        if "fast" in user_msg_lower or "aggressive" in user_msg_lower or "ise" in user_msg_lower:
            bot_memory.update({"mode": 1, "overshoot": 2, "robust": 0, "metric": 2, "preferences_set": True})
        elif "smooth" in user_msg_lower or "safe" in user_msg_lower or "iae" in user_msg_lower:
            bot_memory.update({"mode": 1, "overshoot": 0, "robust": 1, "metric": 1, "preferences_set": True})
        elif "standard" in user_msg_lower or "balance" in user_msg_lower or "itae" in user_msg_lower:
            bot_memory.update({"mode": 0, "overshoot": 1, "robust": 0, "metric": 3, "preferences_set": True})

        # 4. Extract Numbers Locally
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # 5. Gemini API (Only used if numbers are missing or complex context is given)
        if not all([km, tm, taum]) and not bot_memory['preferences_set']:
            ai_prompt = f"""
            You are a helpful engineering bot. User said: "{user_msg}"
            Memory: {bot_memory}
            If they gave parameters, output them. If they didn't, politely ask for missing Km, Tm, or Tau.
            OUTPUT JSON ONLY: {{"km": float/null, "tm": float/null, "taum": float/null, "reply": "String"}}
            """
            try:
                res = llm_model.generate_content(ai_prompt)
                ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                for k in ["km", "tm", "taum"]:
                    if ext.get(k) is not None: bot_memory[k] = ext[k]
                if not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]):
                    return jsonify({"reply": ext.get('reply', 'I still need your Km, Tm, and Tau parameters.'), "options": [], "chart": None})
            except Exception as e:
                pass # Fall through to local parameter check

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # 6. Check if we have parameters, but need preference
        if all([km, tm, taum]) and not bot_memory['preferences_set']:
            return jsonify({
                "reply": f"Parameters locked in (Km={km}, Tm={tm}, Tau={taum}). How would you like me to tune this system?",
                "options": [
                    {"label": "Fast & Aggressive", "val": "fast"},
                    {"label": "Smooth & Safe", "val": "smooth"},
                    {"label": "Standard Balance", "val": "standard"}
                ],
                "chart": None
            })

        # 7. Ready to Calculate (Loop broken)
        if all([km, tm, taum]) and bot_memory['preferences_set']:
            mode = bot_memory.get('mode', 1)
            os_val = bot_memory.get('overshoot', 0)
            rob = bot_memory.get('robust', 0)
            met = bot_memory.get('metric', 1)
            
            ratio = taum / tm
            features = np.array([[km, tm, taum, ratio, mode, os_val, rob, met]])
            
            best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"
            r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

            safe_env = {"km": km, "tm": tm, "taum": taum, "tau_c": 0, "min": min, "max": max}
            if r.get('kc_math') == "SPECIAL_LOOKUP":
                kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
            else:
                kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
                ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
            
            chart, os_est = simulate_step(kc, ti, km, tm, taum)
            
            final_reply = f"**Rule Applied:** {r.get('name')}\n**Proportional Gain (Kc):** {round(kc,4)}\n**Integral Time (Ti):** {round(ti,4)}s\n**Estimated Overshoot:** {os_est}%"
            
            # Reset memory
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "preferences_set": False}
            
            return jsonify({"reply": final_reply, "chart": chart, "options": []})

        return jsonify({"reply": "I am missing some parameters. Please provide Km, Tm, and Tau.", "options": [], "chart": None})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": "An error occurred while processing.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
