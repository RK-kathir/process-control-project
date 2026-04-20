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

# Simplified memory tracking
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
    ext = {"km": None, "tm": None, "taum": None, "tau_c": None}
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

        # 1. Reset Flow (NO MORE BUTTONS HERE)
        if user_msg_lower == "reset":
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "preferences_set": False}
            return jsonify({
                "reply": "Welcome! I am your AI tuning assistant. Please give me your process parameters (Km, Tm, Tau) and tell me a bit about what you are controlling (like a water tank or a heater).", 
                "options": [], "chart": None
            })

        # 2. Extract Numbers Offline First (Bulletproof Memory)
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        # 3. Dynamic LLM Context Translator (Zero-Knowledge Friendly)
        ai_prompt = f"""
        You are TUNING BOT, a highly empathetic process control expert talking to a user who might have ZERO knowledge of process control math.
        User Message: "{user_msg}"
        Current Memory: {json.dumps({k:v for k,v in bot_memory.items() if v is not None})}

        TASK:
        1. Look for parameters (km, tm, taum, tau_c).
        2. TRANSLATE human words to math features if they state a preference:
           - "fast", "aggressive", "rapid" -> mode=1, overshoot=2, robust=0, metric=2 (ISE)
           - "smooth", "safe", "stable", "no splash" -> mode=1, overshoot=0, robust=1, metric=1 (IAE)
           - "standard", "normal", "balanced" -> mode=0, overshoot=1, robust=0, metric=3 (ITAE)
        3. Determine the 'reply':
           - IF THEY ARE MISSING PARAMETERS: Tell them exactly what you have stored so far, and warmly ask for the missing ones.
           - IF YOU HAVE km, tm, taum BUT NO PREFERENCES: Ask a simple, everyday question based on their context (e.g., if it's a heater, ask if they want it to heat fast or smoothly). Provide up to 3 simple button options.
           - DO NOT MENTION "Simple Calculator" or "Advanced Filter".
        
        OUTPUT ONLY VALID JSON:
        {{
            "km": float/null, "tm": float/null, "taum": float/null,
            "mode": int/null, "overshoot": int/null, "robust": int/null, "metric": int/null,
            "preferences_set": boolean (true if they picked a style),
            "reply": "Conversational reply...",
            "options": [{{"label": "Fast & Aggressive", "val": "fast"}}, {{"label": "Smooth & Safe", "val": "smooth"}}]
        }}
        """
        
        try:
            res = llm_model.generate_content(ai_prompt)
            ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
            
            # Merge AI extracted data with memory
            for k in ["km", "tm", "taum", "mode", "overshoot", "robust", "metric"]:
                if ext.get(k) is not None: bot_memory[k] = ext[k]
            if ext.get("preferences_set"): bot_memory["preferences_set"] = True

            reply_text = ext.get('reply', '')
            options = ext.get('options', [])

        except Exception as e:
            # Fallback if Gemini fails entirely
            print("Gemini API Error, using Regex logic.", e)
            reply_text = ""
            options = []
            
        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # 4. Handle Missing Parameters properly (The "Dumb" Fix)
        if not all([km, tm, taum]):
            missing = []
            if km is None: missing.append("Gain (Km)")
            if tm is None: missing.append("Lag (Tm)")
            if taum is None: missing.append("Dead Time (Tau)")
            
            found = []
            if km is not None: found.append(f"Km={km}")
            if tm is not None: found.append(f"Tm={tm}")
            if taum is not None: found.append(f"Tau={taum}")
            
            if not reply_text or "didn't detect" in reply_text.lower():
                if found:
                    reply_text = f"I recorded **{', '.join(found)}**. To tune this, I still need your **{', '.join(missing)}**."
                else:
                    reply_text = "I need your process parameters to get started. What are your Gain (Km), Lag (Tm), and Dead Time (Tau)?"
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 5. Handle Missing Preferences (Zero-Knowledge Questioning)
        if not bot_memory['preferences_set'] and (bot_memory['mode'] is None or bot_memory['metric'] is None):
            if not reply_text:
                reply_text = "I have your parameters! Do you want this system to react aggressively (faster but might overshoot), or smoothly and safely?"
                options = [
                    {"label": "Fast & Aggressive (ISE)", "val": "fast"},
                    {"label": "Smooth & Safe (IAE)", "val": "smooth"},
                    {"label": "Standard Balance (ITAE)", "val": "standard"}
                ]
            return jsonify({"reply": reply_text, "options": options, "chart": None})

        # 6. Execute AI Brain and Math
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
        
        final_reply = f"**Optimization Complete!**\n\nBased on your requirements, I applied the **{r.get('name')}** tuning rule.\n\n**Proportional Gain (Kc):** {round(kc,4)}\n**Integral Time (Ti):** {round(ti,4)}s\n**Estimated Overshoot:** {os_est}%"
        
        # Reset memory for the next tuning run
        bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "preferences_set": False}
        
        return jsonify({"reply": final_reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": f"SYSTEM ERROR: {str(e)}", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
