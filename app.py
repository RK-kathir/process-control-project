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

# Simplified Memory - The LLM handles the context now
bot_memory = {
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "is_simple": False, "mode": None, "overshoot": None, "robust": None, "metric": None
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

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        user_msg_lower = user_msg.lower().strip()

        # 1. Reset Flow
        if user_msg_lower == "reset":
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "is_simple": False, "mode": None, "overshoot": None, "robust": None, "metric": None}
            return jsonify({
                "reply": "Welcome to TUNING BOT. Please provide your process parameters (Km, Tm, Tau). You can also tell me what you are controlling (e.g., a heater, water tank) so I can optimize it for you.", 
                "options": [], "chart": None
            })

        # 2. Professional Rule Listing
        if "rules" in user_msg_lower or "what do you have" in user_msg_lower:
            reply_text = "Here is my tuning database, categorized by Aidan O'Dwyer's handbook:\n\n"
            reply_text += "**1. Process Reaction Curve (Servo & Regulatory)**\n- Ziegler-Nichols, Cohen-Coon, Hazebroek\n"
            reply_text += "**2. Minimum Performance Index (IAE, ISE, ITAE, ITSE)**\n- Rovira, Zhuang & Atherton, Wang-Juang-Chan\n"
            reply_text += "**3. Robust Tuning**\n- Skogestad IMC, AMIGO, Chun et al.\n"
            reply_text += "**4. Direct Synthesis**\n- Lambda Tuning\n\n"
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. Dynamic LLM Context Translator
        ai_prompt = f"""
        You are TUNING BOT, a smart process control assistant. 
        User Message: "{user_msg}"
        Current Memory: {json.dumps(bot_memory)}

        TASK:
        1. Extract Km, Tm, Tau (and Tau_c) if provided.
        2. Determine if the user wants a "Simple Calculator" (Ziegler-Nichols). If they explicitly say "simple" or click it, set "is_simple": true.
        3. If they give natural language preferences (e.g., "smooth", "fast", "aggressive"), translate them to Random Forest numerical features:
           - mode: 0 (Regulator/Disturbance) or 1 (Servo/Setpoint)
           - overshoot: 0 (None), 1 (Low), 2 (Medium), 3 (High)
           - robust: 0 (No), 1 (Yes)
           - metric: 1 (IAE/Smooth), 2 (ISE/Aggressive), 3 (ITAE)
        4. IF you have Km, Tm, Tau, BUT NO PREFERENCES, act as an expert consultant. Look at their context (e.g., if they mentioned a "heater" or "water tank"). Generate a plain-English "reply" asking how they want it to perform. Provide 3 easy-to-understand "options" for them to click.

        OUTPUT ONLY JSON:
        {{
            "km": float/null, "tm": float/null, "taum": float/null, "tau_c": float/null,
            "is_simple": boolean,
            "mode": int/null, "overshoot": int/null, "robust": int/null, "metric": int/null,
            "reply": "Conversational reply, OR a context-specific question (e.g., 'Do you want the water tank to fill rapidly with some splash, or smoothly and safely?')",
            "options": [{{"label": "Simple Calculator", "val": "simple"}}, {{"label": "Fast & Aggressive", "val": "fast response"}}, {{"label": "Smooth & Safe", "val": "smooth and robust"}}] 
        }}
        """
        
        res = llm_model.generate_content(ai_prompt)
        ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
        
        # Update memory
        for k in ["km", "tm", "taum", "tau_c", "is_simple", "mode", "overshoot", "robust", "metric"]:
            if ext.get(k) is not None: 
                bot_memory[k] = ext[k]

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        
        # A. Missing Basic Parameters
        if not all([km, tm, taum]): 
            return jsonify({"reply": ext.get('reply', 'Please provide Km, Tm, and Tau.'), "options": [], "chart": None})

        # B. We have parameters, but we need to know HOW they want it tuned
        if not bot_memory['is_simple'] and (bot_memory['mode'] is None or bot_memory['overshoot'] is None):
            return jsonify({
                "reply": ext.get('reply', 'I have your parameters. Do you want a standard simple calculation, or a specific optimization?'),
                "options": ext.get('options', [
                    {"label": "Simple Calculator", "val": "Simple Calculator"},
                    {"label": "Smooth / No Overshoot", "val": "Smooth response without overshoot"},
                    {"label": "Fast / Aggressive", "val": "Fast response"}
                ]),
                "chart": None
            })

        # C. Ready to Execute Math
        if bot_memory['is_simple']:
            best_rule = "ziegler_nichols"
        else:
            # Provide safe defaults if the LLM didn't extract every single feature perfectly
            mode = bot_memory.get('mode') or 1
            os_val = bot_memory.get('overshoot') or 0
            rob = bot_memory.get('robust') or 0
            met = bot_memory.get('metric') or 1
            
            ratio = taum / tm
            features = np.array([[km, tm, taum, ratio, mode, os_val, rob, met]])
            
            best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"

        r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))
        
        # Tau_C Check for Direct Synthesis
        if "tau_c" in str(r.get('kc_math', '')) and not bot_memory['tau_c']:
            return jsonify({"reply": f"The AI optimized this to **{r.get('name')}**. This rule requires a desired Closed-Loop Time Constant (Tau_c). What should Tau_c be?", "options": [], "chart": None})

        safe_env = {"km": km, "tm": tm, "taum": taum, "tau_c": bot_memory['tau_c'] if bot_memory['tau_c'] else 0, "min": min, "max": max}
        
        if r.get('kc_math') == "SPECIAL_LOOKUP":
            kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
        else:
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
        
        chart, os_est = simulate_step(kc, ti, km, tm, taum)
        
        reply = f"**Optimization Complete!**\n\n**Rule Applied:** {r.get('name')}\n**Proportional Gain (Kc):** {round(kc,4)}\n**Integral Time (Ti):** {round(ti,4)}s\n**Estimated Overshoot:** {os_est}%"
        
        # Wipe memory for the next calculation
        bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "is_simple": False, "mode": None, "overshoot": None, "robust": None, "metric": None}
        
        return jsonify({"reply": reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": f"An error occurred. Let's start over. Provide your parameters.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
