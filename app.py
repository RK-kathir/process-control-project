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

bot_memory = {
    "step": "init", 
    "km": None, "tm": None, "taum": None, "tau_c": None,
    "mode": 0, "overshoot": 0, "robust": 0, "metric": 0
}

# HYBRID FALLBACK ENGINE (Regex)
def local_fallback_engine(user_msg):
    msg = user_msg.lower()
    ext = {"km": None, "tm": None, "taum": None, "tau_c": None, "reply": "[Offline Mode] Parameters received."}
    
    km_m = re.search(r'(km|gain)\s*=?\s*(\d+\.?\d*)', msg)
    tm_m = re.search(r'(tm|lag)\s*=?\s*(\d+\.?\d*)', msg)
    tau_m = re.search(r'(tau|dead)\s*=?\s*(\d+\.?\d*)', msg)
    tau_c_m = re.search(r'(tau_c|tauc)\s*=?\s*(\d+\.?\d*)', msg)
    
    if km_m: ext["km"] = float(km_m.group(2))
    if tm_m: ext["tm"] = float(tm_m.group(2))
    if tau_m: ext["taum"] = float(tau_m.group(2))
    if tau_c_m: ext["tau_c"] = float(tau_c_m.group(2))
    
    if not any([ext["km"], ext["tm"], ext["taum"]]):
        ext["reply"] = "[Offline Mode] Please provide Gain (Km), Lag (Tm), and Dead Time (Tau)."
        
    return ext

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
        user_msg_lower = user_msg.lower()
        action_data = request.json.get('action_data', None)

        # 1. Reset Flow
        if user_msg_lower == "reset":
            bot_memory = {"step": "init", "km": None, "tm": None, "taum": None, "tau_c": None, "mode": 0, "overshoot": 0, "robust": 0, "metric": 0}
            return jsonify({
                "reply": "Welcome to TUNING BOT. How would you like to proceed?", 
                "options": [{"label": "Simple Calculator", "val": "path:simple"}, {"label": "Advanced Filter", "val": "path:advanced"}], 
                "chart": None
            })

        # 2. Professional Rule Categorizer
        if "rules" in user_msg_lower or "what do you have" in user_msg_lower:
            categories = {}
            for k, v in rules_db.items():
                mode = str(v.get("mode", "Uncategorized")).capitalize()
                name = v.get("name", k)
                if mode not in categories:
                    categories[mode] = []
                categories[mode].append(name)
            
            reply_text = "I have a comprehensive database of tuning rules, categorized by their engineering specifications:\n\n"
            for mode, rules in categories.items():
                reply_text += f"**{mode} Rules:**\n- " + "\n- ".join(rules) + "\n\n"
            
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. Advanced Decision Tree Filter
        if action_data:
            key, val = action_data.get('key'), action_data.get('val')
            
            if key == "path":
                if val == "simple":
                    bot_memory['step'] = "simple_calc"
                    return jsonify({"reply": "Simple Mode. Please provide Gain (Km), Lag (Tm), and Dead Time (Tau).", "options": []})
                else:
                    bot_memory['step'] = "advanced_calc"
                    return jsonify({"reply": "What is your target operation mode?", "options": [{"label": "Regulator", "val": "mode:0"}, {"label": "Servo", "val": "mode:1"}]})
            
            if key == "mode":
                bot_memory['mode'] = int(val)
                return jsonify({"reply": "What is your overshoot target?", "options": [
                    {"label": "None (0%)", "val": "os:0"}, {"label": "Low (5-10%)", "val": "os:1"}, 
                    {"label": "Medium (20-25%)", "val": "os:2"}, {"label": "High (>25%)", "val": "os:3"}
                ]})
            
            if key == "os":
                bot_memory['overshoot'] = int(val)
                return jsonify({"reply": "Is Robust Tuning required?", "options": [{"label": "Yes", "val": "rob:1"}, {"label": "No", "val": "rob:0"}]})
            
            if key == "rob":
                bot_memory['robust'] = int(val)
                return jsonify({"reply": "Select an Integral Error Metric to minimize:", "options": [
                    {"label": "None", "val": "met:0"}, {"label": "IAE", "val": "met:1"}, {"label": "ISE", "val": "met:2"}, 
                    {"label": "ITAE", "val": "met:3"}, {"label": "ISTSE", "val": "met:4"}, {"label": "ISTES", "val": "met:5"}
                ]})
            
            if key == "met":
                bot_memory['metric'] = int(val)
                return jsonify({"reply": "Filter complete. Please provide your process parameters: Gain (Km), Lag (Tm), and Dead Time (Tau).", "options": []})

        # 4. NLP Extraction with Local Regex Fallback
        try:
            ai_prompt = f'You are TUNING BOT. Extract parameters from: "{user_msg}". \nOUTPUT ONLY valid JSON: {{"km": float/null, "tm": float/null, "taum": float/null, "tau_c": float/null, "reply": "Plain conversational response, no markdown symbols."}}'
            res = llm_model.generate_content(ai_prompt)
            ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
        except Exception as e:
            print("LLM Quota/Error hit. Switching to local regex fallback:", e)
            ext = local_fallback_engine(user_msg)
        
        for k in ["km", "tm", "taum", "tau_c"]:
            if ext.get(k) is not None: bot_memory[k] = ext[k]

        km, tm, taum, tau_c = bot_memory['km'], bot_memory['tm'], bot_memory['taum'], bot_memory['tau_c']
        
        if not all([km, tm, taum]): 
            return jsonify({"reply": ext.get('reply', '').replace('*', ''), "options": []})

        # 5. AI Rule Selection & Math Execution
        if bot_memory['step'] == "simple_calc":
            best_rule = "ziegler_nichols"
        else:
            ratio = taum / tm
            features = np.array([[km, tm, taum, ratio, bot_memory['mode'], bot_memory['overshoot'], bot_memory['robust'], bot_memory['metric']]])
            best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"

        r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))
        
        if "tau_c" in str(r.get('kc_math', '')) and not tau_c:
            return jsonify({"reply": f"The AI selected {r.get('name')}, which requires a desired Closed-Loop Time Constant (Tau_c). Please provide Tau_c.", "options": []})

        safe_env = {"km": km, "tm": tm, "taum": taum, "tau_c": tau_c if tau_c else 0, "min": min, "max": max}
        
        if r.get('kc_math') == "SPECIAL_LOOKUP":
            kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
        else:
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
        
        chart, os_val = simulate_step(kc, ti, km, tm, taum)
        
        reply = f"Tuned using: {r.get('name')}\nKc: {round(kc,4)}\nTi: {round(ti,4)}s\nEstimated Overshoot: {os_val}%"
        
        bot_memory.update({"km": None, "tm": None, "taum": None, "tau_c": None})
        
        return jsonify({"reply": reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL SERVER ERROR:", traceback.format_exc())
        return jsonify({"reply": f"SYSTEM ERROR: {str(e)}. Please check parameters and try again.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
