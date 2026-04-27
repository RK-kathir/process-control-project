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
    print(f"SUCCESS: {len(rules_db)} rules and AI Brain loaded.")
except Exception as e:
    print(f"STARTUP ERROR: {e}")
    rf_model = None
    rules_db = {}

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

        # 1. Reset
        if user_msg_lower == "reset":
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "ready_to_tune": False}
            return jsonify({"reply": "Session reset. Please give me your process parameters (Km, Tm, Tau) and tell me what you are trying to control.", "options": [], "chart": None})

        # 2. DYNAMIC RULE LISTING (Reads ALL rules from your JSON)
        if "rules" in user_msg_lower or "what do you have" in user_msg_lower:
            categories = {"Servo": [], "Regulator": [], "General/Hybrid": []}
            for k, v in rules_db.items():
                mode_val = v.get("mode", -1)
                mode_str = "Servo" if mode_val == 1 else "Regulator" if mode_val == 0 else "General/Hybrid"
                # Pull the clean name, or format the JSON key if name is missing
                rule_name = v.get("name", k.replace("_", " ").title())
                categories[mode_str].append(rule_name)
            
            reply_text = f"**I have {len(rules_db)} tuning rules loaded from Aidan O'Dwyer's handbook.** Here is the complete list:\n\n"
            for mode, rules_list in categories.items():
                if rules_list:
                    reply_text += f"**{mode} Rules:** {', '.join(rules_list)}\n\n"
            
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 3. Extract Numbers Locally First
        regex_data = local_regex_extract(user_msg)
        for k in ["km", "tm", "taum"]:
            if regex_data[k] is not None:
                bot_memory[k] = regex_data[k]

        # 4. LLM Multi-Turn Interview Process
        ai_prompt = f"""
        You are a highly intelligent Process Control Assistant. 
        User says: "{user_msg}"
        Current Memory: {json.dumps({k:v for k,v in bot_memory.items() if v is not None})}

        YOUR GOALS:
        1. Extract any new parameters (km, tm, taum).
        2. If parameters exist, ACT AS AN EXPERT INTERVIEWER. You need to determine the user's specific performance goals to set the following features:
           - mode (1=Servo/Setpoint, 0=Regulator/Disturbance)
           - overshoot (0=None, 1=Low, 2=Medium, 3=High)
           - robust (1=Yes, 0=No)
           - metric (1=IAE, 2=ISE, 3=ITAE)
        3. DO NOT ASK EVERYTHING AT ONCE. Ask natural, contextual questions based on their process. For example, if it's a heater, ask if they want to avoid overshoot to prevent burning. If they answer one question, figure out the next missing feature and ask about that.
        4. When you have asked enough questions to confidently infer the 'mode', 'overshoot', 'robust', and 'metric' values, set "ready_to_tune": true.

        OUTPUT VALID JSON ONLY:
        {{
            "km": float/null, "tm": float/null, "taum": float/null,
            "mode": int/null, "overshoot": int/null, "robust": int/null, "metric": int/null,
            "ready_to_tune": boolean,
            "reply": "Your intelligent, conversational question or statement here."
        }}
        """
        
        try:
            res = llm_model.generate_content(ai_prompt)
            ext = json.loads(res.text.replace('```json', '').replace('```', '').strip())
            
            for k in ["km", "tm", "taum", "mode", "overshoot", "robust", "metric"]:
                if ext.get(k) is not None: bot_memory[k] = ext[k]
            if ext.get("ready_to_tune"): bot_memory["ready_to_tune"] = True
            
            reply_text = ext.get("reply", "Understood. Please continue.")
            
        except Exception as e:
            reply_text = "I received your input, but I need to know: do you want a fast response, or a smooth one with minimal overshoot?"

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']

        # 5. Handle Missing Parameters
        if not all([km, tm, taum]):
            return jsonify({"reply": reply_text + "\n\n*(Note: I am still missing some process parameters like Km, Tm, or Tau)*", "options": [], "chart": None})

        # 6. Not Ready? Keep Interviewing
        if all([km, tm, taum]) and not bot_memory['ready_to_tune']:
            return jsonify({"reply": reply_text, "options": [], "chart": None})

        # 7. EXECUTE MATH & GRAPH (With Crash Prevention)
        if all([km, tm, taum]) and bot_memory['ready_to_tune']:
            mode = bot_memory.get('mode', 1)
            os_val = bot_memory.get('overshoot', 0)
            rob = bot_memory.get('robust', 0)
            met = bot_memory.get('metric', 1)
            
            ratio = taum / tm
            features = np.array([[km, tm, taum, ratio, mode, os_val, rob, met]])
            
            best_rule = rf_model.predict(features)[0] if rf_model else "ziegler_nichols"
            r = rules_db.get(best_rule, rules_db.get("ziegler_nichols", {}))

            safe_env = {"km": km, "tm": tm, "taum": taum, "tau_c": bot_memory.get('tau_c', max(0.1, taum)), "min": min, "max": max}
            
            # BULLETPROOF MATH EXECUTION
            try:
                if r.get('kc_math') == "SPECIAL_LOOKUP":
                    kc, ti = (0.85 * tm) / (km * taum), 2.4 * taum
                else:
                    kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
                    ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
                    if ti <= 0: ti = 0.001
            except Exception as math_error:
                print(f"Math Error on rule {best_rule}: {math_error}")
                # Fallback so the plot STILL renders
                kc = (1.2 * tm) / (km * taum) 
                ti = 2.0 * taum
                r = {"name": "Ziegler-Nichols (Safety Fallback)"}

            chart, os_est = simulate_step(kc, ti, km, tm, taum)
            
            final_reply = f"**Optimization Complete!**\n\nBased on your specific requirements, I have selected the **{r.get('name')}** tuning rule.\n\n**Proportional Gain (Kc):** {round(kc,4)}\n**Integral Time (Ti):** {round(ti,4)}s\n**Estimated Overshoot:** {os_est}%\n\nHere is your closed-loop step response:"
            
            # Reset memory for next run
            bot_memory = {"km": None, "tm": None, "taum": None, "tau_c": None, "mode": None, "overshoot": None, "robust": None, "metric": None, "ready_to_tune": False}
            
            return jsonify({"reply": final_reply, "chart": chart, "options": []})

    except Exception as e:
        print("CRITICAL ERROR:", traceback.format_exc())
        return jsonify({"reply": "An unexpected system error occurred. Please try entering your parameters again.", "options": []})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
