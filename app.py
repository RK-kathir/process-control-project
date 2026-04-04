import os
import re
import json
import numpy as np
import traceback
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
generation_config = {"response_mime_type": "application/json"}
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

# 1. BOT MEMORY (Strictly Math)
bot_memory = {
    "km": None, "tm": None, "taum": None, "intent": 1, "history": []
}

# 2. LOCAL FALLBACK ENGINE (The Regex Safety Net)
def local_fallback_engine(user_msg):
    user_msg_lower = user_msg.lower()
    extracted = {"action": "chat", "ai_reply": ""}
    
    # Check for reset
    if "reset" in user_msg_lower or "start over" in user_msg_lower or "new chat" in user_msg_lower:
        extracted["action"] = "reset"
        extracted["ai_reply"] = "[Fallback SLM] Memory cleared. Let's start fresh."
        return extracted

    # Extract Math Parameters using Regex
    km_match = re.search(r'(km|gain|k)\s*(?:=|:|-)?\s*(\d+\.?\d*)', user_msg_lower)
    tm_match = re.search(r'(tm|lag|t)\s*(?:=|:|-)?\s*(\d+\.?\d*)', user_msg_lower)
    tau_match = re.search(r'(tau|dead|delay)\s*(?:=|:|-)?\s*(\d+\.?\d*)', user_msg_lower)
    
    if km_match: extracted["km"] = float(km_match.group(2))
    if tm_match: extracted["tm"] = float(tm_match.group(2))
    if tau_match: extracted["taum"] = float(tau_match.group(2))
    
    # Extract Intent
    if "fast" in user_msg_lower: extracted["intent"] = 0
    elif "smooth" in user_msg_lower: extracted["intent"] = 2
    elif "disturbance" in user_msg_lower or "robust" in user_msg_lower: extracted["intent"] = 3

    # Route Action
    if any(k in extracted for k in ["km", "tm", "taum", "intent"]):
        extracted["action"] = "update"
        extracted["ai_reply"] = "[Fallback SLM] Parameters noted. "
    elif "rule" in user_msg_lower:
        extracted["ai_reply"] = "[Fallback SLM] I use Ziegler-Nichols, Cohen-Coon, and Rovira PI tuning rules."
    else:
        extracted["ai_reply"] = "[Fallback SLM] Hello! The cloud AI is currently cooling down. I am your local backup. Please enter your Km, Tm, and Tau parameters."

    return extracted

# 3. EXPERT DATABASE & MATH SIMULATOR
rules_db = {
    "ziegler_nichols": {"name": "Ziegler-Nichols", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["fast", "neutral"], "kc_math": "(0.9 * tm) / (km * taum)", "ti_math": "3.33 * taum"},
    "cohen_coon": {"name": "Cohen-Coon", "min_ratio": 0.1, "max_ratio": 2.0, "tags": ["fast", "neutral"], "kc_math": "(tm / (km * taum)) * (0.9 + (taum / (12 * tm)))", "ti_math": "taum * ((30 + 3 * (taum / tm)) / (9 + 20 * (taum / tm)))"},
    "rovira": {"name": "Rovira PI", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["smooth", "neutral"], "kc_math": "((0.985 * tm) / (km * taum)) * ((taum / tm)**-0.086)", "ti_math": "tm / (0.608 * (taum / tm)**-0.707)"}
}
intent_map = {0: "fast", 1: "neutral", 2: "smooth", 3: "robust"}

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

    max_pv = np.max(pv)
    overshoot = max(0, (max_pv - 1.0) * 100)

    graph_data = {
        "data": [
            {"x": t.tolist(), "y": pv.tolist(), "type": "scatter", "mode": "lines", "name": "Process Output", "line": {"color": "#007bff", "width": 3}},
            {"x": [t[0], t[-1]], "y": [1.0, 1.0], "type": "scatter", "mode": "lines", "name": "Setpoint", "line": {"color": "red", "dash": "dash", "width": 2}}
        ],
        "layout": {"title": "Step Response", "xaxis": {"title": "Time (s)"}, "yaxis": {"title": "PV"}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}
    }
    return json.dumps(graph_data), round(overshoot, 1)

# 4. FLASK ROUTES
@app.route('/')
def home(): return "API Running", 200

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        bot_memory['history'].append(f"User: {user_msg}")
        if len(bot_memory['history']) > 8: bot_memory['history'] = bot_memory['history'][-8:]
            
        history_text = "\n".join(bot_memory['history'])

        # MAIN CLOUD AI (GEMINI)
        ai_prompt = f"""
        You are a professional Process Control Tuning SLM. 
        CURRENT PARAMETERS: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']}
        HISTORY: {history_text}
        USER: "{user_msg}"
        
        Return JSON exactly like this:
        - "action": "reset", "chat" (for greetings/rules), or "update"
        - "km", "tm", "taum": extract as float or null
        - "intent": 0 (fast), 1 (neutral), 2 (smooth), 3 (robust), or null
        - "ai_reply": Conversational response. Answer greetings. List rules if asked. If they provide numbers but are missing Km, Tm, or Tau, ask for them.
        """
        
        try:
            # Try to use Gemini
            response = llm_model.generate_content(ai_prompt)
            extracted_data = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        except Exception as api_error:
            # HYBRID FALLBACK: If Gemini hits rate limit, fail over to local SLM
            print(f"Cloud AI failed: {api_error}. Triggering Fallback.")
            extracted_data = local_fallback_engine(user_msg)

        action = extracted_data.get('action', 'chat')
        ai_reply = extracted_data.get('ai_reply', 'Understood.')

        # 1. RESET
        if action == "reset":
            bot_memory.update({"km": None, "tm": None, "taum": None, "intent": 1})
            return jsonify({"reply": ai_reply, "chart": None})

        # 2. CHAT
        if action == "chat":
            return jsonify({"reply": ai_reply, "chart": None})

        # 3. UPDATE PARAMETERS
        if action == "update":
            for k in ["km", "tm", "taum", "intent"]:
                if extracted_data.get(k) is not None: bot_memory[k] = extracted_data[k]
            
            missing = [m for m, v in zip(["Gain (Km)", "Lag (Tm)", "Dead Time (Tau)"], [bot_memory['km'], bot_memory['tm'], bot_memory['taum']]) if v is None]
            if missing:
                return jsonify({"reply": f"{ai_reply} I still need: {', '.join(missing)}.", "chart": None})

        # 4. MATH EXECUTION
        km, tm, taum, intent_num = bot_memory['km'], bot_memory['tm'], bot_memory['taum'], bot_memory['intent']
        ratio = taum / tm
        user_tag = intent_map.get(intent_num, "neutral")

        valid_rules = [k for k, r in rules_db.items() if r['min_ratio'] <= ratio <= r['max_ratio'] and user_tag in r['tags']]
        if not valid_rules: valid_rules = ["cohen_coon"]

        best_rule, best_os, best_chart, best_kc, best_ti = None, 999, None, 0, 0
        for key in valid_rules:
            r = rules_db[key]
            tkc = eval(r['kc_math'], {}, {"km": km, "tm": tm, "taum": taum})
            tti = eval(r['ti_math'], {}, {"km": km, "tm": tm, "taum": taum})
            tch, tos = simulate_step(tkc, tti, km, tm, taum)
            if tos < best_os: best_os, best_rule, best_chart, best_kc, best_ti = tos, key, tch, tkc, tti

        fr = rules_db[best_rule]
        final_reply = f"{ai_reply}\n\nSelected Rule: {fr['name']} (Optimized for '{user_tag}')\nKc: {round(best_kc,3)}\nTi: {round(best_ti,3)}s\nSimulated Overshoot: {best_os}%"
        
        # Auto-reset memory for the next problem
        bot_memory.update({"km": None, "tm": None, "taum": None, "intent": 1})
        
        return jsonify({"reply": final_reply, "chart": best_chart})

    except Exception as e:
        return jsonify({"reply": f"SYSTEM CRASH: {str(e)}", "chart": None})

if __name__ == '__main__': app.run(debug=True, port=5000)
