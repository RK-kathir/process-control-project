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

# 1. UPGRADED BOT MEMORY (Now requires Scenario & Response Time)
bot_memory = {
    "km": None,
    "tm": None,
    "taum": None,
    "intent": 1,
    "scenario": None, 
    "response_time": None,
    "history": []
}

# 2. LOCAL SLM ENGINE (The Fallback Brain)
def local_slm_engine(user_msg, memory):
    user_msg_lower = user_msg.lower()
    extracted = {"action": "chat", "ai_reply": ""}
    
    # Check for reset
    if "reset" in user_msg_lower or "start over" in user_msg_lower:
        extracted["action"] = "reset"
        extracted["ai_reply"] = "[Local SLM] Memory cleared. Let's start a new process analysis."
        return extracted
        
    # Extract Numbers using Regex (Looks for numbers near keywords)
    km_match = re.search(r'(gain|km).*?(\d+\.?\d*)', user_msg_lower)
    tm_match = re.search(r'(lag|tm).*?(\d+\.?\d*)', user_msg_lower)
    tau_match = re.search(r'(dead|delay|tau).*?(\d+\.?\d*)', user_msg_lower)
    time_match = re.search(r'(response time).*?(\d+\.?\d*)', user_msg_lower)
    
    if km_match: extracted["km"] = float(km_match.group(2))
    if tm_match: extracted["tm"] = float(tm_match.group(2))
    if tau_match: extracted["taum"] = float(tau_match.group(2))
    if time_match: extracted["response_time"] = float(time_match.group(2))
    
    # Extract Intent
    if "fast" in user_msg_lower: extracted["intent"] = 0
    elif "smooth" in user_msg_lower: extracted["intent"] = 2
    elif "disturbance" in user_msg_lower: extracted["intent"] = 3
    
    # Extract Scenario Context
    if any(word in user_msg_lower for word in ["tank", "heater", "valve", "pump", "reactor"]):
        extracted["scenario"] = "identified"

    # Determine action based on if we found new data
    if any(k in extracted for k in ["km", "tm", "taum", "intent", "response_time", "scenario"]):
        extracted["action"] = "update"
        extracted["ai_reply"] = "[Local SLM Fallback] I have noted those parameters. "
    else:
        extracted["ai_reply"] = "[Local SLM Fallback] I am an automated backup system. Please provide process parameters (Gain, Lag, Dead Time), your industrial scenario, and desired response time."

    return extracted

# 3. EXPERT RULE DATABASE
rules_db = {
    "ziegler_nichols": {"name": "Ziegler-Nichols", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["fast", "neutral"], "kc_math": "(0.9 * tm) / (km * taum)", "ti_math": "3.33 * taum"},
    "cohen_coon": {"name": "Cohen-Coon", "min_ratio": 0.1, "max_ratio": 2.0, "tags": ["fast", "neutral"], "kc_math": "(tm / (km * taum)) * (0.9 + (taum / (12 * tm)))", "ti_math": "taum * ((30 + 3 * (taum / tm)) / (9 + 20 * (taum / tm)))"},
    "rovira": {"name": "Rovira PI", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["smooth", "neutral"], "kc_math": "((0.985 * tm) / (km * taum)) * ((taum / tm)**-0.086)", "ti_math": "tm / (0.608 * (taum / tm)**-0.707)"}
}
intent_map = {0: "fast", 1: "neutral", 2: "smooth", 3: "robust"}

# 4. PHYSICS SIMULATOR
def simulate_step(kc, ti, km, tm, taum):
    t = np.linspace(0, (tm + taum) * 6, 400)
    dt = t[1] - t[0]
    pv = np.zeros_like(t)
    mv_hist = np.zeros_like(t)
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
        "layout": {"title": "Step Response Simulation", "xaxis": {"title": "Time (seconds)"}, "yaxis": {"title": "Process Value"}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}
    }
    return json.dumps(graph_data), round(overshoot, 1)

# 5. API ROUTE WITH FALLBACK
@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        bot_memory['history'].append(f"User: {user_msg}")
        if len(bot_memory['history']) > 8:
            bot_memory['history'] = bot_memory['history'][-8:]
            
        history_text = "\n".join(bot_memory['history'])

        # MAIN CLOUD AI (GEMINI)
        ai_prompt = f"""
        You are an advanced industrial Process Control SLM. 
        CURRENT KNOWN PARAMETERS: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']}, Scenario={bot_memory['scenario']}, Response Time={bot_memory['response_time']}
        
        CHAT HISTORY: {history_text}
        USER MESSAGE: "{user_msg}"
        
        Return JSON EXACTLY like this:
        - "action": "reset", "chat", or "update"
        - "km": float or null
        - "tm": float or null
        - "taum": float or null
        - "response_time": float or null (if they mention a target time in seconds)
        - "scenario": "identified" (if they describe a tank, heater, valve, etc.) or null
        - "intent": 0 (fast), 1 (neutral), 2 (smooth), 3 (robust), or null
        - "ai_reply": Your conversational response. If Km, Tm, Tau, Scenario, or Response Time are missing, YOU MUST ASK FOR THEM specifically.
        """
        
        try:
            response = llm_model.generate_content(ai_prompt)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            extracted_data = json.loads(clean_text)
        except Exception as api_error:
            # FALLBACK TRIGGERED! Gemini hit a limit or crashed.
            print(f"Gemini failed, switching to Local SLM: {api_error}")
            extracted_data = local_slm_engine(user_msg, bot_memory)

        action = extracted_data.get('action', 'chat')
        ai_reply = extracted_data.get('ai_reply', 'Understood.')

        # ROUTE 1: RESET
        if action == "reset":
            bot_memory.update({"km": None, "tm": None, "taum": None, "intent": 1, "scenario": None, "response_time": None})
            bot_memory['history'].append(f"Bot: {ai_reply}")
            return jsonify({"reply": ai_reply, "chart": None})

        # ROUTE 2 & 3: CHAT OR UPDATE PARAMETERS
        for key in ["km", "tm", "taum", "intent", "scenario", "response_time"]:
            if extracted_data.get(key) is not None:
                bot_memory[key] = extracted_data[key]

        # SLM DIAGNOSTIC CHECK: Ask for missing context
        missing = []
        if bot_memory['km'] is None: missing.append("Process Gain")
        if bot_memory['tm'] is None: missing.append("Lag Time")
        if bot_memory['taum'] is None: missing.append("Dead Time")
        if bot_memory['scenario'] is None: missing.append("Industrial Scenario (e.g., Tank level, Heater)")
        if bot_memory['response_time'] is None: missing.append("Desired Target Response Time")

        if missing:
            # Force the bot to ask questions if the cloud AI didn't
            if "Local SLM" in ai_reply or not ai_reply.strip():
                ai_reply += f" I still need a bit more context. Could you provide: {', '.join(missing)}?"
            bot_memory['history'].append(f"Bot: {ai_reply}")
            return jsonify({"reply": ai_reply, "chart": None})

        # 6. ALL DATA COLLECTED -> EXPERT FILTERING ENGINE
        km, tm, taum, intent_num = bot_memory['km'], bot_memory['tm'], bot_memory['taum'], bot_memory['intent']
        ratio = taum / tm
        user_tag = intent_map.get(intent_num, "neutral")

        valid_rules = [key for key, rule in rules_db.items() if rule['min_ratio'] <= ratio <= rule['max_ratio'] and user_tag in rule['tags']]
        if not valid_rules: valid_rules = ["cohen_coon"]

        best_rule_key, best_overshoot, best_chart, best_kc, best_ti = None, 999, None, 0, 0
        for key in valid_rules:
            rule = rules_db[key]
            temp_kc = eval(rule['kc_math'], {}, {"km": km, "tm": tm, "taum": taum})
            temp_ti = eval(rule['ti_math'], {}, {"km": km, "tm": tm, "taum": taum})
            temp_chart, temp_overshoot = simulate_step(temp_kc, temp_ti, km, tm, taum)
            
            if temp_overshoot < best_overshoot:
                best_overshoot, best_rule_key, best_chart, best_kc, best_ti = temp_overshoot, key, temp_chart, temp_kc, temp_ti

        final_rule = rules_db[best_rule_key]
        final_reply = (f"{ai_reply}\n\nBased on your scenario and target response time, I filtered the database for '{user_tag}' rules. "
                       f"I simulated the options and selected {final_rule['name']}.\n\n"
                       f"Controller Gain (Kc): {round(best_kc, 3)}\n"
                       f"Integral Time (Ti): {round(best_ti, 3)} seconds\n"
                       f"Simulated Overshoot: {best_overshoot}%")

        bot_memory['history'].append(f"Bot: Selected {final_rule['name']}.")
        return jsonify({"reply": final_reply, "chart": best_chart})

    except Exception as e:
        return jsonify({"reply": f"SYSTEM CRASH: {str(e)}", "chart": None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
