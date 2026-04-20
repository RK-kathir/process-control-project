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

bot_memory = {
    "km": None, "tm": None, "taum": None, 
    "preference": None, "history": []
}

rules_db = {
    "ziegler_nichols": {
        "name": "Ziegler-Nichols", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["fast"],
        "kc_math": "(0.9 * tm) / (km * taum)", "ti_math": "3.33 * taum"
    },
    "hazebroek": {
        "name": "Hazebroek & Van der Waerden", "min_ratio": 0.1, "max_ratio": 2.0, "tags": ["robust", "disturbance"],
        "kc_math": "SPECIAL_LOOKUP", "ti_math": "SPECIAL_LOOKUP"
    },
    "rovira_iae": {
        "name": "Rovira Minimum IAE", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["smooth", "iae"],
        "kc_math": "((0.985 * tm) / (km * taum)) * ((taum / tm)**-0.086)", "ti_math": "tm / (0.608 * (taum / tm)**-0.707)"
    },
    "rovira_ise": {
        "name": "Rovira Minimum ISE", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["fast", "ise"],
        "kc_math": "((1.142 * tm) / (km * taum)) * ((taum / tm)**-0.089)", "ti_math": "tm / (0.540 * (taum / tm)**-0.640)"
    },
    "rovira_itae": {
        "name": "Rovira Minimum ITAE", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["smooth", "itae"],
        "kc_math": "((0.859 * tm) / (km * taum)) * ((taum / tm)**-0.097)", "ti_math": "tm / (0.674 * (taum / tm)**-0.680)"
    }
}

def local_fallback_engine(user_msg):
    msg = user_msg.lower()
    ext = {"action": "chat", "ai_reply": ""}
    
    if "reset" in msg or "new chat" in msg:
        return {"action": "reset", "ai_reply": "Memory cleared for a new session."}

    # Added logic so fallback tells you if it's offline instead of giving a dumb reply
    if any(word in msg for word in ["how", "what", "why", "explain"]):
        ext["ai_reply"] = "I am currently running in offline fallback mode, so I can't answer complex questions right now. I can only calculate tuning math. Please provide Km, Tm, and Tau."
        return ext

    km_m = re.search(r'(km|gain)\s*=?\s*(\d+\.?\d*)', msg)
    tm_m = re.search(r'(tm|lag)\s*=?\s*(\d+\.?\d*)', msg)
    tau_m = re.search(r'(tau|dead)\s*=?\s*(\d+\.?\d*)', msg)
    
    if km_m: ext["km"] = float(km_m.group(2))
    if tm_m: ext["tm"] = float(tm_m.group(2))
    if tau_m: ext["taum"] = float(tau_m.group(2))
    
    if "fast" in msg or "ise" in msg: ext["preference"] = "ise"
    elif "smooth" in msg or "itae" in msg: ext["preference"] = "itae"
    elif "iae" in msg: ext["preference"] = "iae"
    elif "hazebroek" in msg or "disturbance" in msg: ext["preference"] = "disturbance"
    elif "ziegler" in msg: ext["preference"] = "fast"

    if any(k in ext for k in ["km", "tm", "taum", "preference"]):
        ext["action"] = "update"
        ext["ai_reply"] = "Parameters received."
    else:
        ext["ai_reply"] = "I am ready. Please provide Km, Tm, and Tau."
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
            "title": "Closed-Loop Step Response", "xaxis": {"title": "Time (s)", "gridcolor": "#444"}, "yaxis": {"title": "Process Variable", "gridcolor": "#444"}, 
            "paper_bgcolor": "transparent", "plot_bgcolor": "transparent", "font": {"color": "#888"},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40}
        }
    }
    return json.dumps(graph_data), round(max(0, (np.max(pv) - 1.0) * 100), 1)

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        user_msg_lower = user_msg.lower()
        
        bot_memory['history'].append(f"User: {user_msg}")
        if len(bot_memory['history']) > 8: bot_memory['history'] = bot_memory['history'][-8:]
        
        if "rules" in user_msg_lower or "what do you have" in user_msg_lower:
            rules_list = ", ".join([r['name'] for r in rules_db.values()])
            return jsonify({"reply": f"I exclusively support the following tuning rules from the database: {rules_list}.", "chart": None})
            
        available_rules = [r['name'] for r in rules_db.values()]
        
        # SMARTER PROMPT: Forces the bot to answer general questions intelligently
        ai_prompt = f"""
        You are TUNING BOT, an expert Process Control AI. 
        CRITICAL RULES: You ONLY know: {available_rules}. 
        
        CURRENT MEMORY: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']}, Preference={bot_memory['preference']}
        USER MESSAGE: "{user_msg}"
        
        INSTRUCTIONS:
        1. If the user asks a question (e.g., "how do you do that", "what is overshoot", "explain"), ANSWER the question clearly and conversationally in the 'ai_reply' field. Do NOT just ask for parameters. You are a tutor.
        2. If the user provides parameters (Km, Tm, Tau), extract them.
        3. Do NOT use markdown like asterisks (*). Keep text plain.
        
        OUTPUT EXCLUSIVELY IN THIS JSON FORMAT:
        {{
            "action": "chat",
            "km": float or null,
            "tm": float or null,
            "taum": float or null,
            "preference": "fast", "smooth", "iae", "ise", "itae", "disturbance", or null,
            "ai_reply": "Your intelligent answer to their question, OR an acknowledgment of parameters."
        }}
        """
        
        try:
            response = llm_model.generate_content(ai_prompt)
            # BULLETPROOF JSON PARSER to prevent fallback engine taking over by mistake
            json_str = response.text
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if match:
                ext = json.loads(match.group(0))
            else:
                ext = json.loads(json_str)
        except Exception as e:
            print("Fallback Triggered:", e)
            ext = local_fallback_engine(user_msg)

        if ext.get('action') == "reset":
            bot_memory.update({"km": None, "tm": None, "taum": None, "preference": None})
            return jsonify({"reply": ext.get('ai_reply', "Ready for a new tuning session.").replace('*', ''), "chart": None})

        # Update memory if valid values exist
        for k in ["km", "tm", "taum", "preference"]:
            if ext.get(k) is not None: bot_memory[k] = ext[k]

        km, tm, taum, pref = bot_memory['km'], bot_memory['tm'], bot_memory['taum'], bot_memory['preference']
        
        # If the user gives parameters but forgets preference
        if all([km, tm, taum]) and not pref:
            return jsonify({"reply": "I have your parameters (Km, Tm, Tau). To select the best rule, what type of response do you want? (Options: Fast, Minimum IAE, Minimum ISE, Minimum ITAE, or Disturbance)", "chart": None})

        # If parameters are missing
        if not all([km, tm, taum]):
            missing = [m for m, v in zip(["Gain (Km)", "Lag (Tm)", "Dead Time (Tau)"], [km, tm, taum]) if v is None]
            
            # If they gave NO parameters (len=3), they are just chatting/asking a question. Just give the answer.
            if len(missing) == 3:
                return jsonify({"reply": ext.get('ai_reply', "How can I help you?").replace('*', ''), "chart": None})
            
            # If they gave partial parameters, ask for the rest.
            return jsonify({"reply": f"{ext.get('ai_reply','').replace('*', '')} I still need: {', '.join(missing)}.", "chart": None})

        # MATH EXECUTION (Only runs when all parameters + preference are available)
        ratio = taum / tm
        valid_rules = [k for k, r in rules_db.items() if pref in r['tags']]
        if not valid_rules: valid_rules = ["ziegler_nichols"]
        
        best_rule = valid_rules[0]
        r = rules_db[best_rule]
        
        if r['kc_math'] == "SPECIAL_LOOKUP":
            kc = (0.85 * tm) / (km * taum) 
            ti = 2.4 * taum
        else:
            safe_env = {"km": km, "tm": tm, "taum": taum, "min": min, "max": max}
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
        
        chart, os_val = simulate_step(kc, ti, km, tm, taum)
        
        clean_ai_reply = ext.get('ai_reply', '').replace('*', '').replace('#', '')
        reply = f"{clean_ai_reply}\n\nTuned using: {r['name']}\nKc: {round(kc,4)}\nTi: {round(ti,4)}s\nEstimated Overshoot: {os_val}%"
        
        bot_memory.update({"km": None, "tm": None, "taum": None, "preference": None})
        return jsonify({"reply": reply, "chart": chart})

    except Exception as e:
        return jsonify({"reply": f"SYSTEM CRASH: {traceback.format_exc()}", "chart": None})

if __name__ == '__main__': app.run(debug=True, port=5000)
