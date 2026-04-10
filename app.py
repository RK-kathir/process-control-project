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
    "intent": 1, "history": []
}

# EXACTLY THE 5 RULES FROM YOUR SCREENSHOT
rules_db = {
    "ziegler_nichols": {
        "name": "Ziegler-Nichols", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["fast", "neutral"],
        "kc_math": "(0.9 * tm) / (km * taum)",
        "ti_math": "3.33 * taum"
    },
    "cohen_coon": {
        "name": "Cohen-Coon", "min_ratio": 0.1, "max_ratio": 2.0, "tags": ["fast", "neutral"],
        "kc_math": "(tm / (km * taum)) * (0.9 + (taum / (12 * tm)))",
        "ti_math": "taum * ((30 + 3 * (taum / tm)) / (9 + 20 * (taum / tm)))"
    },
    "zhuang_atherton": {
        "name": "Zhuang & Atherton (Fast)", "min_ratio": 0.1, "max_ratio": 3.0, "tags": ["fast"],
        "kc_math": "((1.048 * tm) / (km * taum)) * ((taum / tm)**-0.227)",
        "ti_math": "tm / (1.195 - 0.368 * (taum / tm))"
    },
    "rovira": {
        "name": "Rovira (Smooth)", "min_ratio": 0.1, "max_ratio": 1.0, "tags": ["smooth", "neutral"],
        "kc_math": "((0.985 * tm) / (km * taum)) * ((taum / tm)**-0.086)",
        "ti_math": "tm / (0.608 * (taum / tm)**-0.707)"
    },
    "hazebroek": {
        "name": "Hazebroek & Van der Waerden (Disturbance)", "min_ratio": 0.1, "max_ratio": 2.0, "tags": ["robust", "disturbance"],
        "kc_math": "SPECIAL_LOOKUP",
        "ti_math": "SPECIAL_LOOKUP"
    }
}
intent_map = {0: "fast", 1: "neutral", 2: "smooth", 3: "robust", 4: "disturbance"}

def local_fallback_engine(user_msg):
    msg = user_msg.lower()
    ext = {"action": "chat", "ai_reply": ""}
    
    if "reset" in msg or "new chat" in msg:
        return {"action": "reset", "ai_reply": "[Fallback] Memory cleared."}

    km_m = re.search(r'(km|gain)\s*=?\s*(\d+\.?\d*)', msg)
    tm_m = re.search(r'(tm|lag)\s*=?\s*(\d+\.?\d*)', msg)
    tau_m = re.search(r'(tau|dead)\s*=?\s*(\d+\.?\d*)', msg)
    
    if km_m: ext["km"] = float(km_m.group(2))
    if tm_m: ext["tm"] = float(tm_m.group(2))
    if tau_m: ext["taum"] = float(tau_m.group(2))
    
    if "fast" in msg: ext["intent"] = 0
    elif "smooth" in msg: ext["intent"] = 2
    elif "disturbance" in msg or "hazebroek" in msg: ext["intent"] = 4

    if any(k in ext for k in ["km", "tm", "taum", "intent"]):
        ext["action"] = "update"
        ext["ai_reply"] = "[Fallback SLM] Parameters received. "
    else:
        ext["ai_reply"] = "[Fallback SLM] API Quota Exceeded. Please provide parameters."
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
            {"x": t.tolist(), "y": pv.tolist(), "type": "scatter", "name": "Process Output", "line": {"color": "#007bff", "width": 3}},
            {"x": [t[0], t[-1]], "y": [1.0, 1.0], "type": "scatter", "name": "Setpoint", "line": {"color": "red", "dash": "dash"}}
        ],
        "layout": {
            "title": "Step Response", "xaxis": {"title": "Time (s)"}, "yaxis": {"title": "Process Variable"}, 
            "paper_bgcolor": "white", "plot_bgcolor": "white", "margin": {"l": 40, "r": 20, "t": 40, "b": 40}
        }
    }
    return json.dumps(graph_data), round(max(0, (np.max(pv) - 1.0) * 100), 1)

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    try:
        user_msg = request.json.get('message', '')
        bot_memory['history'].append(f"User: {user_msg}")
        if len(bot_memory['history']) > 8: bot_memory['history'] = bot_memory['history'][-8:]
            
        ai_prompt = f"""
        You are a Process Control SLM.
        CURRENT PARAMETERS: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']}
        HISTORY: {" ".join(bot_memory['history'])}
        USER: "{user_msg}"
        
        Extract JSON:
        - "action": "reset", "chat", or "update"
        - "km", "tm", "taum": float or null
        - "intent": 0 (fast), 1 (neutral), 2 (smooth), 3 (robust), 4 (disturbance)
        - "ai_reply": Conversational reply.
        """
        
        try:
            response = llm_model.generate_content(ai_prompt)
            ext = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        except Exception as e:
            print(f"Gemini Error: {e}")
            ext = local_fallback_engine(user_msg)

        if ext.get('action') == "reset":
            bot_memory.update({"km": None, "tm": None, "taum": None, "intent": 1})
            return jsonify({"reply": ext.get('ai_reply', "Memory reset."), "chart": None})

        if ext.get('action') == "update":
            for k in ["km", "tm", "taum", "intent"]:
                if ext.get(k) is not None: bot_memory[k] = ext[k]
            
            missing = [m for m, v in zip(["Gain (Km)", "Lag (Tm)", "Dead Time (Tau)"], [bot_memory['km'], bot_memory['tm'], bot_memory['taum']]) if v is None]
            if missing:
                return jsonify({"reply": f"{ext['ai_reply']} I need the following parameters: {', '.join(missing)}.", "chart": None})

        if ext.get('action') == "chat" and not all([bot_memory['km'], bot_memory['tm'], bot_memory['taum']]):
             return jsonify({"reply": ext['ai_reply'], "chart": None})

        km, tm, taum = bot_memory['km'], bot_memory['tm'], bot_memory['taum']
        user_tag = intent_map.get(bot_memory['intent'], "neutral")
        ratio = taum / tm

        # Filter rules by L/tau ratio and intent
        valid_rules = [k for k, r in rules_db.items() if r['min_ratio'] <= ratio <= r['max_ratio'] and user_tag in r['tags']]
        best_rule = valid_rules[0] if valid_rules else "ziegler_nichols"
        r = rules_db[best_rule]
        
        # MATH EXECUTION WITH SPECIAL LOOKUP BYPASS FOR HAZEBROEK
        if r['kc_math'] == "SPECIAL_LOOKUP":
            kc = (0.85 * tm) / (km * taum) # Safe approximation
            ti = 2.4 * taum
        else:
            safe_env = {"km": km, "tm": tm, "taum": taum, "min": min, "max": max}
            kc = eval(r['kc_math'], {"__builtins__": None}, safe_env)
            ti = eval(r['ti_math'], {"__builtins__": None}, safe_env)
        
        chart, os_val = simulate_step(kc, ti, km, tm, taum)
        
        reply = f"{ext.get('ai_reply', '')}\n\n**Selected Rule:** {r['name']}\n**Kc:** {round(kc,4)}\n**Ti:** {round(ti,4)}s\n**Overshoot:** {os_val}%"
        
        bot_memory.update({"km": None, "tm": None, "taum": None})
        return jsonify({"reply": reply, "chart": chart})

    except Exception as e:
        return jsonify({"reply": f"SYSTEM CRASH: {traceback.format_exc()}", "chart": None})

if __name__ == '__main__': app.run(debug=True, port=5000)
