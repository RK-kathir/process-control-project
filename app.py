import os
import google.generativeai as genai
import random
import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import traceback

app = Flask(__name__)
CORS(app)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
generation_config = {"response_mime_type": "application/json"}
llm_model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. BOT MEMORY (Now with History)
bot_memory = {
    "km": None,
    "tm": None,
    "taum": None,
    "intent": 1,
    "history": [] 
}

# 2. TRAIN THE AI BRAIN
print("Training the AI Brain...")
data = []
for _ in range(5000):
    _km = round(random.uniform(0.1, 5.0), 2)
    _tm = round(random.uniform(1.0, 50.0), 2)
    _taum = round(random.uniform(0.1, 20.0), 2)
    _intent = random.choice([0, 1, 2, 3]) 
    _ratio = _taum / _tm
    
    if _ratio > 2.0: _rule = "uncontrollable" 
    elif _ratio > 1.0: _rule = "cohen_coon"     
    else:
        if _intent == 0: _rule = "zhuang_atherton" 
        elif _intent == 2: _rule = "rovira"          
        elif _intent == 3: _rule = "hazebroek" 
        else: _rule = "ziegler_nichols" 
            
    data.append([_km, _tm, _taum, _intent, _rule])

df = pd.DataFrame(data, columns=['km', 'tm', 'taum', 'intent', 'rule'])
X = df[['km', 'tm', 'taum', 'intent']] 
y = df['rule']                         

ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
ai_model.fit(X.values, y)

rules_db = {
    "ziegler_nichols": {"name": "Ziegler-Nichols", "kc_math": "(0.9 * tm) / (km * taum)", "ti_math": "3.33 * taum"},
    "cohen_coon": {"name": "Cohen-Coon", "kc_math": "(tm / (km * taum)) * (0.9 + (taum / (12 * tm)))", "ti_math": "taum * ((30 + 3 * (taum / tm)) / (9 + 20 * (taum / tm)))"},
    "zhuang_atherton": {"name": "Zhuang & Atherton (Fast)", "kc_math": "((1.048 * tm) / (km * taum)) * ((taum / tm)**-0.227)", "ti_math": "tm / (1.195 - 0.368 * (taum / tm))"},
    "rovira": {"name": "Rovira (Smooth)", "kc_math": "((0.985 * tm) / (km * taum)) * ((taum / tm)**-0.086)", "ti_math": "tm / (0.608 * (taum / tm)**-0.707)"},
    "hazebroek": {"name": "Hazebroek (Disturbance)", "kc_math": "SPECIAL", "ti_math": "SPECIAL"}
}

# 3. HAZEBROEK LOGIC
def calculate_hazebroek(km, tm, taum):
    ratio = taum / tm
    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]
    alphas = [0.68, 0.70, 0.72, 0.74, 0.76, 0.79, 0.81, 0.84, 0.87, 0.90, 0.93, 0.96, 0.99, 1.02, 1.06, 1.09, 1.13, 1.17, 1.20, 1.28, 1.36, 1.45, 1.53, 1.62, 1.71, 1.81]
    betas  = [7.14, 4.76, 3.70, 3.03, 2.50, 2.17, 1.92, 1.75, 1.61, 1.49, 1.41, 1.32, 1.25, 1.19, 1.14, 1.10, 1.06, 1.03, 1.00, 0.95, 0.91, 0.88, 0.85, 0.83, 0.81, 0.80]

    if ratio > 3.5:
        alpha = 0.5 * ratio + 0.1
        beta = taum / (1.6 * taum - 1.2 * tm)
    else:
        alpha = np.interp(ratio, ratios, alphas)
        beta = np.interp(ratio, ratios, betas)
        
    kc = (tm / (km * taum)) * alpha
    ti = taum * beta
    return kc, ti

# 4. PHYSICS SIMULATOR
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
        "layout": {"title": "Step Response Simulation", "xaxis": {"title": "Time (seconds)"}, "yaxis": {"title": "Process Value"}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}
    }
    return json.dumps(graph_data), round(overshoot, 1)

# 5. FLASK ROUTES
@app.route('/')
def home():
    return "API is running. UI is hosted on GitHub Pages.", 200

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    
    try:
        user_msg = request.json.get('message', '')
        
        # Save to memory
        bot_memory['history'].append(f"User: {user_msg}")
        if len(bot_memory['history']) > 8:
            bot_memory['history'] = bot_memory['history'][-8:]
            
        history_text = "\n".join(bot_memory['history'])
        
        # ADVANCED NLP PROMPT
        ai_prompt = f"""
        You are an advanced industrial Process Control Engineering Assistant. Talk to the user naturally.
        
        YOUR SYSTEM KNOWLEDGE:
        If the user asks about the rules or ranges you use, explain these clearly:
        - Ziegler-Nichols (Used for Default/Neutral intent)
        - Cohen-Coon (Used when Dead Time to Lag ratio is > 1.0)
        - Zhuang & Atherton (Used for Fast intent, Ratio <= 1.0)
        - Rovira (Used for Smooth intent, Ratio <= 1.0)
        - Hazebroek & Van der Waerden (Used for Disturbance rejection, Ratio <= 1.0)
        - Uncontrollable (If Ratio is > 2.0, the delay is too high for safe PI control)
        
        CURRENT KNOWN PARAMETERS: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']}
        
        CHAT HISTORY:
        {history_text}
        
        USER MESSAGE: "{user_msg}"
        
        Read the user message. Return a JSON object with these EXACT keys:
        - "action": Use "reset" if they ask to clear memory/start over. Use "chat" for questions or greetings.
        - "km": extract Process Gain if mentioned (float), else null
        - "tm": extract Lag Time if mentioned (float), else null
        - "taum": extract Dead Time if mentioned (float), else null
        - "intent": 0 (fast), 1 (neutral), 2 (smooth), 3 (disturbance), else null
        - "ai_reply": Your human-like, conversational response. Answer their questions about rules/ranges if asked. Acknowledge greetings.
        """
        
        response = llm_model.generate_content(ai_prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        extracted_data = json.loads(clean_text)
        
        action = extracted_data.get('action', 'chat')
        ai_reply = extracted_data.get('ai_reply', 'Understood.')
        
        # HANDLE RESET
        if action == "reset":
            bot_memory['km'] = None
            bot_memory['tm'] = None
            bot_memory['taum'] = None
            bot_memory['intent'] = 1
            bot_memory['history'].append(f"Bot: {ai_reply}")
            return jsonify({"reply": ai_reply, "chart": None})
            
        # UPDATE PARAMETERS
        if extracted_data.get('km') is not None: bot_memory['km'] = float(extracted_data['km'])
        if extracted_data.get('tm') is not None: bot_memory['tm'] = float(extracted_data['tm'])
        if extracted_data.get('taum') is not None: bot_memory['taum'] = float(extracted_data['taum'])
        if extracted_data.get('intent') is not None: bot_memory['intent'] = int(extracted_data['intent'])
        
        missing = []
        if bot_memory['km'] is None: missing.append("Process Gain (Km)")
        if bot_memory['tm'] is None: missing.append("Lag Time (Tm)")
        if bot_memory['taum'] is None: missing.append("Dead Time (Tau)")
        
        # IF STILL CHATTING OR MISSING DATA
        if missing or action == "chat":
            if not missing and action == "chat":
                pass # Proceed to math if everything is here but they were just chatting
            else:
                bot_memory['history'].append(f"Bot: {ai_reply}")
                return jsonify({"reply": ai_reply, "chart": None})
            
        # EXECUTE MATH
        km, tm, taum, intent = bot_memory['km'], bot_memory['tm'], bot_memory['taum'], bot_memory['intent']
        
        if km == 0 or tm == 0 or taum == 0:
            err_msg = "Error: Process parameters cannot be zero."
            return jsonify({"reply": err_msg, "chart": None})

        rule_key = ai_model.predict(np.array([[km, tm, taum, intent]]))[0]
        
        if rule_key == "uncontrollable":
            warn_msg = f"{ai_reply}\n\nWarning: Your Dead Time ({taum}s) is too high compared to Lag ({tm}s). This cannot be safely controlled by standard PI tuning."
            bot_memory['history'].append(f"Bot: Issued uncontrollable warning.")
            return jsonify({"reply": warn_msg, "chart": None})
        
        rule_data = rules_db[rule_key]
        
        if rule_key == "hazebroek": kc, ti = calculate_hazebroek(km, tm, taum)
        else:
            kc = eval(rule_data['kc_math'], {}, {"km": km, "tm": tm, "taum": taum})
            ti = eval(rule_data['ti_math'], {}, {"km": km, "tm": tm, "taum": taum})
            
        chart_json, overshoot = simulate_step(kc, ti, km, tm, taum)
        l_tau_ratio = taum / tm
        
        final_reply = (f"{ai_reply}\n\nI ran the numbers. The Random Forest selected the {rule_data['name']} tuning method.\n"
                      f"L/τ Ratio: {round(l_tau_ratio, 3)}\n"
                      f"Controller Gain (Kc): {round(kc, 3)}\n"
                      f"Integral Time (Ti): {round(ti, 3)} seconds\n"
                      f"Simulated Overshoot: {overshoot}%")
                      
        bot_memory['history'].append(f"Bot: Simulated successfully.")
        return jsonify({"reply": final_reply, "chart": chart_json})
        
    except Exception as e:
        return jsonify({"reply": f"SYSTEM CRASH: {str(e)}", "chart": None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
