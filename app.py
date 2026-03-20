import os
import re
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import traceback

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1. BOT MEMORY (This fixes the amnesia!)
# =====================================================================
bot_memory = {
    "km": None,
    "tm": None,
    "taum": None,
    "intent": 1 # Default to Neutral
}

# =====================================================================
# 2. BUILD THE AI BRAIN IN MEMORY 
# =====================================================================
print("🧠 Training the AI Brain...")
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
    "hazebroek": {"name": "Hazebroek & Van der Waerden (Disturbance)", "kc_math": "SPECIAL", "ti_math": "SPECIAL"}
}

# --- Hazebroek Lookup Logic ---
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

# --- Physics Simulator ---
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
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pv, name="Process Output", line=dict(color='#007bff', width=3)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", name="Setpoint")
    fig.update_layout(
        title="Step Response Simulation", xaxis_title="Time (seconds)", yaxis_title="Process Value",
        margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    max_pv = np.max(pv)
    overshoot = max(0, (max_pv - 1.0) * 100)
    
    # Bulletproof Plotly Export
    return fig.to_json(), round(overshoot, 1)

@app.route('/')
def home():
    try:
        with open(os.path.join(current_dir, 'templates', 'index.html'), 'r') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Error: index.html not found inside the 'templates' folder!", 404

@app.route('/api/chat', methods=['POST'])
def chat():
    global bot_memory
    
    try:
        user_msg = request.json.get('message', '').lower()
        
        # 1. SMART REGEX: Actively search for specific labels
        km_match = re.search(r'(?:km|gain)\s*(?:is|=|\:)?\s*([-+]?\d*\.\d+|\d+)', user_msg)
        tm_match = re.search(r'(?:tm|lag)\s*(?:is|=|\:)?\s*([-+]?\d*\.\d+|\d+)', user_msg)
        tau_match = re.search(r'(?:tau|dead\s*time|taum)\s*(?:is|=|\:)?\s*([-+]?\d*\.\d+|\d+)', user_msg)
        
        # Update memory if found
        if km_match: bot_memory['km'] = float(km_match.group(1))
        if tm_match: bot_memory['tm'] = float(tm_match.group(1))
        if tau_match: bot_memory['taum'] = float(tau_match.group(1))
        
        # Determine intent and update memory
        if any(word in user_msg for word in ["fast", "quick", "aggressive"]): bot_memory['intent'] = 0
        elif any(word in user_msg for word in ["smooth", "stable", "safe"]): bot_memory['intent'] = 2
        elif any(word in user_msg for word in ["disturbance", "rejection", "outside", "reject"]): bot_memory['intent'] = 3
        
        # 2. CHECK MEMORY
        missing = []
        if bot_memory['km'] is None: missing.append("Km (Gain)")
        if bot_memory['tm'] is None: missing.append("Tm (Lag)")
        if bot_memory['taum'] is None: missing.append("Tau (Dead Time)")
        
        if missing:
            return jsonify({"reply": f"Got it. But I still need: **{', '.join(missing)}**. \n\n(Current memory: Km={bot_memory['km']}, Tm={bot_memory['tm']}, Tau={bot_memory['taum']})", "chart": None})
            
        # 3. IF WE HAVE ALL NUMBERS, DO THE MATH
        km = bot_memory['km']
        tm = bot_memory['tm']
        taum = bot_memory['taum']
        intent = bot_memory['intent']
        
        # Prevent division by zero mathematically
        if km == 0 or tm == 0 or taum == 0:
            return jsonify({"reply": "Error: Process parameters (Km, Tm, Tau) cannot be absolute zero.", "chart": None})

        # Ask the AI Brain (Safely converting to numpy array to prevent warnings)
        rule_key = ai_model.predict(np.array([[km, tm, taum, intent]]))[0]
        
        if rule_key == "uncontrollable":
            return jsonify({"reply": f"Warning: Your Dead Time ({taum}s) is too high compared to Lag ({tm}s). This is delay-dominant and cannot be safely controlled by standard PI tuning.", "chart": None})
        
        rule_data = rules_db[rule_key]
        rule_name = rule_data['name']
        
        if rule_key == "hazebroek":
            kc, ti = calculate_hazebroek(km, tm, taum)
        else:
            variables = {"km": km, "tm": tm, "taum": taum}
            kc = eval(rule_data['kc_math'], {}, variables)
            ti = eval(rule_data['ti_math'], {}, variables)
            
        chart_json, overshoot = simulate_step(kc, ti, km, tm, taum)
        
        # Optional: You can wipe the memory here if you want it to reset after a graph, 
        # but leaving it lets you say "now change km to 12" and it will instantly redraw!
        
        reply_text = (f"Based on your parameters and request, I selected the **{rule_name}** tuning method.\n\n"
                      f"• Controller Gain (Kc): **{round(kc, 3)}**\n"
                      f"• Integral Time (Ti): **{round(ti, 3)}** seconds\n"
                      f"• Simulated Overshoot: **{overshoot}%**")
        
        return jsonify({"reply": reply_text, "chart": chart_json})
        
    except Exception as e:
        # If the server crashes, instead of throwing a 500 error, it will text you the exact python error!
        error_trace = traceback.format_exc()
        return jsonify({"reply": f"SYSTEM CRASH:\n{str(e)}\n\nCheck logs for details.", "chart": None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
