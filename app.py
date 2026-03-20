import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pickle
import json
import re
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load the AI Brain and Rulebook ---
try:
    with open(os.path.join(current_dir, 'ai_brain.pkl'), 'rb') as f:
        ai_model = pickle.load(f)
    with open(os.path.join(current_dir, 'tuning_rules.json'), 'r') as f:
        rules_db = json.load(f)
except FileNotFoundError:
    print("❌ ERROR: Please run 'python train_ai.py' first!")

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
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), round(overshoot, 1)

# --- Serve Frontend directly from Flask (Optional but easier for local testing) ---
@app.route('/')
def home():
    # If index.html is in the same folder, this will serve it.
    with open(os.path.join(current_dir, 'index.html'), 'r') as f:
        return render_template_string(f.read())

# --- API Route ---
@app.route('/api/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_msg)
    
    if len(numbers) < 3:
        return jsonify({"reply": "I need three numbers to tune this: Process Gain (Km), Lag Time (Tm), and Dead Time (Tau).", "chart": None})
    
    km, tm, taum = float(numbers[0]), float(numbers[1]), float(numbers[2])
    
    if any(word in user_msg for word in ["fast", "quick", "aggressive"]): intent = 0
    elif any(word in user_msg for word in ["smooth", "stable", "safe"]): intent = 2
    elif any(word in user_msg for word in ["disturbance", "rejection", "outside", "reject"]): intent = 3
    else: intent = 1 
        
    rule_key = ai_model.predict([[km, tm, taum, intent]])[0]
    
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
    
    reply_text = (f"Based on your parameters and request, I selected the **{rule_name}** tuning method.\n\n"
                  f"• Controller Gain (Kc): **{round(kc, 3)}**\n"
                  f"• Integral Time (Ti): **{round(ti, 3)}** seconds\n"
                  f"• Simulated Overshoot: **{overshoot}%**")
    
    return jsonify({"reply": reply_text, "chart": chart_json})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
