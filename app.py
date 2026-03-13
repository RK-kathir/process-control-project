from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json, pickle, re, numpy as np
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)
CORS(app)

# Load AI and Rules
with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

# Global memory for the session
mem = {"km": None, "tm": None, "taum": None}

def extract_params(text):
    found = {"km": None, "tm": None, "taum": None}
    k = re.search(r"(?:km|gain|k_m)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    t = re.search(r"(?:tm|time constant|lag|t_m)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    tm = re.search(r"(?:taum|tau|dead time|delay|tow|towm)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    if k: found["km"] = float(k.group(1))
    if t: found["tm"] = float(t.group(1))
    if tm: found["taum"] = float(tm.group(1))
    return found

def simulate_step(km, tm, taum, kc, ti):
    t = np.linspace(0, (tm + taum) * 6, 400)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum, d_steps = 0, int(taum / dt)
    for i in range(1, len(t)):
        err = 1.0 - pv[i-1]
        err_sum += err * dt
        mv = max(0, min(100, kc * (err + (1/ti) * err_sum)))
        mv_hist[i] = mv
        d_idx = i - d_steps
        d_mv = mv_history[d_idx] if d_idx >= 0 else 0
        pv[i] = pv[i-1] + ((km * d_mv - pv[i-1]) / tm) * dt
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pv, name="Process Response", line=dict(color='#007bff', width=3)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global mem
    text = request.json.get('text', '').lower()
    
    # Update memory
    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        intent = 0 if any(x in text for x in ["fast", "agg"]) else 2 if any(x in text for x in ["smooth", "stab"]) else 1
        rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent]])[0]
        
        if rule_key == "uncontrollable":
            return jsonify({"response": "⚠️ <b>Warning:</b> This process has too much delay. Standard rules might be unstable."})

        r = RULES_DB[rule_key]
        kc = round(eval(r["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        ti = round(eval(r["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        
        chart_data = simulate_step(mem["km"], mem["tm"], mem["taum"], kc, ti)
        
        resp = f"✅ <b>Tuning Results Found</b><br><br>"
        resp += f"For Gain: {mem['km']}, Lag: {mem['tm']}, Delay: {mem['taum']}:<br><br>"
        resp += f"<b>Selected Rule:</b> {r['name']}<br>"
        resp += f"<b>Controller Gain (Kc):</b> {kc}<br>"
        resp += f"<b>Integral Time (Ti):</b> {ti} seconds<br><br>"
        resp += f"🚀 <b>Next Step:</b> Enter these values into your controller and observe the graph below!"
        
        return jsonify({"response": resp, "chart": chart_data})
    
    missing = [k.upper() for k, v in mem.items() if v is None]
    return jsonify({"response": f"I'm listening! I've noted your data, but I still need: <b>{', '.join(missing)}</b>."})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000)
