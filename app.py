from flask import Flask, request, jsonify, render_template
import json, pickle, re, numpy as np
from scipy import signal
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

mem = {"km": None, "tm": None, "taum": None}

# --- FEATURE 1: Step Response Simulator ---
def simulate_step(km, tm, taum, kc, ti):
    t = np.linspace(0, (tm + taum) * 5, 500)
    dt = t[1] - t[0]
    pv = np.zeros_like(t)
    error_sum = 0
    delay_steps = int(taum / dt)
    mv_history = np.zeros_like(t)
    
    for i in range(1, len(t)):
        error = 1.0 - pv[i-1] # Step input = 1.0
        error_sum += error * dt
        mv = kc * (error + (1/ti) * error_sum)
        mv = max(0, min(100, mv)) # Valve limits 0-100%
        mv_history[i] = mv
        
        # Apply process with delay
        delayed_idx = i - delay_steps
        delayed_mv = mv_history[delayed_idx] if delayed_idx >= 0 else 0
        
        # FOPDT Differential Equation: Tm*dpv/dt + pv = Km*MV(t-L)
        dpv = (km * delayed_mv - pv[i-1]) / tm
        pv[i] = pv[i-1] + dpv * dt
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pv, name="Process Value (PV)", line=dict(color='#007bff', width=3)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Setpoint")
    fig.update_layout(title="Closed-Loop Step Response", xaxis_title="Time (s)", yaxis_title="Output", margin=dict(l=20, r=20, t=40, b=20))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# --- FEATURE 3: Model Identification (Tangent Method) ---
def identify_fopdt(times, values):
    try:
        km = max(values) - min(values)
        # Find 28.3% and 63.2% points (Smith's Method)
        t28 = times[np.where(values >= 0.283 * km)[0][0]]
        t63 = times[np.where(values >= 0.632 * km)[0][0]]
        tm = 1.5 * (t63 - t28)
        taum = t63 - tm
        return round(km, 2), round(tm, 2), round(max(0.1, taum), 2)
    except:
        return None

def extract_params(text):
    found = {"km": None, "tm": None, "taum": None}
    k = re.search(r"(?:km|gain|k_m)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    t = re.search(r"(?:tm|time constant|lag|t_m)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    tm = re.search(r"(?:taum|tau|dead time|delay|tow|towm)(?:\s*(?:is|=|of|:)\s*)?(\d*\.?\d+)", text, re.I)
    if k: found["km"] = float(k.group(1))
    if t: found["tm"] = float(t.group(1))
    if tm: found["taum"] = float(tm.group(1))
    return found

@app.route('/')
def home(): return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global mem
    text = request.json.get('text', '').lower()
    
    # Check for raw data (Feature 3)
    raw_data = re.findall(r"\[(\d.*?)\]", text) # Looks for [0,1,2...]
    if raw_data:
        vals = [float(x) for x in raw_data[0].split(',')]
        times = list(range(len(vals)))
        res = identify_fopdt(times, vals)
        if res:
            mem["km"], mem["tm"], mem["taum"] = res
            return jsonify({"response": f"🔎 <b>Model Identified!</b> I've estimated <b>Km={res[0]}, Tm={res[1]}, Tau={res[2]}</b> from your data. What style of response do you need?"})

    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        ratio = round(mem["taum"] / mem["tm"], 3)
        intent = 0 if any(x in text for x in ["fast", "agg"]) else 2 if any(x in text for x in ["smooth", "stab"]) else 1
        rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent]])[0]
        
        if rule_key == "uncontrollable":
            return jsonify({"response": "⚠️ Ratio too high (>2.0). Process is unstable for PI."})

        # --- FEATURE 4: Comparison Mode ---
        comparison = []
        for k, v in RULES_DB.items():
            try:
                ckc = round(eval(v["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 2)
                cti = round(eval(v["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 2)
                comparison.append({"name": v["name"], "kc": ckc, "ti": cti})
            except: continue

        r = RULES_DB[rule_key]
        kc = round(eval(r["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        ti = round(eval(r["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        
        # --- FEATURE 2: Stability Analysis ---
        stability = "Stable" if ratio < 0.5 else "Aggressive" if ratio < 1.0 else "Critical"
        
        chart_json = simulate_step(mem["km"], mem["tm"], mem["taum"], kc, ti)

        resp = f"✅ <b>Best Choice: {r['name']}</b><br>"
        resp += f"Settings: <b>Kc: {kc}, Ti: {ti}s</b><br>"
        resp += f"Stability Status: <b style='color:{'green' if stability=='Stable' else 'orange'}'>{stability}</b> (Ratio: {ratio})<br><br>"
        resp += "<b>Rule Comparison:</b><table style='width:100%; font-size:12px; border-collapse:collapse;'>"
        resp += "<tr><th>Rule</th><th>Kc</th><th>Ti</th></tr>"
        for c in comparison[:3]:
            resp += f"<tr><td>{c['name']}</td><td>{c['kc']}</td><td>{c['ti']}</td></tr>"
        resp += "</table><br><b>Next Step:</b> View the response graph below!"
        
        return jsonify({"response": resp, "chart": chart_json})
    
    return jsonify({"response": "I'm listening. Please provide Km, Tm, and Tau."})

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
