from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json, pickle, re, math, numpy as np
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)
CORS(app)

# Load AI Brain and Textbook Rules
with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

# The Bot's "Short-Term Memory"
mem = {"km": None, "tm": None, "taum": None}

def extract_params(text):
    """Fuzzy Data Hunter: Finds numbers even if there are words in between"""
    found = {"km": None, "tm": None, "taum": None}
    
    k = re.search(r"(?:km|gain|k_m).*?(\d*\.?\d+)", text, re.I)
    t = re.search(r"(?:tm|time constant|lag|t_m).*?(\d*\.?\d+)", text, re.I)
    tm = re.search(r"(?:taum|tau|dead time|delay|tow|towm).*?(\d*\.?\d+)", text, re.I)
    
    if k: found["km"] = float(k.group(1))
    if t: found["tm"] = float(t.group(1))
    if tm: found["taum"] = float(tm.group(1))
    return found

def simulate_step(km, tm, taum, kc, ti):
    """Generates the step response graph AND calculates overshoot"""
    t = np.linspace(0, (tm + taum) * 6, 400)
    dt = t[1] - t[0]
    pv, mv_hist = np.zeros_like(t), np.zeros_like(t)
    err_sum, d_steps = 0, int(taum / dt)
    
    for i in range(1, len(t)):
        err = 1.0 - pv[i-1]
        err_sum += err * dt
        # Avoid division by zero
        ti_val = ti if ti > 0 else 0.001
        mv = max(0, min(100, kc * (err + (1/ti_val) * err_sum)))
        mv_hist[i] = mv
        
        d_idx = i - d_steps
        d_mv = mv_hist[d_idx] if d_idx >= 0 else 0
        pv[i] = pv[i-1] + ((km * d_mv - pv[i-1]) / tm) * dt
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pv, name="Process Output", line=dict(color='#007bff', width=3)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Setpoint")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
    
    # Calculate the Overshoot %
    max_pv = np.max(pv)
    overshoot = max(0, (max_pv - 1.0) * 100)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), round(overshoot, 1)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global mem
    text = request.json.get('text', '').lower()
    
    # Handle Reset Command
    if text.strip() in ["reset", "clear", "start over"]:
        mem = {"km": None, "tm": None, "taum": None}
        return jsonify({"response": "🔄 <b>Memory cleared!</b> What are your new process parameters?"})

    # Update memory from user input
    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    # Check if we have all 3 numbers to run the AI
    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        if mem["tm"] <= 0:
            return jsonify({"response": "⚠️ <b>Math Error:</b> Lag (Tm) must be greater than 0."})

        try:
            # Determine User Intent
            intent = 1 
            if any(x in text for x in ["fast", "agg", "quick"]): intent = 0
            elif any(x in text for x in ["smooth", "stab", "flat"]): intent = 2
            
            # Ask the Random Forest Brain
            rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent]])[0]
            
            if rule_key == "uncontrollable":
                return jsonify({"response": "⚠️ <b>Warning:</b> This process is heavily delay-dominant. Standard PI tuning might lead to instability."})

            # Look up the math formula and calculate (safely)
            r = RULES_DB.get(rule_key)
            if not r:
                return jsonify({"response": f"🚨 <b>Error:</b> The AI picked rule '{rule_key}', but it is missing from tuning_rules.json!"})

            math_env = {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"], "math": math}
            kc = round(eval(r["kc_math"], math_env), 3)
            ti = round(eval(r["ti_math"], math_env), 3)
            
            # Generate the graph and calculate overshoot
            chart_data, overshoot = simulate_step(mem["km"], mem["tm"], mem["taum"], kc, ti)
            
            # Generate the Graph Analysis Text
            if overshoot > 15:
                graph_analysis = f"Notice the sharp peak? This is an <b>aggressive</b> response with <b>{overshoot}% overshoot</b>. It reaches the target fast but might cause wear on your physical valves."
            elif overshoot > 2:
                graph_analysis = f"The curve shows a <b>balanced</b> response with a small <b>{overshoot}% overshoot</b>. It reaches the target efficiently and settles quickly."
            else:
                graph_analysis = f"Look at that smooth curve! This is a <b>highly stable</b> response with practically <b>no overshoot ({overshoot}%)</b>. It is gentle on your equipment."

            # Format the final response
            resp = f"✨ <b>Analysis Complete!</b><br><br>"
            resp += f"<b>Recommended Rule:</b> {r['name']}<br>"
            resp += f"• Controller Gain (Kc): <b>{kc}</b><br>"
            resp += f"• Integral Time (Ti): <b>{ti}</b> seconds<br><br>"
            resp += f"🚀 <b>Next Step:</b> Review the simulated response curve below."
            
            # Send the text, graph, and explanation back to the UI
            return jsonify({
                "response": resp, 
                "chart": chart_data,
                "explanation": graph_analysis
            })
            
        except Exception as e:
            return jsonify({"response": f"🚨 <b>Backend Crash Detected:</b><br>{str(e)}<br><i>Hint: Check your tuning_rules.json math formulas!</i>"})
    
    # If missing data, ask the user for the remaining pieces
    missing = [k.upper() for k, v in mem.items() if v is None]
    return jsonify({"response": f"I've updated my notes. To run the simulation, I still need: <b>{', '.join(missing)}</b>."})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000)
