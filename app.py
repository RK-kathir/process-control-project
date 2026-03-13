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

mem = {"km": None, "tm": None, "taum": None}

def extract_params(text):
    """Fuzzy Data Hunter: Finds numbers even with words like 'about' or 'around'"""
    found = {"km": None, "tm": None, "taum": None}
    
    # This Regex looks for the keyword, then any words/spaces, then the number
    k = re.search(r"(?:km|gain|k_m).*?(\d*\.?\d+)", text, re.I)
    t = re.search(r"(?:tm|time constant|lag|t_m).*?(\d*\.?\d+)", text, re.I)
    tm = re.search(r"(?:taum|tau|dead time|delay|tow).*?(\d*\.?\d+)", text, re.I)
    
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
    
    # 1. Handle Reset Command
    if "reset" in text or "clear" in text:
        mem = {"km": None, "tm": None, "taum": None}
        return jsonify({"response": "Memory cleared! What are the new process parameters?"})

    # 2. Update memory
    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    # 3. Decision Logic
    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        # Protection against division by zero
        if mem["tm"] <= 0: return jsonify({"response": "Error: Lag (Tm) must be greater than 0."})
        
        intent = 0 if any(x in text for x in ["fast", "agg"]) else 2 if any(x in text for x in ["smooth", "stab"]) else 1
        rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent]])[0]
        
        # ... (rest of your simulation and response code from the previous version)
        # Ensure you include the simulate_step function here!
        
        # For now, a quick text response to verify it works:
        r = RULES_DB.get(rule_key, RULES_DB['ziegler_nichols'])
        kc = round(eval(r["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        ti = round(eval(r["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        
        return jsonify({"response": f"✅ Found everything! Rule: {r['name']}, Kc: {kc}, Ti: {ti}."})
    
    # 4. Show what is still missing
    missing = [k.upper() for k, v in mem.items() if v is None]
    return jsonify({"response": f"I've noted the gain and delay. I still need: <b>{', '.join(missing)}</b>."})

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
