from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json, pickle, re

app = Flask(__name__)
CORS(app)

# Load AI and Rules
with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

# Memory for the current session
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

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    global mem
    mem = {"km": None, "tm": None, "taum": None}
    return jsonify({"status": "success"})

@app.route('/api/chat', methods=['POST'])
def chat():
    global mem
    text = request.json.get('text', '').lower()
    
    # 1. Update memory
    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    # 2. Extract Intent
    intent_val, intent_str = 1, "Balanced"
    if any(w in text for w in ["fast", "aggressive", "quick", "ise"]): 
        intent_val, intent_str = 0, "Fast & Responsive"
    elif any(w in text for w in ["smooth", "stable", "flat", "itae"]): 
        intent_val, intent_str = 2, "Smooth & Stable"

    # 3. Decision Logic
    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        ratio = round(mem["taum"] / mem["tm"], 3)
        rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent_val]])[0]
        
        if rule_key == "uncontrollable":
            return jsonify({"response": f"⚠️ <b>Caution:</b> Your Delay-to-Lag ratio is <b>{ratio}</b>. This process is very difficult to control. Standard rules might fail here."})

        r = RULES_DB[rule_key]
        kc = round(eval(r["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        ti = round(eval(r["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        
        # Neat, Symbol-Free Response
        resp = f"✅ <b>Calculations Complete!</b><br><br>"
        resp += f"For your process with <b>Gain {mem['km']}</b>, <b>Lag {mem['tm']}</b>, and <b>Delay {mem['taum']}</b>:<br><br>"
        resp += f"<b>Recommended Rule:</b> {r['name']}<br>"
        resp += f"<b>Proportional Gain (Kc):</b> {kc}<br>"
        resp += f"<b>Integral Time (Ti):</b> {ti} seconds<br><br>"
        resp += f"🌟 <b>Next Step:</b> Apply these values to your controller. If the response is too 'bouncy', try asking me for a <b>'smoother'</b> result!"
        
        return jsonify({"response": resp})
    
    missing = [k.upper() for k, v in mem.items() if v is None]
    return jsonify({"response": f"I've updated the model with what you provided. To finish, I still need: <b>{', '.join(missing)}</b>."})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000)
