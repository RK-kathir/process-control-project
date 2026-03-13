from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json, pickle, re

app = Flask(__name__)
CORS(app)

with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

# Global memory (For production, use a proper database or session)
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
def home(): return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global mem
    text = request.json.get('text', '').lower()
    
    # 1. Update memory
    updates = extract_params(text)
    for key in updates:
        if updates[key] is not None: mem[key] = updates[key]

    # 2. Extract Intent
    intent_val, intent_str = 1, "Neutral"
    if any(w in text for w in ["fast", "aggressive", "quick", "ise"]): 
        intent_val, intent_str = 0, "Fast"
    elif any(w in text for w in ["smooth", "stable", "flat", "itae"]): 
        intent_val, intent_str = 2, "Smooth"

    # 3. Decision Logic
    if all(mem[v] is not None for v in ["km", "tm", "taum"]):
        ratio = mem["taum"] / mem["tm"]
        rule_key = ai_model.predict([[mem["km"], mem["tm"], mem["taum"], intent_val]])[0]
        
        if rule_key == "uncontrollable":
            return jsonify({"response": f"Process is uncontrollable ($L/\\tau = {ratio:.2f}$). Delay is too high."})

        r = RULES_DB[rule_key]
        kc = round(eval(r["kc_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        ti = round(eval(r["ti_math"], {"km": mem["km"], "tm": mem["tm"], "taum": mem["taum"]}), 3)
        
        resp = f"Analyzed: $K_m={mem['km']}, T_m={mem['tm']}, \\tau_m={mem['taum']}$.<br>"
        resp += f"AI recommends <b>{r['name']}</b> ({intent_str}).<br>"
        resp += f"<b>Parameters:</b> $K_c = {kc}, T_i = {ti}s$.<br><br><i>Reason: {intent_str} intent selected for $L/\\tau$ ratio {ratio:.2f}.</i>"
        return jsonify({"response": resp})
    
    missing = [k for k, v in mem.items() if v is None]
    return jsonify({"response": f"I've noted what I can. I still need: {', '.join(missing)}."})

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
