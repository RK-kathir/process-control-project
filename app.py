from flask import Flask, request, jsonify, render_template
import json
import pickle
import re

app = Flask(__name__)

# Load AI and Rules
with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)
with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

# Simple Session Memory (Stores data for the current session)
session_memory = {
    "km": None, "tm": None, "taum": None, "last_rule_key": None
}

def extract_numbers(text):
    """Helper to find numbers in a sentence like 'km is 5'"""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return [float(n) for n in nums]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_text = request.json.get('text', '').lower()
    
    # 1. Update Memory if user provides numbers
    found_nums = extract_numbers(user_text)
    if "km" in user_text and found_nums: session_memory["km"] = found_nums[0]
    if "tm" in user_text and found_nums: session_memory["tm"] = found_nums[-1] # Simple logic
    if "tow" in user_text or "taum" in user_text and found_nums: session_memory["taum"] = found_nums[-1]

    # 2. Logic: If the user explicitly asks for a recommendation OR provides all data
    if session_memory["km"] and session_memory["tm"] and session_memory["taum"]:
        km, tm, taum = session_memory["km"], session_memory["tm"], session_memory["taum"]
        ratio = taum / tm
        
        # Determine Intent
        intent_val = 1
        if any(w in user_text for w in ["fast", "aggressive"]): intent_val = 0
        elif any(w in user_text for w in ["smooth", "stable"]): intent_val = 2
        
        # Get AI Recommendation
        rule_key = ai_model.predict([[km, tm, taum, intent_val]])[0]
        rule = RULES_DB.get(rule_key)
        
        if rule_key == "uncontrollable":
            return jsonify({"response": f"I've analyzed your data (L/tau = {ratio:.2f}). Unfortunately, this process is too delay-dominant for standard O'Dwyer PI rules."})

        # Calculate Parameters
        kc = round(eval(rule["kc_math"], {"km": km, "tm": tm, "taum": taum}), 3)
        ti = round(eval(rule["ti_math"], {"km": km, "tm": tm, "taum": taum}), 3)
        
        response_text = f"Based on the parameters provided ($K_m={km}, T_m={tm}, \\tau_m={taum}$), I recommend the **{rule['name']}**.<br><br>"
        response_text += f"**Calculated Parameters:**<br>• $K_c$: {kc}<br>• $T_i$: {ti} seconds<br><br>"
        response_text += "Would you like to try a different response style (e.g., 'make it smoother'), or calculate for a different rule?"
        
        return jsonify({"response": response_text})

    else:
        # If numbers are missing
        missing = []
        if not session_memory["km"]: missing.append("Km")
        if not session_memory["tm"]: missing.append("Tm")
        if not session_memory["taum"]: missing.append("Tau_m")
        return jsonify({"response": f"I've noted what I can, but I still need: {', '.join(missing)}. Please provide them to continue."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
