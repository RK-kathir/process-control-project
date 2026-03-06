from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import math

app = Flask(__name__)
CORS(app)

# Load the exact O'Dwyer symbolic rules database
with open('tuning_rules.json', 'r') as file:
    RULES_DB = json.load(file)

@app.route('/')
def home():
    # Serve the frontend UI
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_tuning():
    data = request.json
    
    try:
        km = float(data.get('km'))
        tm = float(data.get('tm'))
        taum = float(data.get('taum'))
        objective = str(data.get('objective')) 
        
        # 1. Map to the correct rule in the dataset
        rule = RULES_DB["FOLPD"].get(objective)
        if not rule:
            return jsonify({"reply": "Error: Objective not found in the database."})

        # 2. Check the Controllability Ratio (Physics Guardrail)
        ratio = taum / tm
        if ratio > rule["max_ratio"]:
            return jsonify({
                "reply": f"⚠️ **CRITICAL WARNING:** Your dead-time ratio (L/Tau = {round(ratio,2)}) exceeds the safe mathematical limit for {rule['method_name']}. The standard math will fail here."
            })

        # 3. Deterministic Symbolic Execution
        local_vars = {"K_m": km, "T_m": tm, "tau_m": taum}
        kc = eval(rule["Kc_formula"], {"__builtins__": None, "math": math}, local_vars)
        ti = eval(rule["Ti_formula"], {"__builtins__": None, "math": math}, local_vars)
        
        # 4. Format the final output
        response = (
            f"✅ **Tuning Complete**\n\n"
            f"**Method Selected:** {rule['method_name']}\n"
            f"**Objective:** {rule['objective']}\n"
            f"• **Proportional Gain (Kc):** {round(kc, 3)}\n"
            f"• **Integral Time (Ti):** {round(ti, 3)} seconds\n\n"
            f"💡 *Consultant Note:* {rule['warning']}"
        )
        return jsonify({"reply": response})

    except Exception as e:
        return jsonify({"reply": f"Mathematical mapping error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
