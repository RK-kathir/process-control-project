from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import math

app = Flask(__name__)
CORS(app)

with open('tuning_rules.json', 'r') as file:
    RULES_DB = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_process():
    data = request.json
    try:
        km = float(data.get('km'))
        tm = float(data.get('tm'))
        taum = float(data.get('taum'))
        
        ratio = taum / tm
        suitable_rules = []

        for key, rule in RULES_DB["FOLPD"].items():
            if rule["min_ratio"] <= ratio <= rule["max_ratio"]:
                suitable_rules.append({
                    "id": key,
                    "name": rule["name"],
                    "advantages": rule["advantages"],
                    "disadvantages": rule["disadvantages"]
                })

        return jsonify({
            "status": "success",
            "ratio": round(ratio, 4),
            "suitable_rules": suitable_rules
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Mapping error: {str(e)}"})

@app.route('/api/calculate', methods=['POST'])
def calculate_final():
    data = request.json
    try:
        km = float(data.get('km'))
        tm = float(data.get('tm'))
        taum = float(data.get('taum'))
        rule_id = data.get('rule_id')
        
        rule = RULES_DB["FOLPD"].get(rule_id)
        
        # Execute O'Dwyer symbolic mathematics
        local_vars = {"K_m": km, "T_m": tm, "tau_m": taum, "math": math}
        kc = eval(rule["kc_eq"], {"__builtins__": None}, local_vars)
        ti = eval(rule["ti_eq"], {"__builtins__": None}, local_vars)
        td = eval(rule["td_eq"], {"__builtins__": None}, local_vars)
        
        ratio = taum / tm
        
        # -------------------------------------------------------------
        # PROCESS PARAMETER INSIGHT ENGINE (What if I use it?)
        # -------------------------------------------------------------
        insight = f"<b>Process Parameter Analysis (L/Tau = {round(ratio, 3)}):</b><br>"
        
        if kc > 3.0:
            insight += f"• Your calculated Proportional Gain (Kc = {round(kc, 2)}) is highly aggressive. If you apply this to your physical process, expect intense control valve movements and potential wear.<br>"
        else:
            insight += f"• Your calculated Proportional Gain (Kc = {round(kc, 2)}) is moderate, which will yield a smoother, less wearing control action on your equipment.<br>"
            
        if td > 0:
            insight += f"• Because you selected a PID rule, the Derivative action (Td = {round(td, 2)}s) will actively help anticipate your {taum}s dead time, giving you a faster recovery, but be warned: if your process sensors are noisy, this will amplify the noise."
        else:
            insight += f"• Because you selected a PI rule (Td = 0s), the system will easily ignore sensor noise, but it might recover slightly slower from your {taum}s delay compared to a full PID structure."

        return jsonify({
            "status": "success",
            "kc_eq": rule["kc_eq"],
            "ti_eq": rule["ti_eq"],
            "td_eq": rule["td_eq"],
            "kc": round(kc, 4),
            "ti": round(ti, 4),
            "td": round(td, 4),
            "insight": insight
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Calculation error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
