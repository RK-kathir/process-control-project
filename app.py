from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import sqlite3
import sympy as sp

app = Flask(__name__)
CORS(app)

def init_memory():
    conn = sqlite3.connect('ai_memory.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (rule_name TEXT, ratio REAL, status TEXT)''')
    conn.commit()
    conn.close()

init_memory()

with open('tuning_rules.json', 'r') as file:
    RULES_DB = json.load(file)

@app.route('/')
def home():
    # Notice this now points to dashboard.html
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_process():
    data = request.json
    try:
        km = float(sp.sympify(data.get('km')))
        tm = float(sp.sympify(data.get('tm')))
        taum = float(sp.sympify(data.get('taum')))
        
        ratio = taum / tm
        valid_rules = []

        for key, rule in RULES_DB["FOLPD"].items():
            if ratio <= rule["max_ratio"]:
                local_vars = {"K_m": km, "T_m": tm, "tau_m": taum}
                kc = eval(rule["Kc_formula"], {"__builtins__": None}, local_vars)
                ti = eval(rule["Ti_formula"], {"__builtins__": None}, local_vars)
                
                valid_rules.append({
                    "name": rule["method_name"],
                    "kc": round(kc, 3),
                    "ti": round(ti, 3),
                    "advantage": rule["advantage"]
                })

        return jsonify({
            "status": "success",
            "ratio": round(ratio, 3),
            "rules": valid_rules
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Parsing error: {str(e)}"})

@app.route('/api/learn', methods=['POST'])
def learn_from_mistake():
    data = request.json
    conn = sqlite3.connect('ai_memory.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?)", 
              (data['rule_name'], data['ratio'], 'FAILED'))
    conn.commit()
    conn.close()
    return jsonify({"message": "Memory updated. I will adjust future recommendations."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
