from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import pickle

app = Flask(__name__)
CORS(app)

with open('tuning_rules.json', 'r') as f:
    RULES_DB = json.load(f)

with open('ai_brain.pkl', 'rb') as f:
    ai_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_text = data.get('text', '').lower()
    km, tm, taum = float(data['km']), float(data['tm']), float(data['taum'])
    ratio = taum / tm

    # 1. NLP: Determine Intent
    intent_val = 1 # Default neutral
    intent_str = "neutral"
    if any(w in user_text for w in ["fast", "aggressive", "quick"]):
        intent_val = 0
        intent_str = "fast"
    elif any(w in user_text for w in ["smooth", "stable", "flat"]):
        intent_val = 2
        intent_str = "smooth"

    # 2. Ask the trained AI Model
    predicted_rule_key = ai_model.predict([[km, tm, taum, intent_val]])[0]

    if predicted_rule_key == "uncontrollable":
        return jsonify({"explanation": f"L/tau ratio is {ratio:.2f}. This process delay is too massive. PI control will fail."})

    # 3. Calculate Math
    rule = RULES_DB[predicted_rule_key]
    kc = round(eval(rule["kc_math"], {"km": km, "tm": tm, "taum": taum}), 3)
    ti = round(eval(rule["ti_math"], {"km": km, "tm": tm, "taum": taum}), 3)

    # 4. Determine other valid options for the explanation
    other_options = []
    if 0.1 <= ratio <= 1.0:
        opts = ["ziegler_nichols", "zhuang_ise_servo", "rovira_iae_servo", "rovira_itae_servo"]
        other_options = [RULES_DB[r]["name"] for r in opts if r != predicted_rule_key]

    alt_text = f"Other valid options for this ratio: {', '.join(other_options)}." if other_options else "No other standard rules recommended for this ratio."

    explanation = f"""
    <b>AI Analysis:</b> You requested a {intent_str} response. The L/tau ratio is {ratio:.2f}.<br>
    The neural network selected the <b>{rule['name']}</b>.<br><br>
    <i>{alt_text}</i><br><br>
    <b>Controller Parameters:</b><br>
    • <b>Kc = {kc}:</b> Proportional action. For every 1% error, the valve moves {kc}%.<br>
    • <b>Ti = {ti}s:</b> Integral action. Repeats the proportional move every {ti} seconds to kill steady-state error.
    """

    return jsonify({"explanation": explanation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
