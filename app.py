from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import json
import os

app = Flask(__name__)
CORS(app)

with open('tuning_rules.json', 'r') as file:
    RULES_DB = json.load(file)

# SECURELY LOAD API KEY FROM RENDER ENVIRONMENT
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_API_KEY environment variable not set.")

def get_ai_tuning_recommendation(user_text):
    prompt = f"""
    You are a specialized mathematical router for a Process Control Chatbot.
    Your ONLY job is to read the user's natural language request and match it to the best PI tuning rule from Aidan O'Dwyer's handbook.
    
    Here is your ONLY allowed database of tuning rules for the FOLPD model:
    {json.dumps(RULES_DB["FOLPD"], indent=2)}

    USER REQUEST: "{user_text}"

    REASONING STEPS:
    1. Does the user want disturbance rejection (Regulator) or setpoint tracking (Servo)?
    2. Do they want aggressive speed (ISE), balanced response (IAE), or perfectly flat stability (ITAE)?
    
    INSTRUCTIONS:
    Select the single most appropriate rule_id from the database above.
    You MUST respond ONLY with a raw JSON object. Do not use Markdown, backticks, or write any other text.
    
    FORMAT:
    {{
        "rule_id": "the_id_you_selected",
        "reasoning": "A 2-sentence explanation directly to the user explaining why you chose this rule based on their words."
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.strip().strip('`').replace('json\n', '')
        ai_decision = json.loads(clean_text)
        return ai_decision["rule_id"], ai_decision["reasoning"]
    except Exception as e:
        print(f"AI Error: {e}")
        return "rovira_iae_servo", "I could not fully determine your intent. Defaulting to a balanced IAE response."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    
    rule_id, reasoning = get_ai_tuning_recommendation(user_message)
    selected_rule = RULES_DB["FOLPD"].get(rule_id)
    
    return jsonify({
        "status": "success",
        "rule_id": rule_id,
        "rule_name": selected_rule["name"],
        "explanation": reasoning,
        "next_step": f"To calculate the parameters for the {selected_rule['name']}, please enter your Process Gain (K_m):"
    })

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    rule_id = data.get('rule_id')
    km, tm, taum = float(data['km']), float(data['tm']), float(data['taum'])
    
    rule = RULES_DB["FOLPD"].get(rule_id)
    if not rule:
        return jsonify({"error": "Rule not found"})

    # Evaluate the math strings from the JSON
    try:
        kc = eval(rule['kc_math'], {"km": km, "tm": tm, "taum": taum})
        ti = eval(rule['ti_math'], {"km": km, "tm": tm, "taum": taum})
        
        return jsonify({
            "status": "success",
            "rule_name": rule["name"],
            "kc": round(kc, 4),
            "ti": round(ti, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
