import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestClassifier

print("1. Generating 10,000 FOPDT scenarios using strict O'Dwyer limits...")
data = []
for _ in range(10000):
    km = round(random.uniform(0.5, 5.0), 2)
    tm = round(random.uniform(5.0, 50.0), 2)
    taum = round(random.uniform(1.0, 30.0), 2)
    
    ratio = taum / tm 
    
    # Intent: 0 = Fast, 1 = Neutral, 2 = Smooth
    intent = random.choice([0, 1, 2])
    
    # O'Dwyer Strict Boundaries
    if ratio > 2.0:
        rule = "uncontrollable"
    elif 1.0 < ratio <= 2.0:
        rule = "cohen_coon" # ZN mathematically fails over 1.0
    elif 0.1 <= ratio <= 1.0:
        # The Golden Zone for PI Servos
        if intent == 0:
            rule = "zhuang_ise_servo"
        elif intent == 1:
            rule = "rovira_iae_servo"
        else:
            rule = "rovira_itae_servo"
    else:
        rule = "ziegler_nichols" # Very fast reaction (< 0.1)

    data.append([km, tm, taum, intent, rule])

df = pd.DataFrame(data, columns=['Km', 'Tm', 'Taum', 'Intent', 'Rule'])

print("2. Training the Random Forest...")
X = df[['Km', 'Tm', 'Taum', 'Intent']].values
y = df['Rule'].values

ai_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
ai_model.fit(X, y)

print(f"Training Complete! Accuracy: {ai_model.score(X, y) * 100:.2f}%")

with open('ai_brain.pkl', 'wb') as f:
    pickle.dump(ai_model, f)
print("SUCCESS: 'ai_brain.pkl' created!")
