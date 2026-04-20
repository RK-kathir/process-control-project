import os
import pandas as pd
import numpy as np
import pickle
import json
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENCODING
# ─────────────────────────────────────────────────────────────────────────────
# mode:            0=regulator, 1=servo
# overshoot_tgt:   0=none (0%), 1=low (5-10%), 2=medium (20-25%), 3=high (>25%)
# robust_req:      0=not required, 1=required
# integral_metric: 0=none, 1=iae, 2=ise, 3=itae, 4=istse, 5=istes
# ─────────────────────────────────────────────────────────────────────────────

MODE_MAP      = {"regulator": 0, "servo": 1}
OS_MAP        = {"none": 0, "low": 1, "medium": 2, "high": 3}
METRIC_MAP    = {"none": 0, "iae": 1, "ise": 2, "itae": 3, "istse": 4, "istes": 5}

# Load the rules database so training is always in sync with the JSON
RULES_PATH = os.path.join(current_dir, "tuning_rules.json")
with open(RULES_PATH) as f:
    rules_db = json.load(f)

# Filter out SPECIAL_LOOKUP rules for training (they can't be computed at inference)
TRAINABLE_RULES = {
    k: v for k, v in rules_db.items()
    if v["kc_math"] != "SPECIAL_LOOKUP"
}

print(f"✅ Loaded {len(rules_db)} rules total, {len(TRAINABLE_RULES)} trainable")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("⚙️  Generating 30,000 FOPDT training scenarios …")

data = []
rule_keys = list(TRAINABLE_RULES.keys())

for _ in range(30000):
    km   = round(random.uniform(0.1, 5.0), 3)
    tm   = round(random.uniform(1.0, 50.0), 3)
    taum = round(random.uniform(0.1, 20.0), 3)
    ratio = taum / tm

    # Randomly pick a user intent (mode, overshoot, robust, metric)
    mode_val   = random.choice([0, 1])          # 0=regulator, 1=servo
    os_val     = random.choice([0, 1, 2, 3])    # overshoot target
    robust_val = random.choice([0, 1])           # robust required?
    metric_val = random.choice([0, 1, 2, 3, 4, 5])  # integral metric

    # ── Hard limits from O'Dwyer ──────────────────────────────────────────
    if ratio > 2.0:
        rule = "cohen_coon"   # Only valid option above ratio 2 (boundary)
        data.append([km, tm, taum, ratio, mode_val, os_val, robust_val, metric_val, rule])
        continue

    # ── Find best matching rule ───────────────────────────────────────────
    candidates = []
    for rk, rv in TRAINABLE_RULES.items():
        # Check ratio bounds
        if not (rv["ratio_min"] <= ratio <= rv["ratio_max"]):
            # Allow ±20% relaxation so boundaries don't create cliffs
            if not (rv["ratio_min"] * 0.8 <= ratio <= rv["ratio_max"] * 1.2):
                continue

        score = 0

        # Mode match (most important)
        if MODE_MAP.get(rv["mode"], -1) == mode_val:
            score += 10
        else:
            score -= 5   # Penalise mode mismatch

        # Robust match
        if int(rv["robust"]) == robust_val:
            score += 6
        else:
            score -= 3

        # Overshoot match
        rule_os = OS_MAP.get(rv["overshoot"], 0)
        score += max(0, 5 - abs(rule_os - os_val) * 2)

        # Integral metric match
        rule_metric = METRIC_MAP.get(rv["integral_metric"], 0)
        if rule_metric == metric_val:
            score += 4
        elif metric_val == 0 or rule_metric == 0:
            score += 1  # partial credit when one is "none"

        candidates.append((rk, score))

    if not candidates:
        rule = "ziegler_nichols"  # fallback
    else:
        # Add controlled noise so the forest sees variety
        candidates.sort(key=lambda x: x[1] + random.uniform(-1, 1), reverse=True)
        rule = candidates[0][0]

    data.append([km, tm, taum, ratio, mode_val, os_val, robust_val, metric_val, rule])

df = pd.DataFrame(data, columns=[
    "km", "tm", "taum", "ratio",
    "mode", "overshoot_tgt", "robust_req", "integral_metric",
    "rule"
])

print(f"   Rule distribution (top 15):")
print(df["rule"].value_counts().head(15).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["km", "tm", "taum", "ratio", "mode", "overshoot_tgt", "robust_req", "integral_metric"]
X = df[FEATURES].values
y = df["rule"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("\n🧠 Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train) * 100
test_acc  = model.score(X_test,  y_test)  * 100
print(f"   Train accuracy : {train_acc:.2f}%")
print(f"   Test  accuracy : {test_acc:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
brain_path = os.path.join(current_dir, "ai_brain.pkl")
meta_path  = os.path.join(current_dir, "ai_brain_meta.json")

with open(brain_path, "wb") as f:
    pickle.dump(model, f)
print(f"\n✅ Brain saved  : {brain_path}")

# Save feature names and maps so app.py can use them correctly
meta = {
    "features"        : FEATURES,
    "mode_map"        : MODE_MAP,
    "overshoot_map"   : OS_MAP,
    "metric_map"      : METRIC_MAP,
    "trainable_rules" : list(TRAINABLE_RULES.keys()),
    "train_accuracy"  : round(train_acc, 2),
    "test_accuracy"   : round(test_acc,  2)
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)
print(f"✅ Metadata saved: {meta_path}")
print(f"\n🚀 Done — run  app.py  to start the server.")
