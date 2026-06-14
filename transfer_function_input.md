# Transfer-Function Parameter Extraction — Integration Guide

## What this adds

A new module, `tf_parser.py`, that takes a transfer function (typed as
text) and directly computes `Km`, `Tm`, `Taum` — no step-response data,
no system identification, no Simulink simulation required. It also
exposes a `handle_chat_message()` function so your bot's normal chat
interface can recognize and respond to a transfer function on the spot.

This is **independent** of (and complements) any signal-based or
step-response-based parameter estimation you build later — it's a
direct-input fallback path.

---

## 1. New files to add to your GitHub repo

```
your-repo/
├── tf_parser.py                  <-- NEW (core module, drop in as-is)
├── tests/
│   └── test_tf_parser.py         <-- NEW (unit tests, incl. up to 5th order)
└── docs/
    └── transfer_function_input.md <-- NEW (user-facing docs, optional)
```

**Dependencies:** `sympy` (already needed for the previous version) plus
`numpy` (new — used for numerical root-finding on the denominator
polynomial, which is what makes 3rd/4th/5th-order — and any higher
order — transfer functions work reliably). Add to `requirements.txt`:

```
sympy
numpy
```

## 2. Existing files you need to modify

- **`matlab_bridge.py`** — add the chat hook (see below) and a branch
  that, when `tf_parser` returns params, calls your existing
  `anfis_training.m` pipeline with those `Km/Tm/Taum` values instead of
  values from system identification.

- **Your dashboard front-end** (whatever serves the chat UI) — no
  changes strictly required; the feature works through plain text
  messages. Optionally add a placeholder hint like
  `"e.g. tf: 5/((10s+1)(2s+1))*exp(-1.5s)"` to the chat input box.

---

## 3. Wiring it into `matlab_bridge.py`

Add near your other imports:

```python
from tf_parser import handle_chat_message, extract_fopdt_from_text
```

In your Socket.IO message handler for incoming chat messages, call
`handle_chat_message()` **first**, before your normal NLP/intent logic:

```python
@socketio.on('chat_message')
def on_chat_message(data):
    message = data.get('text', '')

    # 1. Try transfer-function / direct-parameter extraction first
    reply, params = handle_chat_message(message)
    if params is not None:
        # Got Km, Tm, Taum directly from the user's TF
        emit('chat_response', {'text': reply})

        # Kick off the existing ANFIS retraining pipeline with these
        # values, exactly as you would after system identification:
        run_anfis_retraining(params['Km'], params['Tm'], params['Taum'])
        return
    elif reply is not None:
        # Looked like a TF/parameter message but failed to parse
        emit('chat_response', {'text': reply})
        return

    # 2. Fall through to your normal chat handling
    handle_normal_chat(message)
```

`run_anfis_retraining(Km, Tm, Taum)` is whatever function you already
have (or are building) that calls `anfis_training.m` via the MATLAB
Engine and then refreshes the Fuzzy Logic Controller blocks with
`set_param`.

---

## 4. Accepted input formats (what users can type in chat)

| What the user types                                  | What happens |
|-------------------------------------------------------|--------------|
| `tf: 5/(10*s+1)`                                       | First-order, no delay → `Km=5, Tm=10, Taum=0` |
| `tf: 5*exp(-2*s)/(10*s+1)`                             | First-order + delay → `Km=5, Tm=10, Taum=2` |
| `G(s) = 5/((10*s+1)*(2*s+1))*exp(-1.5*s)`              | 2nd order → reduced via Skogestad's half rule |
| `G(s) = 5/((8*s+1)*(4*s+1)*(1*s+1))`                   | 3rd order → reduced via Skogestad's half rule |
| `G(s) = 8/((15*s+1)*(7*s+1)*(3*s+1)*(1*s+1)*(0.2*s+1))*exp(-1*s)` | 5th order → reduced via Skogestad's half rule |
| `Km=2.5, Tm=8, Taum=1.2`                               | Direct specification, bypasses TF parsing entirely |

Notes:
- Denominators of **any order** are supported (root-finding is fully
  numerical via `numpy.roots`, not symbolic). The half rule itself
  generalizes naturally: the two largest time constants form `Tm`, and
  every remaining time constant (plus any existing delay) is folded into
  `Taum`.
- Use `s` as the Laplace variable, `*` for multiplication, `^` or `**`
  for powers, and `exp(-theta*s)` for the dead-time term.
- The parser is forgiving of spacing but **does** need explicit `*`
  for multiplication (sympy doesn't support implicit `5s` — write `5*s`).

---

## 5. About Skogestad's half rule (for higher-order TFs)

Given time constants sorted largest-first, τ1 ≥ τ2 ≥ τ3 ≥ ... and an
existing dead time θ0:

- New (FOPDT) time constant: `Tm = τ1 + τ2/2`
- New (FOPDT) dead time:     `Taum = θ0 + τ2/2 + τ3 + τ4 + ...`
- `Km` is unchanged (it's just the DC gain, `num(0)/den(0)`)

This is a standard, widely-used approximation for reducing
higher-order linear models to FOPDT form for PI/PID tuning. It's a
reasonable engineering approximation, not an exact match — for plants
with dominant complex-conjugate pole pairs (oscillatory/underdamped
dynamics), the magnitude-based time-constant conversion is a
simplification and you may want to sanity-check the resulting `Tm`
against a step response if one is available.

---

## 6. Quick test

```bash
python3 tf_parser.py
```

This runs the built-in self-test with five example inputs and prints
the extracted `Km/Tm/Taum` for each.
