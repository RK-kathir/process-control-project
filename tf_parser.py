
"""
tf_parser.py
============
Extracts FOPDT (First-Order Plus Dead Time) process parameters
(Km, Tm, Taum) directly from a transfer function expression — no
step-response data or system identification required.
 
Supports:
 - Already-FOPDT transfer functions:   K * exp(-theta*s) / (tau*s + 1)
 - Higher-order transfer functions (2nd through 5th order denominators,
   and beyond), reduced to FOPDT using Skogestad's "half rule" model
   reduction. Root-finding is fully numerical (numpy.roots), so any
   polynomial order is supported - not just degrees sympy can factor
   symbolically.
 - Direct numeric specification:       "Km=2.5, Tm=8, Taum=1.2"
 - Pure gain + delay:                  "5*exp(-2*s)"
 
Usage
-----
    from tf_parser import extract_fopdt_from_text
 
    result = extract_fopdt_from_text("5/((10*s+1)*(2*s+1))*exp(-1.5*s)")
    print(result)
    # {'Km': 5.0, 'Tm': 11.0, 'Taum': 2.5,
    #  'method': 'skogestad_half_rule', 'num_poles_found': 2}
"""
 
import re
import numpy as np
import sympy as sp
 
s = sp.symbols('s')
 
 
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
 
def _strip_delay(expr_str):
    """
    Pull a dead-time term of the form exp(-theta*s) or exp(-theta s)
    out of the expression string.
 
    Returns (remaining_str, theta).
    """
    theta = 0.0
    pattern = r'exp\(\s*-\s*([0-9.]+)\s*\*?\s*s\s*\)'
    match = re.search(pattern, expr_str)
    if match:
        theta = float(match.group(1))
        expr_str = re.sub(pattern, '', expr_str)
        expr_str = re.sub(r'\*\s*\*', '*', expr_str)   # collapse "**" left behind
        expr_str = re.sub(r'\*\s*/', '/', expr_str)    # "5*/(...)" -> "5/(...)"
        expr_str = re.sub(r'/\s*\*', '/', expr_str)    # "(...)/* (...)" -> "(...)/(...)"
        expr_str = expr_str.strip().strip('*').strip()
    return expr_str, theta
 
 
def _poles_to_time_constants(den_poly):
    """
    Given a sympy Poly in `s`, return the time constants tau_i = 1/|pole_i|
    for every finite, non-zero root, sorted largest-first.
 
    Uses numerical root-finding (numpy.roots) rather than sympy.roots, since
    sympy can only solve polynomials symbolically up to degree 4 (and even
    then only for "nice" coefficients) - this would fail for most real-world
    3rd, 4th and 5th order plant models. numpy.roots handles any order.
 
    Complex-conjugate pairs are converted to an equivalent real time
    constant via the pole magnitude (a simplification, but standard
    practice for FOPDT reduction).
    """
    coeffs = [float(c) for c in den_poly.all_coeffs()]  # highest power first
    roots = np.roots(coeffs)
 
    time_constants = []
    for root_c in roots:
        if abs(root_c) < 1e-9:
            continue  # integrator (pole at origin) - not FOPDT-representable
        tau = 1.0 / abs(root_c)
        time_constants.append(tau)
    return sorted(time_constants, reverse=True)
 
 
def fopdt_from_polynomial_tf(num_expr, den_expr, theta0=0.0):
    """
    Reduce a rational transfer function num/den (sympy expressions in `s`)
    plus an existing dead time theta0 down to FOPDT parameters using
    Skogestad's half rule.
    """
    num_poly = sp.Poly(sp.expand(num_expr), s)
    den_poly = sp.Poly(sp.expand(den_expr), s)
 
    # Steady-state (DC) gain
    Km = float(num_poly.eval(0) / den_poly.eval(0))
 
    taus = _poles_to_time_constants(den_poly)
    if not taus:
        raise ValueError("No finite, non-zero poles found in the denominator.")
 
    if len(taus) == 1:
        Tm = taus[0]
        Taum = theta0
        method = "direct"
    else:
        tau1, tau2 = taus[0], taus[1]
        rest = taus[2:]
        # Skogestad's half rule
        Tm = tau1 + tau2 / 2.0
        Taum = theta0 + tau2 / 2.0 + sum(rest)
        method = "skogestad_half_rule"
 
    return {
        "Km": round(float(Km), 6),
        "Tm": round(float(Tm), 6),
        "Taum": round(float(Taum), 6),
        "method": method,
        "num_poles_found": len(taus),
    }
 
 
# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
 
def extract_fopdt_from_text(tf_text):
    """
    Accepts a free-form transfer-function string and returns:
        {'Km': ..., 'Tm': ..., 'Taum': ..., 'method': ..., 'num_poles_found': ...}
 
    Accepted formats (examples):
        "5/(10*s+1)"
        "5*exp(-2*s)/(10*s+1)"
        "5/((10*s+1)*(2*s+1))*exp(-1.5*s)"
        "5*exp(-2*s)"
        "Km=2.5, Tm=8, Taum=1.2"
    """
    tf_text = tf_text.strip()
 
    # --- Case 1: direct numeric specification ---
    direct = re.findall(r'(Km|Tm|Taum)\s*=\s*(-?[0-9.]+)', tf_text, flags=re.IGNORECASE)
    if direct:
        values = {}
        for key, val in direct:
            k = key.lower()
            label = "Taum" if k == "taum" else ("Tm" if k == "tm" else "Km")
            values[label] = float(val)
        if {"Km", "Tm", "Taum"}.issubset(values.keys()):
            values["method"] = "direct_specification"
            values["num_poles_found"] = None
            return values
        # incomplete -> fall through and try TF parsing instead
 
    # --- Case 2: transfer function expression ---
    cleaned, theta0 = _strip_delay(tf_text)
    cleaned = cleaned.replace('^', '**')
 
    try:
        expr = sp.sympify(cleaned)
    except (sp.SympifyError, TypeError, SyntaxError) as e:
        raise ValueError(f"Could not parse transfer function: {tf_text!r} ({e})")
 
    num_expr, den_expr = sp.fraction(sp.together(expr))
 
    if den_expr == 1:
        # pure gain + delay, e.g. "5*exp(-2*s)" -> Tm ~ 0
        Km = float(num_expr.subs(s, 0))
        return {
            "Km": round(Km, 6),
            "Tm": 0.0,
            "Taum": round(theta0, 6),
            "method": "pure_gain_delay",
            "num_poles_found": 0,
        }
 
    return fopdt_from_polynomial_tf(num_expr, den_expr, theta0)
 
 
# ----------------------------------------------------------------------
# Chat-message integration
# ----------------------------------------------------------------------
 
_TF_TRIGGERS = ('transfer function', 'tf:', 'tf=', 'g(s)', 'gp(s)')
 
 
def handle_chat_message(message):
    """
    Call this from your chat / Socket.IO message handler BEFORE your
    normal NLP routing. If the message looks like a transfer-function
    or direct Km/Tm/Taum specification, returns (reply_text, params_dict).
    Otherwise returns (None, None) so the caller falls through to normal
    chat handling.
    """
    lower = message.lower()
    looks_like_tf = any(t in lower for t in _TF_TRIGGERS)
    looks_like_direct = re.search(r'(km|tm|taum)\s*=', lower) is not None
 
    if not (looks_like_tf or looks_like_direct):
        return None, None
 
    # Strip common prefixes like "TF:" / "transfer function =" / "G(s) ="
    expr = re.sub(r'(?i)^(transfer function|tf|g\(s\)|gp\(s\))\s*[:=]?\s*', '', message).strip()
 
    try:
        params = extract_fopdt_from_text(expr)
    except ValueError as e:
        return f"⚠️ Couldn't parse that transfer function: {e}", None
 
    reply = (
        "✅ Process parameters extracted "
        f"(method: {params['method']}):\n"
        f"   Km   = {params['Km']}\n"
        f"   Tm   = {params['Tm']}\n"
        f"   Taum = {params['Taum']}"
    )
    return reply, params
 
 
if __name__ == "__main__":
    # quick self-test - includes 1st through 5th order denominators
    tests = [
        "5/(10*s+1)",                                              # 1st order
        "5*exp(-2*s)/(10*s+1)",                                    # 1st order + delay
        "5/((10*s+1)*(2*s+1))*exp(-1.5*s)",                        # 2nd order + delay
        "5/((8*s+1)*(4*s+1)*(1*s+1))",                             # 3rd order
        "10/((12*s+1)*(6*s+1)*(2*s+1)*(0.5*s+1))*exp(-0.5*s)",     # 4th order + delay
        "8/((15*s+1)*(7*s+1)*(3*s+1)*(1*s+1)*(0.2*s+1))*exp(-1*s)", # 5th order + delay
        "5*exp(-2*s)",                                             # pure gain + delay
        "Km=2.5, Tm=8, Taum=1.2",                                  # direct specification
    ]
    for t in tests:
        print(t, "->", extract_fopdt_from_text(t))
