"""
tf_parser.py
============
Extracts process parameters (Km, time constants, Taum) directly from 
a transfer function expression of any order.
"""

import re
import numpy as np
import sympy as sp

s = sp.symbols('s')

class TFParseError(Exception):
    """Custom error for when a transfer function cannot be parsed."""
    pass

def _strip_delay(expr_str):
    """Pulls exp(-theta*s) out of the expression string."""
    theta = 0.0
    pattern = r'exp\(\s*-\s*([0-9.]+)\s*\*?\s*s\s*\)'
    match = re.search(pattern, expr_str)
    if match:
        theta = float(match.group(1))
        expr_str = re.sub(pattern, '', expr_str)
        expr_str = re.sub(r'\*\s*\*', '*', expr_str)   
        expr_str = re.sub(r'\*\s*/', '/', expr_str)    
        expr_str = re.sub(r'/\s*\*', '/', expr_str)    
        expr_str = expr_str.strip().strip('*').strip()
    return expr_str, theta

def _poles_to_time_constants(den_poly):
    """Returns time constants from denominator poles."""
    coeffs = [float(c) for c in den_poly.all_coeffs()]
    roots = np.roots(coeffs)
    
    time_constants = []
    warnings = []
    
    for root_c in roots:
        if abs(root_c) < 1e-9:
            warnings.append("Integrator pole (at s=0) detected and bypassed.")
            continue 
        if root_c.real > 1e-9:
            warnings.append(f"Unstable pole (s={root_c.real:.4f}) detected. FOPDT reduction may be inaccurate.")
            
        tau = 1.0 / abs(root_c)
        time_constants.append(tau)
        
    return sorted(time_constants, reverse=True), warnings

def parse_transfer_function(tf_text):
    """
    Main parser expected by app.py.
    Returns: {"order": int, "km": float, "taum": float, "time_constants": [float, ...], "warnings": [str]}
    """
    tf_text = tf_text.strip()
    cleaned, theta0 = _strip_delay(tf_text)
    cleaned = cleaned.replace('^', '**')

    try:
        expr = sp.sympify(cleaned)
    except Exception as e:
        raise TFParseError(f"Syntax error in expression: {e}")

    num_expr, den_expr = sp.fraction(sp.together(expr))

    # Case: Pure gain + delay
    if den_expr == 1:
        Km = float(num_expr.subs(s, 0))
        return {
            "order": 0,
            "km": float(Km),
            "taum": float(theta0),
            "time_constants": [],
            "warnings": ["Pure gain/delay system (no dynamics)."]
        }

    num_poly = sp.Poly(sp.expand(num_expr), s)
    den_poly = sp.Poly(sp.expand(den_expr), s)

    # Steady-state (DC) gain
    try:
        Km = float(num_poly.eval(0) / den_poly.eval(0))
    except ZeroDivisionError:
        raise TFParseError("System has an integrator (pole at s=0) which lacks a steady-state gain.")

    taus, warnings_list = _poles_to_time_constants(den_poly)

    return {
        "order": len(taus),
        "km": float(Km),
        "taum": float(theta0),
        "time_constants": [float(t) for t in taus],
        "warnings": warnings_list
    }
