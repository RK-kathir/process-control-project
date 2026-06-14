"""
tests/test_tf_parser.py
Run with: pytest tests/test_tf_parser.py  (or python3 -m pytest ...)
"""
 
import pytest
from tf_parser import extract_fopdt_from_text, handle_chat_message
 
 
def test_first_order_no_delay():
    r = extract_fopdt_from_text("5/(10*s+1)")
    assert r["Km"] == 5.0
    assert r["Tm"] == 10.0
    assert r["Taum"] == 0.0
    assert r["method"] == "direct"
 
 
def test_first_order_with_delay():
    r = extract_fopdt_from_text("5*exp(-2*s)/(10*s+1)")
    assert r["Km"] == 5.0
    assert r["Tm"] == 10.0
    assert r["Taum"] == 2.0
 
 
def test_second_order_with_delay_skogestad():
    r = extract_fopdt_from_text("5/((10*s+1)*(2*s+1))*exp(-1.5*s)")
    assert r["Km"] == 5.0
    assert r["Tm"] == pytest.approx(11.0)
    assert r["Taum"] == pytest.approx(2.5)
    assert r["method"] == "skogestad_half_rule"
 
 
def test_third_order():
    # taus = [8, 4, 1] -> Tm = 8 + 4/2 = 10, Taum = 0 + 4/2 + 1 = 3
    r = extract_fopdt_from_text("5/((8*s+1)*(4*s+1)*(1*s+1))")
    assert r["Km"] == pytest.approx(5.0)
    assert r["Tm"] == pytest.approx(10.0)
    assert r["Taum"] == pytest.approx(3.0)
    assert r["num_poles_found"] == 3
 
 
def test_fourth_order_with_delay():
    # taus = [12, 6, 2, 0.5], theta0 = 0.5
    # Tm = 12 + 6/2 = 15, Taum = 0.5 + 6/2 + 2 + 0.5 = 6.0
    r = extract_fopdt_from_text("10/((12*s+1)*(6*s+1)*(2*s+1)*(0.5*s+1))*exp(-0.5*s)")
    assert r["Km"] == pytest.approx(10.0)
    assert r["Tm"] == pytest.approx(15.0)
    assert r["Taum"] == pytest.approx(6.0)
    assert r["num_poles_found"] == 4
 
 
def test_fifth_order_with_delay():
    # taus = [15, 7, 3, 1, 0.2], theta0 = 1
    # Tm = 15 + 7/2 = 18.5, Taum = 1 + 7/2 + 3 + 1 + 0.2 = 8.7
    r = extract_fopdt_from_text(
        "8/((15*s+1)*(7*s+1)*(3*s+1)*(1*s+1)*(0.2*s+1))*exp(-1*s)")
    assert r["Km"] == pytest.approx(8.0)
    assert r["Tm"] == pytest.approx(18.5)
    assert r["Taum"] == pytest.approx(8.7)
    assert r["num_poles_found"] == 5
    assert isinstance(r["Tm"], float)  # not np.float64 - must be JSON-safe
 
 
def test_pure_gain_and_delay():
    r = extract_fopdt_from_text("5*exp(-2*s)")
    assert r["Km"] == 5.0
    assert r["Tm"] == 0.0
    assert r["Taum"] == 2.0
    assert r["method"] == "pure_gain_delay"
 
 
def test_direct_specification():
    r = extract_fopdt_from_text("Km=2.5, Tm=8, Taum=1.2")
    assert r == {"Km": 2.5, "Tm": 8.0, "Taum": 1.2,
                  "method": "direct_specification", "num_poles_found": None}
 
 
def test_invalid_expression_raises():
    with pytest.raises(ValueError):
        extract_fopdt_from_text("this is not a transfer function")
 
 
def test_chat_handler_recognizes_tf():
    reply, params = handle_chat_message("tf: 5/(10*s+1)")
    assert params is not None
    assert params["Km"] == 5.0
    assert "Km   = 5.0" in reply
 
 
def test_chat_handler_ignores_unrelated_message():
    reply, params = handle_chat_message("hello, how are you?")
    assert reply is None
    assert params is None
 
 
def test_chat_handler_recognizes_direct_spec():
    reply, params = handle_chat_message("Km=2.5, Tm=8, Taum=1.2")
    assert params["Tm"] == 8.0
