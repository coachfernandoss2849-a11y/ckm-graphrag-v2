# modules/counterfactual.py
"""
Counterfactual simulation: modify a target variable's Year-5 value,
then extrapolate the change across the full 5-year trajectory so that
GBTM classification (which uses all 5 time points) actually responds.
"""
import copy
import requests

VARIABLE_CONFIG = {
    "SBP": {
        "label": "Systolic BP (mmHg)",
        "unit": "mmHg",
        "min_val": 90,
        "max_val": 200,
        "step": 1,
        "affects": "map",
    },
    "DBP": {
        "label": "Diastolic BP (mmHg)",
        "unit": "mmHg",
        "min_val": 50,
        "max_val": 130,
        "step": 1,
        "affects": "map",
    },
    "Scr": {
        "label": "Serum Creatinine (μmol/L)",
        "unit": "μmol/L",
        "min_val": 40,
        "max_val": 600,
        "step": 1,
        "affects": "egfr",
    },
    "ALB": {
        "label": "Albumin (g/L)",
        "unit": "g/L",
        "min_val": 15,
        "max_val": 60,
        "step": 0.5,
        "affects": "alb",
    },
    "BMI": {
        "label": "BMI (kg/m²)",
        "unit": "kg/m²",
        "min_val": 14.0,
        "max_val": 45.0,
        "step": 0.1,
        "affects": "bmi",
    },
    "HDL": {
        "label": "HDL Cholesterol (mmol/L)",
        "unit": "mmol/L",
        "min_val": 0.3,
        "max_val": 3.0,
        "step": 0.05,
        "affects": "hdl",
    },
}


def _shift_trajectory(original: list, target_y5: float) -> list:
    """
    Shift the entire 5-year trajectory so that Year-5 equals target_y5,
    while preserving the original year-to-year differences (shape).
    This ensures GBTM sees a meaningfully different trajectory.
    """
    delta = target_y5 - original[4]
    # Apply a linearly increasing shift: Year1 gets 1/5 delta, Year5 gets full delta
    return [v + delta * (i + 1) / 5.0 for i, v in enumerate(original)]


def simulate(
    original_inputs: dict,
    variable: str,
    target_value: float,
    r_api_url: str,
    calc_map_fn,
    calc_egfr_fn,
    age: int,
    gender: str,
) -> dict:
    inp = copy.deepcopy(original_inputs)
    cfg = VARIABLE_CONFIG[variable]
    affects = cfg["affects"]

    if affects == "map":
        if variable == "SBP":
            inp["sbp"] = _shift_trajectory(inp["sbp"], target_value)
        else:
            inp["dbp"] = _shift_trajectory(inp["dbp"], target_value)
        inp["map"] = [calc_map_fn(s, d) for s, d in zip(inp["sbp"], inp["dbp"])]

    elif affects == "egfr":
        inp["scr"] = _shift_trajectory(inp["scr"], target_value)
        inp["egfr"] = [calc_egfr_fn(scr, age, gender) for scr in inp["scr"]]

    else:
        inp[affects] = _shift_trajectory(inp[affects], target_value)

    payload = {
        "alb":         inp["alb"],
        "bmi":         inp["bmi"],
        "hdl":         inp["hdl"],
        "map":         inp["map"],
        "egfr":        inp["egfr"],
        "follow_time": inp["follow_time"],
    }
    response = requests.post(r_api_url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()
