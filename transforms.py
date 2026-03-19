# transforms.py — Core MMM Transformation Functions
# These are the two building blocks of your Marketing Mix Model.
# Used by: generate_data.py (Phase 2), mmm_model.py (Phase 4), optimizer.py (Phase 6)
#
# Pipeline: raw_spend → adstock_geometric() → hill_saturation() → effective_spend


import numpy as np


def adstock_geometric(spend_array, decay_rate):
    """
    Apply geometric adstock decay to a time series of weekly spend.
    
    What it does:
        Models the idea that advertising effects don't disappear instantly.
        Each week's "effective advertising pressure" is this week's spend
        PLUS a decaying portion of all previous weeks' spend.
    
    Formula:
        adstocked[t] = spend[t] + decay_rate * adstocked[t-1]
    
    Args:
        spend_array: numpy array of weekly spend values (e.g., 104 weeks of TV spend)
        decay_rate:  float between 0 and 1 (from config.py)
                     0.75 for TV = 75% of last week's effect carries forward
                     0.20 for search = only 20% carries forward (fast decay)
    
    Returns:
        numpy array of same length with adstocked values
    
    Interview note:
        "Adstock captures the carryover effect of advertising. A TV ad you saw
        last week still influences your purchase decision this week, but with
        diminishing strength. The decay rate controls how fast that memory fades."
    """
    adstocked = np.zeros(len(spend_array))
    adstocked[0] = spend_array[0]
    
    for t in range(1, len(spend_array)):
        adstocked[t] = spend_array[t] + decay_rate * adstocked[t - 1]
    
    return adstocked


def hill_saturation(x, K, S):
    """
    Apply Hill saturation function to model diminishing returns.
    
    What it does:
        Transforms spend so that low spend gives strong returns
        and high spend gives progressively weaker returns.
        Think of it as: the first $10K of TV spend reaches new eyeballs,
        but the 10th $10K is hitting people who've already seen your ad 5 times.
    
    Formula:
        hill(x) = 1 / (1 + (K / x)^S)
    
    Args:
        x:  numpy array of adstocked spend values (output of adstock_geometric)
        K:  half-saturation point (normalized, 0-1 scale from config.py)
            When x = K, the output is exactly 0.5 (50% of maximum effect)
            Lower K = channel saturates faster (email K=0.30)
            Higher K = channel absorbs more budget before flattening (TV K=0.50)
        S:  slope/shape parameter
            S < 1: concave curve (diminishing returns from dollar one)
            S = 1: standard Michaelis-Menten curve
            S > 1: S-shaped curve (slow start, steep middle, then flattens)
            TV S=2.0 (S-shaped), email S=1.0 (concave)
    
    Returns:
        numpy array of same length, values between 0 and 1
    
    Interview note:
        "The Hill function captures diminishing returns. K tells me where the
        curve hits 50% — for TV that's 0.50, meaning TV needs to reach half
        its maximum observed spend before getting half its maximum effect.
        For email, K=0.30 means it saturates much faster — inbox fatigue
        kicks in at lower spend levels."
    """
    # Avoid division by zero: where x is 0, output should be 0
    x = np.maximum(x, 1e-10)
    
    return 1.0 / (1.0 + (K / x) ** S)


def normalize_spend(spend_array):
    """
    Normalize spend to 0-1 scale. Required before applying Hill function
    because our K values from config.py are on a normalized scale (0.3-1.0).
    
    Args:
        spend_array: raw or adstocked spend values
    
    Returns:
        numpy array scaled to [0, 1] range
    """
    spend_min = spend_array.min()
    spend_max = spend_array.max()
    
    if spend_max == spend_min:
        return np.zeros(len(spend_array))
    
    return (spend_array - spend_min) / (spend_max - spend_min)


def apply_pipeline(spend_array, decay_rate, hill_K, hill_S):
    """
    Full transformation pipeline: raw spend → adstock → normalize → Hill saturation.
    
    This is what you'll call for each channel when generating data or
    building the regression model.
    
    Args:
        spend_array: numpy array of raw weekly spend
        decay_rate:  from config (e.g., 0.75 for TV)
        hill_K:      from config (e.g., 0.50 for TV)
        hill_S:      from config (e.g., 2.0 for TV)
    
    Returns:
        dict with intermediate and final arrays:
            'raw':        original spend
            'adstocked':  after geometric decay
            'normalized': after scaling to 0-1
            'saturated':  after Hill function (this goes into your regression)
    """
    adstocked = adstock_geometric(spend_array, decay_rate)
    normalized = normalize_spend(adstocked)
    saturated = hill_saturation(normalized, hill_K, hill_S)
    
    return {
        'raw': spend_array,
        'adstocked': adstocked,
        'normalized': normalized,
        'saturated': saturated,
    }
