# config.py — MMM Simulation Parameters (FINAL)
# Validated against: Joseph MPRA 7683, Meta Robyn docs & community,
# Google Meridian, Recast, Improvado, PyMC-Marketing, Analytic Partners ROI Genome.
# Full citation trail: see mmm_parameter_reference_FINAL.xlsx → Sources tab.

import numpy as np

# ═══════════════════════════════════════════════════
# CHANNEL PARAMETERS
# ═══════════════════════════════════════════════════

CHANNEL_PARAMS = {
    'tv': {
        'weekly_spend_mean': 40_000,
        'weekly_spend_std': 10_000,
        'decay': 0.75,      # Joseph: FMCG half-life 2–5 wks; Robyn TV: 0.3–0.8; Recast: λ=0.75 → ~2.4 wk half-life
        'hill_K': 0.50,     # Robyn gamma: 0.3–1.0; Prompt 4 TV range: 0.5–0.9
        'hill_S': 2.0,      # Robyn alpha: 0.5–3.0; Prompt 4 TV range: 1.0–2.5 (S-shaped allowed)
        'coeff': 0.15,      # Tune in Phase 4 regression
    },
    'search': {
        'weekly_spend_mean': 20_000,
        'weekly_spend_std': 5_000,
        'decay': 0.20,      # Robyn community: search 0.1–0.3; Improvado: 0.0–0.2; half-life ~2–3 days
        'hill_K': 0.30,     # Prompt 4: search 0.2–0.6; saturates early (finite keyword audience)
        'hill_S': 1.2,      # CHANGED from 1.5. Prompt 4: performance channels center S ≤1 (concave). 1.2 = near-concave.
        'coeff': 0.25,
    },
    'social': {
        'weekly_spend_mean': 15_000,
        'weekly_spend_std': 4_000,
        'decay': 0.45,      # PyMC tutorial: social between TV and search; video 0.4–0.6, static 0.3–0.5
        'hill_K': 0.40,     # Prompt 4: social 0.3–0.8; moderate saturation
        'hill_S': 1.8,      # Prompt 4: social 0.5–2.0; between TV (S-shaped) and search (concave)
        'coeff': 0.18,
    },
    'email': {
        'weekly_spend_mean': 3_000,
        'weekly_spend_std': 800,
        'decay': 0.20,      # CHANGED from 0.30. Prompt 3: promotional email 0.0–0.2; nurture 0.1–0.3. We model promo.
        'hill_K': 0.30,     # Robyn gamma floor = 0.3; Prompt 4 email: 0.1–0.5 (practitioners go lower)
        'hill_S': 1.0,      # CHANGED from 1.2. Prompt 4: email "close to linear at low spend"; performance = concave (S≤1)
        'coeff': 0.10,
    },
    'display': {
        'weekly_spend_mean': 10_000,
        'weekly_spend_std': 3_000,
        'decay': 0.30,      # Improvado display: 0.1–0.4; Recast: λ=0.3 → ~3–4 day half-life
        'hill_K': 0.35,     # Prompt 4: display 0.3–0.8; saturates relatively early (banner blindness)
        'hill_S': 1.5,      # Prompt 4: display 0.5–2.0; intermediate
        'coeff': 0.12,
    },
}

# ═══════════════════════════════════════════════════
# CONTROL VARIABLES
# ═══════════════════════════════════════════════════

CONTROL_PARAMS = {
    'base_revenue': 200_000,            # Weekly base revenue with zero marketing (design choice)
    'seasonality_amplitude': 0.20,      # CHANGED from 0.15. Prompt 6: Q4 is 25–40% above baseline.
    'noise_std': 0.10,                  # CHANGED from 0.05. Prompt 6: MMM residuals 5–15% of mean; 10% = conservative realistic.
}

# Holiday lifts — CHANGED from flat 0.25 to granular per-holiday.
# Source: NRF/Adobe via Yahoo Finance (BF/CM), CNBC (Christmas), Nationwide Group (July 4th).
HOLIDAY_EFFECTS = {
    'black_friday':   0.75,   # +75% vs baseline week (Prompt 6: BF/CM week +50–100%, we use +75%)
    'christmas':      0.60,   # +60% vs baseline week (Prompt 6: +40–80% for gift-heavy categories)
    'memorial_day':   0.20,   # +20% (Prompt 6: +15–25%)
    'july_4th':       0.20,   # +20% (Prompt 6: +15–30%)
    'labor_day':      0.15,   # +15% (interpolated from similar minor holidays)
    'easter':         0.10,   # +10% (smaller retail impact outside grocery/candy)
}

# Competitor promotion effect — NEW parameter.
# Source: Zigpoll MMM case (CPG brand lost ~8% during competitor promos).
COMPETITOR_PARAMS = {
    'effect': -0.08,          # −8% revenue impact when competitor runs a promo
    'probability': 0.15,      # ~15% of weeks have a competitor promo (design choice: ~8 out of 52 weeks)
}

N_WEEKS = 104  # 2 years of weekly data


