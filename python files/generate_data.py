# generate_data.py — Synthetic Marketing Dataset Generator
# Phase 2, Task 2 of the MMM project.
#
# What this does:
#   Creates a 104-week (2-year) dataset with 5 channels of marketing spend,
#   control variables (seasonality, holidays, competitor promos), and revenue.
#   Revenue is constructed from KNOWN relationships so your regression model
#   can be validated against ground truth.
#
# How revenue is built:
#   revenue = base_revenue × (1 + seasonality) × (1 + holiday_lift) × (1 + competitor_effect)
#             + Σ(channel_coeff × saturated_spend × base_revenue)
#             + noise
#
#   Multiplicative: seasonality, holidays, competitor promos scale the base.
#   Additive: each channel's transformed spend adds incremental revenue on top.
#
# Run: python generate_data.py
# Output: mmm_dataset.csv (104 rows × 14 columns)

import numpy as np
import pandas as pd
from config import CHANNEL_PARAMS, CONTROL_PARAMS, HOLIDAY_EFFECTS, COMPETITOR_PARAMS, N_WEEKS
from transforms import apply_pipeline

# ═══════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════

np.random.seed(42)  # Reproducibility — same dataset every time you run this

CHANNELS = list(CHANNEL_PARAMS.keys())  # ['tv', 'search', 'social', 'email', 'display']


# ═══════════════════════════════════════════════════
# STEP 1: Generate Weekly Spend Per Channel
# ═══════════════════════════════════════════════════
# Each channel's spend is drawn from a normal distribution.
# np.maximum(..., 0) prevents negative spend (can't spend negative dollars).

print("Step 1: Generating weekly spend for 5 channels...")

spend_data = {}
for ch in CHANNELS:
    mean = CHANNEL_PARAMS[ch]['weekly_spend_mean']
    std = CHANNEL_PARAMS[ch]['weekly_spend_std']
    spend = np.maximum(np.random.normal(mean, std, N_WEEKS), 0)
    spend_data[ch] = spend
    print(f"  {ch:10s}  mean=${mean:>7,}  std=${std:>6,}  "
          f"actual range: ${spend.min():,.0f} – ${spend.max():,.0f}")


# ═══════════════════════════════════════════════════
# STEP 2: Transform Spend Through Pipeline
# ═══════════════════════════════════════════════════
# Each channel's raw spend → adstock → normalize → Hill saturation.
# The 'saturated' output is what drives revenue in the model.

print("\nStep 2: Applying adstock + Hill pipeline to each channel...")

saturated_data = {}
for ch in CHANNELS:
    params = CHANNEL_PARAMS[ch]
    result = apply_pipeline(spend_data[ch], params['decay'], params['hill_K'], params['hill_S'])
    saturated_data[ch] = result['saturated']
    print(f"  {ch:10s}  saturated range: {result['saturated'].min():.3f} – {result['saturated'].max():.3f}")


# ═══════════════════════════════════════════════════
# STEP 3: Build Control Variables
# ═══════════════════════════════════════════════════

print("\nStep 3: Building control variables...")

# --- 3a: Seasonality ---
# Sine wave that peaks at week 48 (late November / Q4) and troughs at week 22 (late May / Q2).
# Formula: amplitude × sin(2π × (week - phase_shift) / 52)
# Phase shift of 37 puts the peak at ~week 48 (2π × (48-37)/52 ≈ π/2 → sin = 1)

amplitude = CONTROL_PARAMS['seasonality_amplitude']
weeks = np.arange(N_WEEKS)
seasonality = amplitude * np.sin(2 * np.pi * (weeks - 37) / 52)

print(f"  Seasonality: amplitude={amplitude}, peak at ~week 48 (Q4), trough at ~week 22 (Q2)")
print(f"    Range: {seasonality.min():.3f} to {seasonality.max():.3f}")

# --- 3b: Holiday Flags ---
# Map holidays to specific weeks within each of the 2 years.
# Week 0 = first week of January Year 1.

holiday_lift = np.zeros(N_WEEKS)

# Holiday week mappings (approximate — week number within a year)
HOLIDAY_WEEKS = {
    'easter':       [14, 66],       # ~mid April (week 14 year 1, week 66 year 2)
    'memorial_day': [21, 73],       # ~late May
    'july_4th':     [26, 78],       # ~early July
    'labor_day':    [35, 87],       # ~early September
    'black_friday': [47, 99],       # ~late November
    'christmas':    [51, 103],      # ~late December
}

for holiday, effect in HOLIDAY_EFFECTS.items():
    week_indices = HOLIDAY_WEEKS[holiday]
    for w in week_indices:
        if w < N_WEEKS:
            holiday_lift[w] = effect
            
    print(f"  {holiday:15s}  lift=+{effect*100:.0f}%  weeks={week_indices}")

# Create a binary flag column (1 = any holiday that week, 0 = none)
holiday_flag = (holiday_lift > 0).astype(int)

print(f"  Total holiday weeks: {holiday_flag.sum()} out of {N_WEEKS}")

# --- 3c: Competitor Promotions ---
# Randomly assign ~15% of weeks as competitor promo weeks.

comp_prob = COMPETITOR_PARAMS['probability']
comp_effect_size = COMPETITOR_PARAMS['effect']
competitor_promo = np.random.binomial(1, comp_prob, N_WEEKS)

print(f"  Competitor promos: {competitor_promo.sum()} weeks "
      f"(target ~{comp_prob*100:.0f}% = ~{comp_prob*N_WEEKS:.0f} weeks)")


# ═══════════════════════════════════════════════════
# STEP 4: Calculate Revenue
# ═══════════════════════════════════════════════════
# Revenue = base × multipliers + channel contributions + noise

print("\nStep 4: Calculating revenue...")

base = CONTROL_PARAMS['base_revenue']

# 4a: Multiplicative effects (scale the base)
multiplier = (1 + seasonality) * (1 + holiday_lift) * (1 + competitor_promo * comp_effect_size)

base_with_controls = base * multiplier

# 4b: Additive channel contributions
channel_contribution = np.zeros(N_WEEKS)
for ch in CHANNELS:
    coeff = CHANNEL_PARAMS[ch]['coeff']
    contribution = coeff * saturated_data[ch] * base
    channel_contribution += contribution
    avg = contribution.mean()
    print(f"  {ch:10s}  coeff={coeff}  avg weekly contribution: ${avg:,.0f}")

# 4c: Combine
revenue_clean = base_with_controls + channel_contribution

# 4d: Add noise
noise_std = CONTROL_PARAMS['noise_std']
noise = np.random.normal(0, noise_std * base, N_WEEKS)
revenue = revenue_clean + noise

# Ensure no negative revenue
revenue = np.maximum(revenue, 0)

print(f"\n  Base revenue:        ${base:>10,}/week")
print(f"  Avg total revenue:   ${revenue.mean():>10,.0f}/week")
print(f"  Revenue range:       ${revenue.min():>10,.0f} – ${revenue.max():>10,.0f}")
print(f"  Noise std:           {noise_std*100:.0f}% of base (${noise_std * base:,.0f})")


# ═══════════════════════════════════════════════════
# STEP 5: Assemble and Save DataFrame
# ═══════════════════════════════════════════════════

print("\nStep 5: Saving dataset...")

df = pd.DataFrame({
    'week': np.arange(1, N_WEEKS + 1),
    'tv_spend': spend_data['tv'],
    'search_spend': spend_data['search'],
    'social_spend': spend_data['social'],
    'email_spend': spend_data['email'],
    'display_spend': spend_data['display'],
    'seasonality': seasonality,
    'holiday_flag': holiday_flag,
    'holiday_lift': holiday_lift,
    'competitor_promo': competitor_promo,
    'revenue': revenue,
})

# Round spend and revenue to 2 decimal places for cleanliness
for col in ['tv_spend', 'search_spend', 'social_spend', 'email_spend', 'display_spend', 'revenue']:
    df[col] = df[col].round(2)

df['seasonality'] = df['seasonality'].round(4)
df['holiday_lift'] = df['holiday_lift'].round(2)

# Save
df.to_csv('mmm_dataset.csv', index=False)

print("\n  Saved: mmm_dataset.csv")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n  Columns: {list(df.columns)}")


# ═══════════════════════════════════════════════════
# STEP 6: Quick Sanity Checks
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Check 1: Holiday weeks should have higher revenue
holiday_rev = revenue[holiday_flag == 1].mean()
normal_rev = revenue[holiday_flag == 0].mean()
print(f"\n  Avg revenue (holiday weeks):  ${holiday_rev:,.0f}")
print(f"  Avg revenue (normal weeks):   ${normal_rev:,.0f}")
print(f"  Holiday premium:              +{(holiday_rev/normal_rev - 1)*100:.1f}%")

# Check 2: Q4 revenue should be higher than Q2
q4_mask = (weeks % 52 >= 39) & (weeks % 52 <= 51)  # Oct-Dec
q2_mask = (weeks % 52 >= 13) & (weeks % 52 <= 25)  # Apr-Jun
q4_rev = revenue[q4_mask].mean()
q2_rev = revenue[q2_mask].mean()
print(f"\n  Avg revenue (Q4 weeks):       ${q4_rev:,.0f}")
print(f"  Avg revenue (Q2 weeks):       ${q2_rev:,.0f}")
print(f"  Q4 vs Q2 premium:             +{(q4_rev/q2_rev - 1)*100:.1f}%")

# Check 3: Competitor promo weeks should have lower revenue
comp_rev = revenue[competitor_promo == 1].mean()
no_comp_rev = revenue[competitor_promo == 0].mean()
print(f"\n  Avg revenue (competitor promo): ${comp_rev:,.0f}")
print(f"  Avg revenue (no competitor):    ${no_comp_rev:,.0f}")
print(f"  Competitor impact:              {(comp_rev/no_comp_rev - 1)*100:.1f}%")

# Check 4: Revenue should be reasonable
print("\n  Revenue stats:")
print(f"    Mean:   ${revenue.mean():,.0f}")
print(f"    Median: ${np.median(revenue):,.0f}")
print(f"    Std:    ${revenue.std():,.0f}")
print(f"    CV:     {revenue.std()/revenue.mean()*100:.1f}%")

print("\n" + "=" * 60)
print("DONE — Dataset ready for Phase 3 (EDA) and Phase 4 (regression)")
print("=" * 60)
print("\nNext step: Run EDA on mmm_dataset.csv to check distributions,")
print("correlations, and multicollinearity before building the model.")
