# plot_response_curves.py — Visualize adstock decay and Hill saturation for all 5 channels
# RUN THIS IN THONNY to verify your transformation functions produce realistic curves.
# 
# What you're checking:
#   1. Adstock plot: TV should decay slowly (tall, wide curve), search should drop fast (short, narrow)
#   2. Hill plot: email should flatten earliest (low K), TV should flatten latest (high K)
#   3. Combined plot: each channel's full pipeline from raw spend to effective spend
#
# If something looks wrong, the issue is in config.py parameters, not the functions.

import numpy as np
import matplotlib.pyplot as plt
from config import CHANNEL_PARAMS
from transforms import adstock_geometric, hill_saturation, normalize_spend, apply_pipeline


# ═══════════════════════════════════════════════════
# SETUP: Generate sample spend data for visualization
# ═══════════════════════════════════════════════════

np.random.seed(42)
n_weeks = 104

# Channel colors (consistent across all plots)
COLORS = {
    'tv': '#1f77b4',        # blue
    'search': '#ff7f0e',    # orange
    'social': '#2ca02c',    # green
    'email': '#d62728',     # red
    'display': '#9467bd',   # purple
}

LABELS = {
    'tv': 'TV',
    'search': 'Paid Search',
    'social': 'Social Media',
    'email': 'Email',
    'display': 'Display',
}


# ═══════════════════════════════════════════════════
# PLOT 1: Adstock Decay Comparison (impulse response)
# ═══════════════════════════════════════════════════
# This shows: if you spend $1 in week 0 and $0 after, how long does the effect last?

print("=" * 60)
print("PLOT 1: Adstock Impulse Response (spend $1 in week 0 only)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

impulse_weeks = 20  # show 20 weeks of decay
impulse = np.zeros(impulse_weeks)
impulse[0] = 1.0  # $1 spent in week 0, nothing after

for channel, params in CHANNEL_PARAMS.items():
    decay = params['decay']
    adstocked = adstock_geometric(impulse, decay)
    
    ax.plot(range(impulse_weeks), adstocked, 
            color=COLORS[channel], linewidth=2.5, marker='o', markersize=4,
            label=f"{LABELS[channel]} (λ={decay})")
    
    # Print half-life
    half_life = np.log(0.5) / np.log(decay) if decay > 0 else 0
    print(f"  {LABELS[channel]:15s}  decay={decay}  half-life={half_life:.1f} weeks ({half_life*7:.0f} days)")

ax.set_xlabel('Weeks After $1 Spend', fontsize=12)
ax.set_ylabel('Remaining Effect', fontsize=12)
ax.set_title('Adstock Decay: How Fast Does Each Channel\'s Effect Fade?', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, impulse_weeks - 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('plot1_adstock_decay.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - TV (blue) should decay the SLOWEST (tall curve, still visible at week 10+)")
print("  - Search (orange) and Email (red) should drop to near-zero by week 3-4")
print("  - Social (green) should be in between")
print()


# ═══════════════════════════════════════════════════
# PLOT 2: Hill Saturation Curves
# ═══════════════════════════════════════════════════
# This shows: as you increase spend from 0 to max, how does the response change?

print("=" * 60)
print("PLOT 2: Hill Saturation Curves (spend vs. response)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

x_range = np.linspace(0.01, 1.0, 200)  # normalized spend from 0 to 1

for channel, params in CHANNEL_PARAMS.items():
    K = params['hill_K']
    S = params['hill_S']
    y = hill_saturation(x_range, K, S)
    
    ax.plot(x_range, y, color=COLORS[channel], linewidth=2.5,
            label=f"{LABELS[channel]} (K={K}, S={S})")
    
    # Mark the half-saturation point
    ax.plot(K, 0.5, 'o', color=COLORS[channel], markersize=8, zorder=5)
    
    print(f"  {LABELS[channel]:15s}  K={K}  S={S}  → 50% response at {K*100:.0f}% of max spend")

# Add reference line at y=0.5
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% response level')

ax.set_xlabel('Normalized Spend (0 = min, 1 = max)', fontsize=12)
ax.set_ylabel('Response (0 = no effect, 1 = max effect)', fontsize=12)
ax.set_title('Hill Saturation: Where Do Diminishing Returns Kick In?', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_hill_saturation.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - Email (red) should flatten EARLIEST (lowest K = saturates fastest)")
print("  - TV (blue) should flatten LATEST and have a more S-shaped curve (highest K and S)")
print("  - The dots mark each channel's half-saturation point (where response = 0.5)")
print("  - Search/Email curves should be more concave (gradual bend), TV more S-shaped (steeper middle)")
print()


# ═══════════════════════════════════════════════════
# PLOT 3: Full Pipeline — Raw Spend vs Effective Spend (104 weeks)
# ═══════════════════════════════════════════════════
# This shows the complete transformation on realistic simulated spend data.

print("=" * 60)
print("PLOT 3: Full Pipeline — 104 Weeks of Simulated Data")
print("=" * 60)

fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

for idx, (channel, params) in enumerate(CHANNEL_PARAMS.items()):
    ax = axes[idx]
    
    # Generate realistic spend for this channel
    raw_spend = np.maximum(
        np.random.normal(params['weekly_spend_mean'], params['weekly_spend_std'], n_weeks),
        0  # no negative spend
    )
    
    # Run full pipeline
    result = apply_pipeline(raw_spend, params['decay'], params['hill_K'], params['hill_S'])
    
    # Plot raw spend (light) and saturated output (bold)
    weeks = range(n_weeks)
    
    # Normalize raw spend to 0-1 for visual comparison
    raw_norm = normalize_spend(raw_spend)
    
    ax.fill_between(weeks, raw_norm, alpha=0.2, color=COLORS[channel], label='Raw spend (normalized)')
    ax.plot(weeks, result['adstocked'] / result['adstocked'].max(), 
            color=COLORS[channel], alpha=0.5, linewidth=1, linestyle='--', label='After adstock')
    ax.plot(weeks, result['saturated'], 
            color=COLORS[channel], linewidth=2, label='After adstock + Hill (effective spend)')
    
    ax.set_ylabel('Effect', fontsize=10)
    ax.set_title(f"{LABELS[channel]}  —  decay={params['decay']}, K={params['hill_K']}, S={params['hill_S']}", 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    print(f"  {LABELS[channel]:15s}  mean raw=${params['weekly_spend_mean']:,}  "
          f"→ adstock smooths the peaks → Hill compresses the range")

axes[-1].set_xlabel('Week', fontsize=12)

plt.suptitle('Full Transformation Pipeline: Raw Spend → Adstock → Hill Saturation', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot3_full_pipeline.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - The dashed line (adstock) should be smoother than the shaded area (raw spend)")
print("  - TV's dashed line should be MUCH smoother (high decay = lots of smoothing)")
print("  - Search's dashed line should look almost identical to raw (low decay = minimal smoothing)")
print("  - The solid line (after Hill) should be compressed toward the top (diminishing returns)")
print()


# ═══════════════════════════════════════════════════
# PLOT 4: Response Curves — Spend ($) vs Revenue Impact
# ═══════════════════════════════════════════════════
# This is the chart that goes in your executive deck.
# Shows: for each channel, if I spend $X/week, what's the marginal effect?

print("=" * 60)
print("PLOT 4: Channel Response Curves (for your executive deck)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

for channel, params in CHANNEL_PARAMS.items():
    # Create a range of possible weekly spend from $0 to 2x the mean
    max_spend = params['weekly_spend_mean'] * 2.5
    spend_range = np.linspace(100, max_spend, 200)
    
    # Apply adstock to a constant-spend scenario (steady state)
    # At steady state, adstock = spend / (1 - decay)
    steady_state_adstock = spend_range / (1 - params['decay'])
    
    # Normalize and apply Hill
    normalized = normalize_spend(steady_state_adstock)
    response = hill_saturation(normalized, params['hill_K'], params['hill_S'])
    
    ax.plot(spend_range / 1000, response, color=COLORS[channel], linewidth=2.5,
            label=f"{LABELS[channel]}")
    
    # Mark current average spend
    current = params['weekly_spend_mean']
    current_adstock = current / (1 - params['decay'])
    current_norm = (current_adstock - steady_state_adstock.min()) / (steady_state_adstock.max() - steady_state_adstock.min())
    current_response = hill_saturation(np.array([max(current_norm, 1e-10)]), params['hill_K'], params['hill_S'])[0]
    
    ax.plot(current / 1000, current_response, 's', color=COLORS[channel], 
            markersize=10, zorder=5, markeredgecolor='black', markeredgewidth=1)

ax.set_xlabel('Weekly Spend ($K)', fontsize=12)
ax.set_ylabel('Response (proportion of max effect)', fontsize=12)
ax.set_title('Channel Response Curves — Square = Current Spend Level', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot4_response_curves.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - Each channel should show diminishing returns (curve flattens at higher spend)")
print("  - TV should have the widest range before flattening (highest K)")
print("  - Email should flatten earliest (lowest K, smallest spend range)")
print("  - The squares show where your 'current' average spend sits on each curve")
print("  - If a square is near the flat part, that channel is nearly saturated")
print()


# ═══════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════

print("=" * 60)
print("SUMMARY: Files saved")
print("=" * 60)
print("  plot1_adstock_decay.png      — Impulse response comparison")
print("  plot2_hill_saturation.png    — Saturation curves comparison")
print("  plot3_full_pipeline.png      — 104-week pipeline visualization")
print("  plot4_response_curves.png    — Executive-ready response curves")
print()
print("If all 4 plots look right, your transforms.py is working correctly.")
print("Next step: generate_data.py (Task 2 — create the 104-week synthetic dataset)")
