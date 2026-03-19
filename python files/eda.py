# eda.py — Exploratory Data Analysis on MMM Dataset
# Phase 2, Task 3 of the MMM project.
#
# BUSINESS CONTEXT:
#   Nova — a mid-size DTC consumer electronics brand (wireless headphones,
#   speakers, smart home devices). ~$15M annual revenue via website + Amazon.
#   Marketing budget ~$4.6M/year across 5 channels.
#
# What this does:
#   Runs 5 diagnostic checks on mmm_dataset.csv before you build the regression.
#   If something looks wrong here, fix it BEFORE modeling — garbage in, garbage out.
#
# Run: python eda.py
# Output: 5 plots saved as PNGs + printed diagnostics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════

df = pd.read_csv('mmm_dataset.csv')

print("=" * 60)
print("MMM EXPLORATORY DATA ANALYSIS — Nova DTC Electronics")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Define channel columns for reuse
SPEND_COLS = ['tv_spend', 'search_spend', 'social_spend', 'email_spend', 'display_spend']

LABELS = {
    'tv_spend': 'TV',
    'search_spend': 'Paid Search',
    'social_spend': 'Social Media',
    'email_spend': 'Email',
    'display_spend': 'Display',
}

COLORS = {
    'tv_spend': '#1f77b4',
    'search_spend': '#ff7f0e',
    'social_spend': '#2ca02c',
    'email_spend': '#d62728',
    'display_spend': '#9467bd',
}


# ═══════════════════════════════════════════════════
# CHECK 1: Summary Statistics
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 1: Summary Statistics")
print("=" * 60)

print("\nSpend columns:")
print(df[SPEND_COLS].describe().round(0).to_string())

print(f"\nRevenue:")
print(f"  Mean:   ${df['revenue'].mean():,.0f}")
print(f"  Median: ${df['revenue'].median():,.0f}")
print(f"  Std:    ${df['revenue'].std():,.0f}")
print(f"  Min:    ${df['revenue'].min():,.0f}")
print(f"  Max:    ${df['revenue'].max():,.0f}")
print(f"  Skew:   {df['revenue'].skew():.2f}")

# Flag any issues
skew = df['revenue'].skew()
if abs(skew) > 1:
    print("  ⚠️  Revenue is heavily skewed — consider log transform")
else:
    print("  ✓  Revenue skew is acceptable (between -1 and 1)")


# ═══════════════════════════════════════════════════
# CHECK 2: Spend Distributions (Histograms)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 2: Spend Distributions")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, col in enumerate(SPEND_COLS):
    ax = axes[idx]
    ax.hist(df[col], bins=20, color=COLORS[col], edgecolor='white', alpha=0.8)
    ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean: ${df[col].mean():,.0f}')
    ax.set_title(LABELS[col], fontsize=12, fontweight='bold')
    ax.set_xlabel('Weekly Spend ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    print(f"  {LABELS[col]:15s}  mean=${df[col].mean():>8,.0f}  std=${df[col].std():>7,.0f}  "
          f"min=${df[col].min():>7,.0f}  max=${df[col].max():>7,.0f}")

# Revenue histogram in the 6th subplot
ax = axes[5]
ax.hist(df['revenue'], bins=20, color='#333333', edgecolor='white', alpha=0.8)
ax.axvline(df['revenue'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: ${df["revenue"].mean():,.0f}')
ax.set_title('Revenue', fontsize=12, fontweight='bold')
ax.set_xlabel('Weekly Revenue ($)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=8)

plt.suptitle('Distribution of Weekly Spend by Channel + Revenue', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plot1_distributions.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - Each channel should look roughly bell-shaped (normal)")
print("  - No negative values (we clipped at 0)")
print("  - Revenue should be right-skewed due to holiday spikes")


# ═══════════════════════════════════════════════════
# CHECK 3: Correlation Matrix
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 3: Correlation Matrix")
print("=" * 60)

# Correlations between spend columns and revenue
corr_cols = SPEND_COLS + ['seasonality', 'holiday_lift', 'competitor_promo', 'revenue']
corr_matrix = df[corr_cols].corr()

# Print spend-revenue correlations
print("\nSpend vs. Revenue correlations:")
for col in SPEND_COLS:
    r = corr_matrix.loc[col, 'revenue']
    print(f"  {LABELS[col]:15s}  r = {r:+.3f}")

print(f"\n  Seasonality     r = {corr_matrix.loc['seasonality', 'revenue']:+.3f}")
print(f"  Holiday lift    r = {corr_matrix.loc['holiday_lift', 'revenue']:+.3f}")
print(f"  Competitor      r = {corr_matrix.loc['competitor_promo', 'revenue']:+.3f}")

# Print inter-channel correlations (multicollinearity check)
print("\nInter-channel correlations (watch for |r| > 0.7):")
for i in range(len(SPEND_COLS)):
    for j in range(i + 1, len(SPEND_COLS)):
        col_i, col_j = SPEND_COLS[i], SPEND_COLS[j]
        r = corr_matrix.loc[col_i, col_j]
        flag = "  ⚠️ HIGH" if abs(r) > 0.7 else ""
        print(f"  {LABELS[col_i]:15s} vs {LABELS[col_j]:15s}  r = {r:+.3f}{flag}")

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=[LABELS.get(c, c.replace('_', ' ').title()) for c in corr_cols],
            yticklabels=[LABELS.get(c, c.replace('_', ' ').title()) for c in corr_cols])
ax.set_title('Correlation Matrix — Spend, Controls, and Revenue', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plot2_correlations.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - Spend-revenue correlations should be POSITIVE (spend drives revenue)")
print("  - Seasonality and holiday_lift should correlate positively with revenue")
print("  - Inter-channel correlations should be LOW (< 0.7)")
print("  - If two channels are highly correlated, regression can't separate their effects")


# ═══════════════════════════════════════════════════
# CHECK 4: Variance Inflation Factor (VIF)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 4: Variance Inflation Factor (VIF)")
print("=" * 60)
print("  VIF measures how much each predictor is explained by the others.")
print("  VIF > 5 = concerning  |  VIF > 10 = serious multicollinearity\n")

# Build the feature matrix (what will go into your regression)
vif_cols = SPEND_COLS + ['seasonality', 'holiday_lift', 'competitor_promo']
X_vif = df[vif_cols].copy()

# Add constant (intercept) — required for VIF calculation
X_vif.insert(0, 'const', 1)

vif_results = []
for i in range(1, len(X_vif.columns)):  # skip the constant
    col_name = X_vif.columns[i]
    vif_val = variance_inflation_factor(X_vif.values, i)
    label = LABELS.get(col_name, col_name.replace('_', ' ').title())
    
    if vif_val > 10:
        status = "🔴 SERIOUS"
    elif vif_val > 5:
        status = "🟡 CONCERNING"
    else:
        status = "🟢 OK"
    
    vif_results.append({'Variable': label, 'VIF': vif_val, 'Status': status})
    print(f"  {label:20s}  VIF = {vif_val:>6.2f}  {status}")

print("\nInterpretation:")
max_vif = max(r['VIF'] for r in vif_results)
if max_vif > 10:
    print("  ⚠️  At least one variable has VIF > 10. Consider ridge regression or removing it.")
elif max_vif > 5:
    print("  ⚠️  At least one variable has VIF > 5. Monitor in regression, may need correction.")
else:
    print("  ✓  All VIF scores below 5. No multicollinearity issues — safe for OLS regression.")


# ═══════════════════════════════════════════════════
# CHECK 5: Revenue Over Time (Time Series Plot)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 5: Revenue Over Time")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot 5a: Revenue with holiday and competitor markers
ax = axes[0]
ax.plot(df['week'], df['revenue'], color='#333333', linewidth=1.5, label='Weekly Revenue')
ax.fill_between(df['week'], df['revenue'], alpha=0.1, color='#333333')

# Mark holidays
holiday_weeks = df[df['holiday_flag'] == 1]
ax.scatter(holiday_weeks['week'], holiday_weeks['revenue'], color='red', s=80,
           zorder=5, label='Holiday Weeks', marker='^')

# Mark competitor promos
comp_weeks = df[df['competitor_promo'] == 1]
ax.scatter(comp_weeks['week'], comp_weeks['revenue'], color='orange', s=60,
           zorder=5, label='Competitor Promo', marker='v')

ax.axhline(df['revenue'].mean(), color='gray', linestyle='--', alpha=0.5, label=f'Mean: ${df["revenue"].mean():,.0f}')
ax.set_ylabel('Revenue ($)', fontsize=11)
ax.set_title('Weekly Revenue — Nova DTC Electronics (104 Weeks)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 5b: Seasonality wave
ax = axes[1]
ax.plot(df['week'], df['seasonality'], color='#e377c2', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Seasonality Index', fontsize=11)
ax.set_title('Seasonality Pattern (peaks Q4, troughs Q2)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 5c: All channel spend stacked
ax = axes[2]
for col in SPEND_COLS:
    ax.plot(df['week'], df[col] / 1000, color=COLORS[col], linewidth=1.2,
            alpha=0.8, label=LABELS[col])
ax.set_ylabel('Weekly Spend ($K)', fontsize=11)
ax.set_xlabel('Week', fontsize=11)
ax.set_title('Marketing Spend by Channel Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plot3_time_series.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - Revenue should show a wave pattern (seasonality) peaking around weeks 48 and 100")
print("  - Red triangles (holidays) should sit ABOVE the trend line")
print("  - Orange triangles (competitor promos) should sit slightly below nearby weeks")
print("  - Channel spend should look random (no trend) — we generated it that way")
print("  - The seasonality wave in the middle plot should repeat every 52 weeks")


# ═══════════════════════════════════════════════════
# CHECK 6: Spend vs Revenue Scatter Plots
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CHECK 6: Spend vs Revenue (Scatter Plots)")
print("=" * 60)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for idx, col in enumerate(SPEND_COLS):
    ax = axes[idx]
    ax.scatter(df[col] / 1000, df['revenue'] / 1000, color=COLORS[col],
               alpha=0.5, s=30, edgecolors='white', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df[col], df['revenue'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line / 1000, p(x_line) / 1000, color='black', linewidth=1.5, linestyle='--')
    
    r = df[col].corr(df['revenue'])
    ax.set_title(f"{LABELS[col]}\nr = {r:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel('Spend ($K)', fontsize=9)
    if idx == 0:
        ax.set_ylabel('Revenue ($K)', fontsize=9)

plt.suptitle('Channel Spend vs Revenue (with trend lines)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plot4_scatter.png', dpi=150)
plt.show()

print("\nWhat to check:")
print("  - All trend lines should slope UPWARD (more spend → more revenue)")
print("  - Scatter should be loose (noisy) not tight — we added 10% noise")
print("  - If any trend line is flat or negative, that channel may be problematic in regression")


# ═══════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print("\n  Files saved:")
print("    eda_plot1_distributions.png   — Spend + revenue histograms")
print("    eda_plot2_correlations.png    — Correlation heatmap")
print("    eda_plot3_time_series.png     — Revenue, seasonality, and spend over time")
print("    eda_plot4_scatter.png         — Spend vs revenue scatter plots")
print(f"\n  Max VIF: {max_vif:.2f} — {'OK for OLS' if max_vif < 5 else 'May need attention'}")
print(f"  Revenue skew: {skew:.2f} — {'Acceptable' if abs(skew) < 1 else 'Consider transform'}")
print("\n  If all checks pass: proceed to Phase 4 (OLS regression)")
print("  If VIF > 10 for any variable: consider ridge regression or dropping that variable")
