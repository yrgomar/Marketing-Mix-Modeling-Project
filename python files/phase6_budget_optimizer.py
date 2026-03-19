# phase6_budget_optimizer.py — Budget Optimization via Scipy
# Phase 6.1: Given total budget, find optimal allocation across 5 channels
#             to maximize predicted weekly revenue.
#
# Three scenarios:
#   1. Current allocation (from historical averages)
#   2. Model-optimal allocation (scipy.optimize)
#   3. What-if: Cut TV by 50%, redistribute to digital channels
#
# Key design decision:
#   The OLS model was trained on *saturated* features, which depend on the
#   full adstock → normalize → Hill pipeline. The optimizer must replay
#   this pipeline for each candidate allocation. Normalization anchors
#   (min/max of adstocked series) are fixed from the training data to
#   keep the model's coefficient scale valid.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.optimize import minimize
from config import CHANNEL_PARAMS, CONTROL_PARAMS, N_WEEKS
from transforms import adstock_geometric, hill_saturation


# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════

INPUT_FILE = "mmm_dataset_transformed.csv"

CHANNELS = ["tv", "search", "social", "email", "display"]

FEATURE_COLS = [
    "tv_saturated", "search_saturated", "social_saturated",
    "email_saturated", "display_saturated",
    "seasonality", "holiday_lift", "competitor_promo",
]

# Spend bounds per channel (weekly): no channel goes to zero or eats everything
# These are business-realistic constraints, not just math constraints.
# Interview note: "We set minimum floors because you can't completely shut off
# a channel overnight, and maximums because there's finite inventory."
CHANNEL_BOUNDS = {
    "tv":      (5_000,  80_000),
    "search":  (5_000,  50_000),
    "social":  (3_000,  40_000),
    "email":   (500,    10_000),
    "display": (2_000,  30_000),
}


# ═══════════════════════════════════════════════════
# STEP 1: Load model and compute normalization anchors
# ═══════════════════════════════════════════════════

def load_model_and_data():
    """Load transformed data, fit OLS, and extract normalization anchors."""
    df = pd.read_csv(INPUT_FILE)

    # Fit the same OLS model from Phase 4
    X = sm.add_constant(df[FEATURE_COLS], has_constant="add")
    y = df["revenue"]
    model = sm.OLS(y, X).fit()

    # Extract normalization anchors from training data
    # These are the min/max of each channel's adstocked series,
    # used in Phase 3 to normalize before Hill transformation.
    # We MUST reuse the same anchors so the model coefficients stay valid.
    norm_anchors = {}
    for ch in CHANNELS:
        adstocked_col = f"{ch}_adstocked"
        norm_anchors[ch] = {
            "min": df[adstocked_col].min(),
            "max": df[adstocked_col].max(),
        }

    # Mean values of control variables (held constant during optimization)
    control_means = {
        "seasonality": df["seasonality"].mean(),
        "holiday_lift": df["holiday_lift"].mean(),
        "competitor_promo": df["competitor_promo"].mean(),
    }

    # Current average weekly spend per channel
    current_spend = {}
    for ch in CHANNELS:
        current_spend[ch] = df[f"{ch}_spend"].mean()

    return model, df, norm_anchors, control_means, current_spend


# ═══════════════════════════════════════════════════
# STEP 2: Spend → Saturated feature (for optimizer)
# ═══════════════════════════════════════════════════

def spend_to_saturated(weekly_spend, channel, norm_anchors):
    """
    Convert a single weekly spend value to a saturated feature value.

    For the optimizer, we simulate a steady-state scenario:
    if you spend $X every week, what does the adstock converge to?

    Steady-state adstock = spend / (1 - decay)
    Then normalize using training anchors, then apply Hill.

    Interview note:
        "In steady state, geometric adstock converges to spend/(1-decay).
        This lets us evaluate long-run channel efficiency without simulating
        104 weeks for every candidate allocation."
    """
    params = CHANNEL_PARAMS[channel]
    decay = params["decay"]

    # Steady-state adstock level
    steady_adstock = weekly_spend / (1.0 - decay)

    # Normalize using training-data anchors
    anchor_min = norm_anchors[channel]["min"]
    anchor_max = norm_anchors[channel]["max"]

    if anchor_max == anchor_min:
        normalized = 0.0
    else:
        normalized = (steady_adstock - anchor_min) / (anchor_max - anchor_min)

    # Clip to [0, 1] — spend outside training range gets bounded
    normalized = np.clip(normalized, 0.0, 1.0)

    # Apply Hill saturation
    saturated = hill_saturation(
        np.array([normalized]),
        K=params["hill_K"],
        S=params["hill_S"]
    )[0]

    return saturated


# ═══════════════════════════════════════════════════
# STEP 3: Objective function for scipy
# ═══════════════════════════════════════════════════

def predict_revenue(spend_vector, model, norm_anchors, control_means):
    """
    Given a 5-element spend vector [tv, search, social, email, display],
    predict weekly revenue using the trained OLS model.

    Returns NEGATIVE revenue (because scipy minimizes).
    """
    saturated_values = {}
    for i, ch in enumerate(CHANNELS):
        saturated_values[f"{ch}_saturated"] = spend_to_saturated(
            spend_vector[i], ch, norm_anchors
        )

    # Build feature vector in the same order as FEATURE_COLS
    features = {
        "const": 1.0,
        "tv_saturated": saturated_values["tv_saturated"],
        "search_saturated": saturated_values["search_saturated"],
        "social_saturated": saturated_values["social_saturated"],
        "email_saturated": saturated_values["email_saturated"],
        "display_saturated": saturated_values["display_saturated"],
        "seasonality": control_means["seasonality"],
        "holiday_lift": control_means["holiday_lift"],
        "competitor_promo": control_means["competitor_promo"],
    }

    feature_array = np.array([features[col] for col in ["const"] + FEATURE_COLS])
    predicted = model.predict(feature_array.reshape(1, -1))[0]
    return predicted


def neg_revenue(spend_vector, model, norm_anchors, control_means):
    """Negative revenue for scipy.optimize.minimize."""
    return -predict_revenue(spend_vector, model, norm_anchors, control_means)


# ═══════════════════════════════════════════════════
# STEP 4: Run optimizer
# ═══════════════════════════════════════════════════

def optimize_budget(total_budget, model, norm_anchors, control_means):
    """
    Find the spend allocation across 5 channels that maximizes
    predicted weekly revenue, subject to:
        - Total spend = total_budget
        - Each channel within its bounds

    Uses multiple random starting points to avoid local minima.
    """
    bounds = [CHANNEL_BOUNDS[ch] for ch in CHANNELS]

    # Constraint: all channels sum to total budget
    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

    best_result = None
    best_revenue = -np.inf

    # Try 20 random starting points
    np.random.seed(42)
    for trial in range(20):
        # Random starting point within bounds that sums to budget
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        x0 = x0 / x0.sum() * total_budget  # Scale to meet budget constraint

        # Clip to bounds after scaling
        for i, (lo, hi) in enumerate(bounds):
            x0[i] = np.clip(x0[i], lo, hi)

        result = minimize(
            neg_revenue,
            x0,
            args=(model, norm_anchors, control_means),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success and -result.fun > best_revenue:
            best_revenue = -result.fun
            best_result = result

    return best_result


# ═══════════════════════════════════════════════════
# STEP 5: Build scenario comparison
# ═══════════════════════════════════════════════════

def build_scenarios(model, norm_anchors, control_means, current_spend):
    """Build 3 scenarios: current, optimal, what-if."""

    total_budget = sum(current_spend.values())

    print("\n" + "=" * 72)
    print("PHASE 6 — BUDGET OPTIMIZATION")
    print("=" * 72)
    print(f"\nTotal weekly budget: ${total_budget:,.0f}")

    # --- Scenario 1: Current allocation ---
    current_vector = np.array([current_spend[ch] for ch in CHANNELS])
    current_revenue = predict_revenue(current_vector, model, norm_anchors, control_means)

    print("\n--- Scenario 1: Current Allocation ---")
    for i, ch in enumerate(CHANNELS):
        pct = current_vector[i] / total_budget * 100
        print(f"  {ch:8s}: ${current_vector[i]:>10,.0f}  ({pct:5.1f}%)")
    print(f"  Predicted weekly revenue: ${current_revenue:,.0f}")

    # --- Scenario 2: Model-optimal ---
    print("\n--- Scenario 2: Model-Optimal Allocation ---")
    print("  Running scipy optimizer with 20 random starts...")

    result = optimize_budget(total_budget, model, norm_anchors, control_means)

    if result is None or not result.success:
        print("  WARNING: Optimizer did not converge. Using current allocation.")
        optimal_vector = current_vector.copy()
        optimal_revenue = current_revenue
    else:
        optimal_vector = result.x
        optimal_revenue = -result.fun

    for i, ch in enumerate(CHANNELS):
        pct = optimal_vector[i] / total_budget * 100
        delta = optimal_vector[i] - current_vector[i]
        arrow = "+" if delta >= 0 else ""
        print(f"  {ch:8s}: ${optimal_vector[i]:>10,.0f}  ({pct:5.1f}%)  [{arrow}${delta:,.0f}]")
    print(f"  Predicted weekly revenue: ${optimal_revenue:,.0f}")

    lift = (optimal_revenue - current_revenue) / current_revenue * 100
    print(f"  Revenue lift vs current:  {lift:+.1f}%  (${optimal_revenue - current_revenue:+,.0f}/week)")

    # --- Scenario 3: What-if (cut TV 50%, redistribute to digital) ---
    print("\n--- Scenario 3: What-If (Cut TV 50% → Redistribute to Digital) ---")

    whatif_vector = current_vector.copy()
    tv_cut = whatif_vector[0] * 0.50       # Cut TV by 50%
    whatif_vector[0] -= tv_cut             # Reduce TV

    # Redistribute the freed budget proportionally to search, social, display
    digital_channels = [1, 2, 4]  # search, social, display indices
    digital_total = sum(whatif_vector[i] for i in digital_channels)
    for i in digital_channels:
        whatif_vector[i] += tv_cut * (whatif_vector[i] / digital_total)

    whatif_revenue = predict_revenue(whatif_vector, model, norm_anchors, control_means)

    for i, ch in enumerate(CHANNELS):
        pct = whatif_vector[i] / total_budget * 100
        delta = whatif_vector[i] - current_vector[i]
        arrow = "+" if delta >= 0 else ""
        print(f"  {ch:8s}: ${whatif_vector[i]:>10,.0f}  ({pct:5.1f}%)  [{arrow}${delta:,.0f}]")
    print(f"  Predicted weekly revenue: ${whatif_revenue:,.0f}")

    lift_wf = (whatif_revenue - current_revenue) / current_revenue * 100
    print(f"  Revenue lift vs current:  {lift_wf:+.1f}%  (${whatif_revenue - current_revenue:+,.0f}/week)")

    # --- Summary table ---
    scenarios = pd.DataFrame({
        "Channel": CHANNELS,
        "Current ($)": current_vector,
        "Optimal ($)": optimal_vector,
        "What-If ($)": whatif_vector,
    })

    summary = pd.DataFrame({
        "Scenario": ["Current", "Optimal", "What-If"],
        "Predicted Revenue": [current_revenue, optimal_revenue, whatif_revenue],
        "vs Current ($)": [0, optimal_revenue - current_revenue, whatif_revenue - current_revenue],
        "vs Current (%)": [0, lift, lift_wf],
    })

    print("\n" + "=" * 72)
    print("SCENARIO COMPARISON TABLE")
    print("=" * 72)
    print(scenarios.round(0).to_string(index=False))

    print("\n")
    print(summary.round(2).to_string(index=False))

    return scenarios, summary, current_revenue, optimal_revenue, whatif_revenue


# ═══════════════════════════════════════════════════
# STEP 6: Visualization — side-by-side bar chart
# ═══════════════════════════════════════════════════

def plot_scenario_comparison(scenarios, current_rev, optimal_rev, whatif_rev):
    """Side-by-side grouped bar chart: spend by channel per scenario + revenue."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # --- Left panel: spend allocation by channel ---
    x = np.arange(len(CHANNELS))
    width = 0.25

    bars1 = ax1.bar(x - width, scenarios["Current ($)"], width,
                     label="Current", color="#1f77b4", edgecolor="black")
    bars2 = ax1.bar(x, scenarios["Optimal ($)"], width,
                     label="Optimal", color="#2ca02c", edgecolor="black")
    bars3 = ax1.bar(x + width, scenarios["What-If ($)"], width,
                     label="What-If", color="#ff7f0e", edgecolor="black")

    ax1.set_xlabel("Channel", fontsize=12)
    ax1.set_ylabel("Weekly Spend ($)", fontsize=12)
    ax1.set_title("Budget Allocation by Scenario", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ch.upper() for ch in CHANNELS])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add dollar labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 3000:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 500,
                        f"${h/1000:.0f}K", ha="center", va="bottom", fontsize=7)

    # --- Right panel: predicted revenue comparison ---
    scenarios_list = ["Current", "Optimal", "What-If"]
    revenues = [current_rev, optimal_rev, whatif_rev]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    rev_bars = ax2.bar(scenarios_list, revenues, color=colors, edgecolor="black", width=0.6)

    for bar, rev in zip(rev_bars, revenues):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                f"${rev:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Predicted Weekly Revenue ($)", fontsize=12)
    ax2.set_title("Revenue by Scenario", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Set y-axis to start from a reasonable base for readability
    min_rev = min(revenues)
    ax2.set_ylim(min_rev * 0.9, max(revenues) * 1.05)

    plt.tight_layout()
    plt.savefig("phase6_scenario_comparison.png", dpi=150)
    plt.show()

    print("\nSaved: phase6_scenario_comparison.png")


# ═══════════════════════════════════════════════════
# STEP 7: Executive narrative
# ═══════════════════════════════════════════════════

def print_executive_narrative(scenarios, current_rev, optimal_rev, whatif_rev):
    """Print interview-ready narrative of the optimization results."""

    total_budget = scenarios["Current ($)"].sum()
    annual_lift = (optimal_rev - current_rev) * 52

    print("\n" + "=" * 72)
    print("EXECUTIVE NARRATIVE")
    print("=" * 72)

    print(f"""
BUDGET OPTIMIZATION SUMMARY
----------------------------
Total weekly marketing budget: ${total_budget:,.0f} (${total_budget * 52:,.0f} annually)

The model-optimal allocation projects ${optimal_rev:,.0f}/week in revenue,
compared to ${current_rev:,.0f}/week under the current allocation.

Annualized, this represents a potential revenue lift of ${annual_lift:,.0f}.

Key reallocation moves (optimal vs current):""")

    for i, ch in enumerate(CHANNELS):
        delta = scenarios["Optimal ($)"].iloc[i] - scenarios["Current ($)"].iloc[i]
        if abs(delta) > 500:
            direction = "Increase" if delta > 0 else "Decrease"
            print(f"  • {direction} {ch.upper()} by ${abs(delta):,.0f}/week")

    print(f"""
The what-if scenario (cutting TV by 50% and redistributing to digital)
produces ${whatif_rev:,.0f}/week — {'outperforming' if whatif_rev > current_rev else 'underperforming'}
the current allocation by ${abs(whatif_rev - current_rev):,.0f}/week.

CAVEATS:
  • These projections assume the model's saturation curves hold at new
    spend levels. In practice, test with incremental budget shifts of
    10-15% per quarter, not a full reallocation overnight.
  • Email's high efficiency is partly driven by a small spend base;
    scaling email aggressively would likely hit diminishing returns faster
    than other channels.
  • Competitor behavior and market conditions are held constant.
""")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    model, df, norm_anchors, control_means, current_spend = load_model_and_data()

    scenarios, summary, c_rev, o_rev, w_rev = build_scenarios(
        model, norm_anchors, control_means, current_spend
    )

    plot_scenario_comparison(scenarios, c_rev, o_rev, w_rev)
    print_executive_narrative(scenarios, c_rev, o_rev, w_rev)

    print("\n" + "=" * 72)
    print("FILES SAVED")
    print("=" * 72)
    print("  phase6_scenario_comparison.png")
    print("\nNext step: Phase 7 — Executive deliverables (deck, summary, README).")


if __name__ == "__main__":
    main()
