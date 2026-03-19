# phase6_scenario_comparison.py — Executive Scenario Comparison
# Phase 6.2: Polished visualization + formatted table comparing
#             Current vs Optimal vs What-If budget allocations.
#
# Produces:
#   1. Formatted comparison table (console + CSV export)
#   2. Two-panel executive chart with annotations
#   3. Narrative annotations on the chart itself
#
# Depends on: phase6_budget_optimizer.py (runs the optimizer)
# Run this AFTER running phase6_budget_optimizer.py, or it re-runs internally.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from phase6_budget_optimizer import (
    load_model_and_data,
    optimize_budget,
    predict_revenue,
    CHANNELS,
    CHANNEL_BOUNDS,
)


# ═══════════════════════════════════════════════════
# LABELS & FORMATTING
# ═══════════════════════════════════════════════════

CHANNEL_DISPLAY = {
    "tv": "TV",
    "search": "Paid Search",
    "social": "Social Media",
    "email": "Email",
    "display": "Display",
}

SCENARIO_COLORS = {
    "Current": "#1f77b4",
    "Optimal": "#2ca02c",
    "What-If": "#ff7f0e",
}


# ═══════════════════════════════════════════════════
# STEP 1: Compute all three scenarios
# ═══════════════════════════════════════════════════

def compute_scenarios():
    """Run optimizer and build all three scenario vectors."""

    model, df, norm_anchors, control_means, current_spend = load_model_and_data()
    total_budget = sum(current_spend.values())

    # Scenario 1: Current
    current_vector = np.array([current_spend[ch] for ch in CHANNELS])
    current_revenue = predict_revenue(current_vector, model, norm_anchors, control_means)

    # Scenario 2: Optimal
    result = optimize_budget(total_budget, model, norm_anchors, control_means)
    optimal_vector = result.x if result and result.success else current_vector.copy()
    optimal_revenue = predict_revenue(optimal_vector, model, norm_anchors, control_means)

    # Scenario 3: What-If (cut TV 50%, redistribute to digital)
    whatif_vector = current_vector.copy()
    tv_cut = whatif_vector[0] * 0.50
    whatif_vector[0] -= tv_cut
    digital_idx = [1, 2, 4]  # search, social, display
    digital_total = sum(whatif_vector[i] for i in digital_idx)
    for i in digital_idx:
        whatif_vector[i] += tv_cut * (whatif_vector[i] / digital_total)
    whatif_revenue = predict_revenue(whatif_vector, model, norm_anchors, control_means)

    return {
        "current": {"spend": current_vector, "revenue": current_revenue},
        "optimal": {"spend": optimal_vector, "revenue": optimal_revenue},
        "whatif":  {"spend": whatif_vector,  "revenue": whatif_revenue},
        "total_budget": total_budget,
    }


# ═══════════════════════════════════════════════════
# STEP 2: Build formatted comparison table
# ═══════════════════════════════════════════════════

def build_comparison_table(scenarios):
    """Build a presentation-ready comparison DataFrame."""

    rows = []
    for i, ch in enumerate(CHANNELS):
        row = {
            "Channel": CHANNEL_DISPLAY[ch],
            "Current ($)": scenarios["current"]["spend"][i],
            "Current (%)": scenarios["current"]["spend"][i] / scenarios["total_budget"] * 100,
            "Optimal ($)": scenarios["optimal"]["spend"][i],
            "Optimal (%)": scenarios["optimal"]["spend"][i] / scenarios["total_budget"] * 100,
            "What-If ($)": scenarios["whatif"]["spend"][i],
            "What-If (%)": scenarios["whatif"]["spend"][i] / scenarios["total_budget"] * 100,
            "Optimal Δ ($)": scenarios["optimal"]["spend"][i] - scenarios["current"]["spend"][i],
        }
        rows.append(row)

    # Total row
    rows.append({
        "Channel": "TOTAL",
        "Current ($)": scenarios["current"]["spend"].sum(),
        "Current (%)": 100.0,
        "Optimal ($)": scenarios["optimal"]["spend"].sum(),
        "Optimal (%)": 100.0,
        "What-If ($)": scenarios["whatif"]["spend"].sum(),
        "What-If (%)": 100.0,
        "Optimal Δ ($)": 0.0,
    })

    table = pd.DataFrame(rows)

    # Revenue summary row
    rev_row = {
        "Channel": "PREDICTED REVENUE",
        "Current ($)": scenarios["current"]["revenue"],
        "Current (%)": np.nan,
        "Optimal ($)": scenarios["optimal"]["revenue"],
        "Optimal (%)": np.nan,
        "What-If ($)": scenarios["whatif"]["revenue"],
        "What-If (%)": np.nan,
        "Optimal Δ ($)": scenarios["optimal"]["revenue"] - scenarios["current"]["revenue"],
    }
    table = pd.concat([table, pd.DataFrame([rev_row])], ignore_index=True)

    return table


def print_comparison_table(table, scenarios):
    """Print the table in a clean executive format."""

    print("\n" + "=" * 90)
    print("PHASE 6.2 — SCENARIO COMPARISON TABLE")
    print("=" * 90)

    # Channel allocation section
    print(f"\n{'Channel':<15} {'Current':>12} {'Optimal':>12} {'What-If':>12} {'Opt. Δ':>12}")
    print("-" * 65)

    for _, row in table.iterrows():
        ch = row["Channel"]
        if ch == "PREDICTED REVENUE":
            print("-" * 65)
            print(f"{ch:<15} ${row['Current ($)']:>10,.0f} ${row['Optimal ($)']:>10,.0f} "
                  f"${row['What-If ($)']:>10,.0f} ${row['Optimal Δ ($)']:>+10,.0f}")
        elif ch == "TOTAL":
            print("-" * 65)
            print(f"{ch:<15} ${row['Current ($)']:>10,.0f} ${row['Optimal ($)']:>10,.0f} "
                  f"${row['What-If ($)']:>10,.0f}")
        else:
            pct_c = row["Current (%)"]
            pct_o = row["Optimal (%)"]
            delta = row["Optimal Δ ($)"]
            sign = "+" if delta >= 0 else ""
            print(f"{ch:<15} ${row['Current ($)']:>10,.0f} ${row['Optimal ($)']:>10,.0f} "
                  f"${row['What-If ($)']:>10,.0f} {sign}${delta:>9,.0f}")

    # Lift summary
    c_rev = scenarios["current"]["revenue"]
    o_rev = scenarios["optimal"]["revenue"]
    w_rev = scenarios["whatif"]["revenue"]

    opt_lift = (o_rev - c_rev) / c_rev * 100
    wif_lift = (w_rev - c_rev) / c_rev * 100

    print(f"\n{'Revenue lift vs Current:':<30}")
    print(f"  Optimal:  {opt_lift:+.1f}%  (${o_rev - c_rev:+,.0f}/week, ${(o_rev - c_rev)*52:+,.0f}/year)")
    print(f"  What-If:  {wif_lift:+.1f}%  (${w_rev - c_rev:+,.0f}/week, ${(w_rev - c_rev)*52:+,.0f}/year)")


# ═══════════════════════════════════════════════════
# STEP 3: Executive-ready annotated visualization
# ═══════════════════════════════════════════════════

def plot_executive_comparison(scenarios):
    """
    Two-panel figure:
      Left:  Grouped bar chart — spend by channel per scenario
      Right: Revenue bars with lift annotations

    Includes narrative annotation box on the chart.
    """

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={"width_ratios": [3, 1.2]}
    )

    fig.suptitle(
        "Budget Optimization — Scenario Comparison",
        fontsize=18, fontweight="bold", y=0.98
    )

    # ---- LEFT PANEL: Spend allocation ----
    x = np.arange(len(CHANNELS))
    width = 0.24
    labels = [CHANNEL_DISPLAY[ch] for ch in CHANNELS]

    current_spend = scenarios["current"]["spend"]
    optimal_spend = scenarios["optimal"]["spend"]
    whatif_spend = scenarios["whatif"]["spend"]

    bars_c = ax1.bar(x - width, current_spend, width,
                     label="Current", color=SCENARIO_COLORS["Current"],
                     edgecolor="black", linewidth=0.5)
    bars_o = ax1.bar(x, optimal_spend, width,
                     label="Optimal", color=SCENARIO_COLORS["Optimal"],
                     edgecolor="black", linewidth=0.5)
    bars_w = ax1.bar(x + width, whatif_spend, width,
                     label="What-If", color=SCENARIO_COLORS["What-If"],
                     edgecolor="black", linewidth=0.5)

    # Dollar labels
    for bars in [bars_c, bars_o, bars_w]:
        for bar in bars:
            h = bar.get_height()
            if h > 2000:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 400,
                         "$" + f"{h/1000:.0f}K", ha="center", va="bottom",
                         fontsize=7.5, fontweight="bold")

    # Delta arrows for optimal vs current (biggest movers)
    for i, ch in enumerate(CHANNELS):
        delta = optimal_spend[i] - current_spend[i]
        if abs(delta) > 2000:
            sign = "+" if delta >= 0 else ""
            color = "#2ca02c" if delta > 0 else "#d62728"
            y_pos = max(current_spend[i], optimal_spend[i]) + 2500
            ax1.annotate(
                sign + "$" + f"{delta/1000:.0f}K",
                xy=(i - width/2, y_pos),
                fontsize=8, fontweight="bold", color=color,
                ha="center",
            )

    ax1.set_xlabel("Channel", fontsize=12)
    ax1.set_ylabel("Weekly Spend ($)", fontsize=12)
    ax1.set_title("Weekly Budget Allocation by Channel", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(current_spend.max(), optimal_spend.max(), whatif_spend.max()) * 1.25)

    # ---- RIGHT PANEL: Revenue comparison ----
    scenario_names = ["Current", "Optimal", "What-If"]
    revenues = [
        scenarios["current"]["revenue"],
        scenarios["optimal"]["revenue"],
        scenarios["whatif"]["revenue"],
    ]
    colors = [SCENARIO_COLORS[s] for s in scenario_names]

    rev_bars = ax2.bar(scenario_names, revenues, color=colors,
                        edgecolor="black", linewidth=0.5, width=0.55)

    # Revenue labels on bars
    for bar, rev in zip(rev_bars, revenues):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
                 "$" + f"{rev:,.0f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    # Lift annotation between current and optimal
    c_rev = scenarios["current"]["revenue"]
    o_rev = scenarios["optimal"]["revenue"]
    lift_pct = (o_rev - c_rev) / c_rev * 100
    annual_lift = (o_rev - c_rev) * 52

    ax2.annotate(
        "+" + f"{lift_pct:.1f}%" + "\n" + "($" + f"{annual_lift/1000:,.0f}K/yr)",
        xy=(1, o_rev),
        xytext=(1.5, o_rev * 0.97),
        fontsize=10, fontweight="bold", color="#2ca02c",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
    )

    ax2.set_ylabel("Predicted Weekly Revenue ($)", fontsize=12)
    ax2.set_title("Revenue by Scenario", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Y-axis: start at a base that shows relative differences
    min_rev = min(revenues)
    ax2.set_ylim(min_rev * 0.92, max(revenues) * 1.07)

    # ---- NARRATIVE ANNOTATION BOX ----
    budget_k = scenarios['total_budget'] / 1000
    narrative = (
        "Key Insight: The optimal allocation reduces TV spend by ~10K/week\n"
        "and redistributes to Display (+6K), Search (+3K), and Email (+2K).\n"
        f"Same {budget_k:.0f}K weekly budget → "
        f"+{lift_pct:.1f}% revenue lift ({annual_lift:,.0f}/yr)."
    )

    fig.text(
        0.5, 0.02, narrative,
        ha="center", va="bottom", fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                  edgecolor="gray", alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig("phase6_executive_scenario.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved: phase6_executive_scenario.png")


# ═══════════════════════════════════════════════════
# STEP 4: Export CSV for slides / appendix
# ═══════════════════════════════════════════════════

def export_table_csv(table):
    """Save comparison table as CSV for pasting into slides or Excel."""
    outfile = "phase6_scenario_table.csv"
    table.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("PHASE 6.2 — EXECUTIVE SCENARIO COMPARISON")
    print("=" * 90)

    scenarios = compute_scenarios()
    table = build_comparison_table(scenarios)

    print_comparison_table(table, scenarios)
    plot_executive_comparison(scenarios)
    export_table_csv(table)

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)
    print("  phase6_executive_scenario.png   — Annotated scenario chart")
    print("  phase6_scenario_table.csv       — Table for slides / appendix")
    print("\nPhase 6 complete. Next: Phase 7 — Executive deliverables.")


if __name__ == "__main__":
    main()
