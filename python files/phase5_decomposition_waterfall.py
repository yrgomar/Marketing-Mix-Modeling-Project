import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


INPUT_FILE = "mmm_dataset_transformed.csv"

TARGET_COL = "revenue"
FEATURE_COLS = [
    "tv_saturated",
    "search_saturated",
    "social_saturated",
    "email_saturated",
    "display_saturated",
    "seasonality",
    "holiday_lift",
    "competitor_promo",
]

LABELS = {
    "const": "Base Revenue",
    "tv_saturated": "TV",
    "search_saturated": "Paid Search",
    "social_saturated": "Social Media",
    "email_saturated": "Email",
    "display_saturated": "Display",
    "seasonality": "Seasonality",
    "holiday_lift": "Holiday Lift",
    "competitor_promo": "Competitor Promo",
}


def fit_model(df: pd.DataFrame):
    X = sm.add_constant(df[FEATURE_COLS], has_constant="add")
    y = df[TARGET_COL]
    model = sm.OLS(y, X).fit()
    return model, X, y


def build_weekly_contributions(model, df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    intercept = model.params["const"]
    rows.append({
        "component": "Base Revenue",
        "weekly_contribution": intercept
    })

    for col in FEATURE_COLS:
        contribution = model.params[col] * df[col].mean()
        rows.append({
            "component": LABELS[col],
            "weekly_contribution": contribution
        })

    contrib_df = pd.DataFrame(rows)
    return contrib_df


def print_contribution_table(contrib_df: pd.DataFrame, mean_revenue: float) -> None:
    out = contrib_df.copy()
    out["pct_of_mean_revenue"] = out["weekly_contribution"] / mean_revenue * 100

    print("\n" + "=" * 72)
    print("PHASE 5.1 — WEEKLY REVENUE DECOMPOSITION")
    print("=" * 72)
    print(out.round(2).to_string(index=False))

    print(f"\nMean actual weekly revenue: ${mean_revenue:,.0f}")
    print(f"Sum of weekly contributions: ${out['weekly_contribution'].sum():,.0f}")


def plot_waterfall(contrib_df: pd.DataFrame, mean_revenue: float) -> None:
    plot_df = contrib_df.copy()

    # Rename labels for cleaner executive presentation
    rename_map = {
        "Paid Search": "Search",
        "Social Media": "Social",
        "Seasonality": "Net Seasonal Effect",
        "Competitor Promo": "Competitor",
    }
    plot_df["component"] = plot_df["component"].replace(rename_map)

    # Executive ordering: biggest story first
    desired_order = [
        "Base Revenue",
        "Search",
        "Display",
        "Email",
        "TV",
        "Social",
        "Holiday Lift",
        "Net Seasonal Effect",
        "Competitor",
    ]

    plot_df["order"] = plot_df["component"].map({name: i for i, name in enumerate(desired_order)})
    plot_df = plot_df.sort_values("order").drop(columns="order").reset_index(drop=True)

    # Compute floating bar starts
    starts = []
    running_total = 0

    for _, row in plot_df.iterrows():
        value = row["weekly_contribution"]
        if value >= 0:
            starts.append(running_total)
        else:
            starts.append(running_total + value)
        running_total += value

    plot_df["start"] = starts
    plot_df["color"] = np.where(plot_df["weekly_contribution"] >= 0, "#2ca02c", "#d62728")

    total_predicted = plot_df["weekly_contribution"].sum()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Floating bars
    for i, row in plot_df.iterrows():
        ax.bar(
            x=i,
            height=abs(row["weekly_contribution"]),
            bottom=row["start"],
            color=row["color"],
            edgecolor="black",
            width=0.7
        )

        y_text = row["start"] + abs(row["weekly_contribution"]) / 2
        ax.text(
            i,
            y_text,
            f"${row['weekly_contribution']:,.0f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if abs(row["weekly_contribution"]) > 15000 else "black"
        )

    # Final total bar
    ax.bar(
        x=len(plot_df),
        height=total_predicted,
        bottom=0,
        color="#1f77b4",
        edgecolor="black",
        width=0.7
    )
    ax.text(
        len(plot_df),
        total_predicted / 2,
        f"${total_predicted:,.0f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white"
    )

    # Mean actual revenue reference line
    ax.axhline(
        mean_revenue,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Actual Revenue = ${mean_revenue:,.0f}"
    )

    # Connector lines
    running = 0
    for i, row in plot_df.iterrows():
        running += row["weekly_contribution"]
        if i < len(plot_df) - 1:
            ax.plot([i + 0.35, i + 0.65], [running, running], color="black", linewidth=1)

    x_labels = list(plot_df["component"]) + ["Total Predicted"]

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Weekly Revenue Contribution ($)")
    ax.set_title("Revenue Decomposition Waterfall — Average Weekly Revenue", fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase5_waterfall_chart.png", dpi=150)
    plt.show()

    print("\nSaved: phase5_waterfall_chart.png")
    print(f"Total predicted weekly revenue: ${total_predicted:,.0f}")
    print(f"Mean actual weekly revenue:     ${mean_revenue:,.0f}")
    print(f"Difference:                    ${total_predicted - mean_revenue:,.2f}")


def main():
    df = pd.read_csv(INPUT_FILE)

    model, X, y = fit_model(df)
    contrib_df = build_weekly_contributions(model, df)
    mean_revenue = df[TARGET_COL].mean()

    print_contribution_table(contrib_df, mean_revenue)
    plot_waterfall(contrib_df, mean_revenue)


if __name__ == "__main__":
    main()