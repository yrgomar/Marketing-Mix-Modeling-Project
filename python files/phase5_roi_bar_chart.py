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

CHANNEL_FEATURES = [
    "tv_saturated",
    "search_saturated",
    "social_saturated",
    "email_saturated",
    "display_saturated",
]

CHANNEL_SPEND_COLS = {
    "tv_saturated": "tv_spend",
    "search_saturated": "search_spend",
    "social_saturated": "social_spend",
    "email_saturated": "email_spend",
    "display_saturated": "display_spend",
}

LABELS = {
    "tv_saturated": "TV",
    "search_saturated": "Search",
    "social_saturated": "Social",
    "email_saturated": "Email",
    "display_saturated": "Display",
}


def fit_model(df: pd.DataFrame):
    X = sm.add_constant(df[FEATURE_COLS], has_constant="add")
    y = df[TARGET_COL]
    model = sm.OLS(y, X).fit()
    return model


def build_roi_table(model, df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in CHANNEL_FEATURES:
        spend_col = CHANNEL_SPEND_COLS[feature]

        coef = model.params[feature]
        mean_spend = df[spend_col].mean()
        mean_feature_value = df[feature].mean()

        weekly_contribution = coef * mean_feature_value

        # Average ROI: average weekly contribution / average weekly raw spend
        avg_roi = weekly_contribution / mean_spend

        # Marginal ROI proxy: coefficient per average dollar of spend
        marginal_roi_proxy = coef / mean_spend

        rows.append({
            "channel": LABELS[feature],
            "feature": feature,
            "mean_weekly_spend": mean_spend,
            "coefficient": coef,
            "mean_feature_value": mean_feature_value,
            "weekly_contribution": weekly_contribution,
            "avg_roi": avg_roi,
            "marginal_roi_proxy": marginal_roi_proxy,
        })

    roi_df = pd.DataFrame(rows).sort_values("avg_roi", ascending=False).reset_index(drop=True)
    return roi_df


def print_roi_table(roi_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("PHASE 5.3 — ROI COMPARISON TABLE")
    print("=" * 72)
    display_df = roi_df.copy()
    print(
        display_df[
            [
                "channel",
                "mean_weekly_spend",
                "weekly_contribution",
                "avg_roi",
                "marginal_roi_proxy",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nInterpretation:")
    print("  avg_roi = average weekly modeled contribution / average weekly raw spend")
    print("  Example: avg_roi = 1.20 means about $1.20 of modeled weekly revenue per $1 of weekly spend")
    print("  This is a practical comparison metric for your executive deck.")


def plot_roi_chart(roi_df: pd.DataFrame) -> None:
    plot_df = roi_df.copy()

    average_roi = plot_df["avg_roi"].mean()
    colors = np.where(plot_df["avg_roi"] >= average_roi, "#2ca02c", "#ff7f0e")

    _, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        plot_df["channel"],
        plot_df["avg_roi"],
        color=colors,
        edgecolor="black"
    )

    for bar, value in zip(bars, plot_df["avg_roi"]):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}x",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.axvline(
        average_roi,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Average ROI = {average_roi:.2f}x"
    )

    ax.set_title("Channel ROI Comparison — Average Weekly Revenue per $1 Spend", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average ROI (Modeled Revenue Contribution / Raw Spend)")
    ax.set_ylabel("Channel")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()

    # Highest ROI at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("phase5_roi_bar_chart.png", dpi=150)
    plt.show()

    print("\nSaved: phase5_roi_bar_chart.png")


def main():
    df = pd.read_csv(INPUT_FILE)
    model = fit_model(df)
    roi_df = build_roi_table(model, df)

    print_roi_table(roi_df)
    plot_roi_chart(roi_df)


if __name__ == "__main__":
    main()