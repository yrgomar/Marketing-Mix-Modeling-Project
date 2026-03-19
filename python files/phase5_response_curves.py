import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from config import CHANNEL_PARAMS
from transforms import normalize_spend, hill_saturation


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

CHANNEL_NAME_MAP = {
    "tv_saturated": "tv",
    "search_saturated": "search",
    "social_saturated": "social",
    "email_saturated": "email",
    "display_saturated": "display",
}

LABELS = {
    "tv_saturated": "TV",
    "search_saturated": "Search",
    "social_saturated": "Social",
    "email_saturated": "Email",
    "display_saturated": "Display",
}

SPEND_COLS = {
    "tv_saturated": "tv_spend",
    "search_saturated": "search_spend",
    "social_saturated": "social_spend",
    "email_saturated": "email_spend",
    "display_saturated": "display_spend",
}


def fit_model(df: pd.DataFrame):
    X = sm.add_constant(df[FEATURE_COLS], has_constant="add")
    y = df[TARGET_COL]
    model = sm.OLS(y, X).fit()
    return model


def steady_state_adstock(spend, decay):
    """
    For a constant spend level over time, geometric adstock converges to:
        adstock = spend / (1 - decay)
    """
    return spend / (1 - decay)


def build_response_curve(model, df: pd.DataFrame, feature: str, n_points: int = 250) -> pd.DataFrame:
    channel_key = CHANNEL_NAME_MAP[feature]
    spend_col = SPEND_COLS[feature]

    params = CHANNEL_PARAMS[channel_key]
    coef = model.params[feature]

    current_spend = df[spend_col].mean()

    # Build spend range from near-zero to 2.5x current mean
    max_spend = max(current_spend * 2.5, 100)
    spend_range = np.linspace(100, max_spend, n_points)

    # Convert constant spend into steady-state adstock
    adstock_range = steady_state_adstock(spend_range, params["decay"])

    # Normalize within the curve range, matching your earlier project logic
    normalized_range = normalize_spend(adstock_range)

    # Apply Hill saturation
    saturated_range = hill_saturation(normalized_range, params["hill_K"], params["hill_S"])

    # Convert transformed feature into modeled incremental revenue
    incremental_revenue = coef * saturated_range

    # Current point
    current_adstock = steady_state_adstock(current_spend, params["decay"])
    norm_current = (current_adstock - adstock_range.min()) / (adstock_range.max() - adstock_range.min())
    norm_current = np.clip(norm_current, 1e-10, 1.0)
    current_saturated = hill_saturation(np.array([norm_current]), params["hill_K"], params["hill_S"])[0]
    current_incremental_revenue = coef * current_saturated

    curve_df = pd.DataFrame({
        "spend": spend_range,
        "adstock": adstock_range,
        "normalized": normalized_range,
        "saturated": saturated_range,
        "incremental_revenue": incremental_revenue,
    })

    return curve_df, current_spend, current_incremental_revenue


def print_curve_summary(model, df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("PHASE 5.2 — CHANNEL RESPONSE CURVE SUMMARY")
    print("=" * 72)

    rows = []

    for feature in CHANNEL_FEATURES:
        curve_df, current_spend, current_incremental_revenue = build_response_curve(model, df, feature)

        max_revenue = curve_df["incremental_revenue"].max()
        pct_of_curve_max = current_incremental_revenue / max_revenue * 100 if max_revenue != 0 else 0

        rows.append({
            "channel": LABELS[feature],
            "current_weekly_spend": current_spend,
            "current_modeled_incremental_revenue": current_incremental_revenue,
            "max_curve_revenue": max_revenue,
            "current_as_pct_of_curve_max": pct_of_curve_max,
        })

    summary_df = pd.DataFrame(rows).sort_values("current_modeled_incremental_revenue", ascending=False)
    print(summary_df.round(2).to_string(index=False))

    print("\nInterpretation:")
    print("  current_as_pct_of_curve_max estimates how close current spend is to the flat part of the curve.")
    print("  Higher % = more saturated channel.")
    print("  Lower % = more headroom remaining.")


def plot_individual_response_curves(model, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, feature in enumerate(CHANNEL_FEATURES):
        ax = axes[i]
        curve_df, current_spend, current_incremental_revenue = build_response_curve(model, df, feature)

        ax.plot(
            curve_df["spend"] / 1000,
            curve_df["incremental_revenue"] / 1000,
            linewidth=2.5,
            label=LABELS[feature]
        )

        ax.axvline(
            current_spend / 1000,
            linestyle="--",
            linewidth=1.5,
            label=f"Current Spend = ${current_spend:,.0f}"
        )

        ax.scatter(
            current_spend / 1000,
            current_incremental_revenue / 1000,
            s=80,
            zorder=5,
            marker="s"
        )

        # Simple "flattening region" annotation
        y_max = curve_df["incremental_revenue"].max() / 1000
        ax.text(
            0.98,
            0.08,
            "Flat = diminishing returns",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.2)
        )

        ax.set_title(LABELS[feature], fontweight="bold")
        ax.set_xlabel("Weekly Spend ($K)")
        ax.set_ylabel("Modeled Incremental Revenue ($K)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Remove unused 6th subplot
    fig.delaxes(axes[5])

    fig.suptitle("Channel Response Curves — Current Spend Marked with Square", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("phase5_response_curves_grid.png", dpi=150)
    plt.show()

    print("\nSaved: phase5_response_curves_grid.png")


def plot_overlay_response_curves(model, df: pd.DataFrame) -> None:
    _, ax = plt.subplots(figsize=(11, 7))

    for feature in CHANNEL_FEATURES:
        curve_df, current_spend, current_incremental_revenue = build_response_curve(model, df, feature)

        ax.plot(
            curve_df["spend"] / 1000,
            curve_df["incremental_revenue"] / 1000,
            linewidth=2.2,
            label=LABELS[feature]
        )

        ax.scatter(
            current_spend / 1000,
            current_incremental_revenue / 1000,
            s=70,
            marker="s",
            zorder=5
        )

    ax.set_title("Overlay of Channel Response Curves", fontsize=15, fontweight="bold")
    ax.set_xlabel("Weekly Spend ($K)")
    ax.set_ylabel("Modeled Incremental Revenue ($K)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("phase5_response_curves_overlay.png", dpi=150)
    plt.show()

    print("Saved: phase5_response_curves_overlay.png")


def main():
    df = pd.read_csv(INPUT_FILE)
    model = fit_model(df)

    print_curve_summary(model, df)
    plot_individual_response_curves(model, df)
    plot_overlay_response_curves(model, df)

    print("\nNext step: use these curves to support the ROI chart and budget optimizer.")


if __name__ == "__main__":
    main()