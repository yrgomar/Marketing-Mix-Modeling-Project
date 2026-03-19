import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


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

CHANNEL_LABELS = {
    "tv_saturated": "TV",
    "search_saturated": "Paid Search",
    "social_saturated": "Social Media",
    "email_saturated": "Email",
    "display_saturated": "Display",
    "seasonality": "Seasonality",
    "holiday_lift": "Holiday Lift",
    "competitor_promo": "Competitor Promo",
}

TRAIN_WEEKS = 80


def validate_columns(df: pd.DataFrame) -> None:
    required = ["week", TARGET_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def fit_ols(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X, y


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_model_summary(model) -> None:
    print_header("PHASE 4.1 — OLS REGRESSION SUMMARY")
    print(model.summary())

    print("\nKey checks:")
    print(f"  R-squared:           {model.rsquared:.4f}")
    print(f"  Adjusted R-squared:  {model.rsquared_adj:.4f}")
    print(f"  Durbin-Watson:       {sm.stats.stattools.durbin_watson(model.resid):.4f}")

    print("\nChannel coefficient signs:")
    for col in CHANNEL_FEATURES:
        coef = model.params[col]
        sign = "POSITIVE" if coef > 0 else "NEGATIVE"
        print(f"  {CHANNEL_LABELS[col]:15s}  {coef:>12,.2f}  {sign}")


def build_coefficient_table(model) -> pd.DataFrame:
    rows = []
    for term in ["const"] + FEATURE_COLS:
        rows.append({
            "term": term,
            "label": "Intercept" if term == "const" else CHANNEL_LABELS[term],
            "coefficient": model.params[term],
            "std_error": model.bse[term],
            "t_stat": model.tvalues[term],
            "p_value": model.pvalues[term],
            "ci_lower": model.conf_int().loc[term, 0],
            "ci_upper": model.conf_int().loc[term, 1],
        })
    return pd.DataFrame(rows)


def print_coefficient_table(model) -> None:
    coef_table = build_coefficient_table(model)

    print_header("PHASE 4.4 — COEFFICIENT TABLE")
    display_cols = ["label", "coefficient", "std_error", "t_stat", "p_value", "ci_lower", "ci_upper"]
    print(coef_table[display_cols].round(4).to_string(index=False))


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X = sm.add_constant(X)

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        if vif > 10:
            status = "SERIOUS"
        elif vif > 5:
            status = "CONCERNING"
        else:
            status = "OK"

        rows.append({
            "variable": CHANNEL_LABELS[col],
            "vif": vif,
            "status": status
        })

    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def print_vif_table(df: pd.DataFrame) -> None:
    vif_df = calculate_vif(df)

    print_header("PHASE 4.2 — VIF / MULTICOLLINEARITY CHECK")
    print(vif_df.round(4).to_string(index=False))


def run_train_test_validation(df: pd.DataFrame):
    train_df = df.iloc[:TRAIN_WEEKS].copy()
    test_df = df.iloc[TRAIN_WEEKS:].copy()

    X_train = sm.add_constant(train_df[FEATURE_COLS], has_constant="add")
    y_train = train_df[TARGET_COL]

    X_test = sm.add_constant(test_df[FEATURE_COLS], has_constant="add")
    y_test = test_df[TARGET_COL]

    model = sm.OLS(y_train, X_train).fit()

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print_header("PHASE 4.3 — TRAIN / TEST VALIDATION")
    print(f"Train weeks: 1-{TRAIN_WEEKS}")
    print(f"Test weeks:  {TRAIN_WEEKS + 1}-{len(df)}")
    print(f"\nTraining MAPE: {train_mape:.2f}%")
    print(f"Test MAPE:     {test_mape:.2f}%")
    print(f"Training RMSE: ${train_rmse:,.0f}")
    print(f"Test RMSE:     ${test_rmse:,.0f}")

    if test_mape <= 15:
        print("\nInterpretation: Test MAPE is in a strong MMM range.")
    elif test_mape <= 20:
        print("\nInterpretation: Test MAPE is acceptable but could be improved.")
    else:
        print("\nInterpretation: Test MAPE is weak — revisit features/specification.")

    return model, train_df, test_df, train_pred, test_pred


def build_contribution_table(model, df: pd.DataFrame) -> pd.DataFrame:
    mean_revenue = df[TARGET_COL].mean()

    rows = []

    intercept = model.params["const"]
    base_pct = (intercept / mean_revenue) * 100
    rows.append({
        "component": "Base Revenue",
        "coefficient": intercept,
        "mean_feature_value": 1.0,
        "mean_weekly_contribution": intercept,
        "pct_of_mean_revenue": base_pct
    })

    for col in FEATURE_COLS:
        coef = model.params[col]
        feature_mean = df[col].mean()
        contrib = coef * feature_mean
        pct = (contrib / mean_revenue) * 100

        rows.append({
            "component": CHANNEL_LABELS[col],
            "coefficient": coef,
            "mean_feature_value": feature_mean,
            "mean_weekly_contribution": contrib,
            "pct_of_mean_revenue": pct
        })

    contrib_df = pd.DataFrame(rows)
    return contrib_df.sort_values("mean_weekly_contribution", ascending=False)


def print_contribution_table(model, df: pd.DataFrame) -> None:
    contrib_df = build_contribution_table(model, df)

    print_header("PHASE 4.4 — REVENUE CONTRIBUTION TABLE")
    print(contrib_df.round(4).to_string(index=False))

    total_pct = contrib_df["pct_of_mean_revenue"].sum()
    print(f"\nCheck: contribution percentages sum to {total_pct:.2f}% of mean revenue")


def plot_diagnostics(model, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> None:
    fitted = model.fittedvalues
    resid = model.resid
    std_resid = (resid - resid.mean()) / resid.std(ddof=1)

    # Plot 1: Residuals vs fitted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, resid, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted Revenue")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase4_residuals_vs_fitted.png", dpi=150)
    plt.show()

    # Plot 2: Q-Q plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    sm.qqplot(resid, line="45", fit=True, ax=ax)
    ax.set_title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig("phase4_qq_plot.png", dpi=150)
    plt.show()

    # Plot 3: Actual vs predicted over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["week"], y, label="Actual Revenue", linewidth=1.8)
    ax.plot(df["week"], fitted, label="Predicted Revenue", linewidth=1.8)
    ax.set_title("Actual vs Predicted Revenue")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase4_actual_vs_predicted.png", dpi=150)
    plt.show()

    # Plot 4: Standardized residuals over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["week"], std_resid, linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--")
    ax.axhline(2, color="red", linestyle="--", alpha=0.7)
    ax.axhline(-2, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Standardized Residuals Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Standardized Residual")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase4_standardized_residuals.png", dpi=150)
    plt.show()

    print_header("PHASE 4.2 — DIAGNOSTIC TESTS")
    bp_test = het_breuschpagan(resid, X)
    bp_labels = ["LM Stat", "LM p-value", "F Stat", "F p-value"]
    print("Breusch-Pagan Test:")
    for label, value in zip(bp_labels, bp_test):
        print(f"  {label:12s}: {value:.4f}")

    shapiro_stat, shapiro_p = stats.shapiro(resid)
    print(f"\nShapiro-Wilk Test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  p-value:   {shapiro_p:.4f}")

    dw = sm.stats.stattools.durbin_watson(resid)
    print(f"\nDurbin-Watson: {dw:.4f}")


def compare_to_ground_truth() -> None:
    ground_truth = {
        "tv_saturated": 0.15 * 200_000,
        "search_saturated": 0.25 * 200_000,
        "social_saturated": 0.18 * 200_000,
        "email_saturated": 0.10 * 200_000,
        "display_saturated": 0.12 * 200_000,
    }

    print_header("GROUND-TRUTH REFERENCE")
    print("These are the approximate theoretical weekly contribution scales implied")
    print("by generate_data.py: coefficient × base_revenue")
    for k, v in ground_truth.items():
        print(f"  {CHANNEL_LABELS[k]:15s}: ${v:,.0f} per unit saturated effect")


def main():
    print_header("PHASE 4 — MMM REGRESSION MODEL")

    df = pd.read_csv(INPUT_FILE)
    validate_columns(df)

    print(f"Loaded {INPUT_FILE}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    model, X, y = fit_ols(df)

    print_model_summary(model)
    print_coefficient_table(model)
    print_vif_table(df)
    print_contribution_table(model, df)
    plot_diagnostics(model, X, y, df)
    compare_to_ground_truth()

    _, _, _, _, _ = run_train_test_validation(df)

    print_header("FILES SAVED")
    print("phase4_residuals_vs_fitted.png")
    print("phase4_qq_plot.png")
    print("phase4_actual_vs_predicted.png")
    print("phase4_standardized_residuals.png")

    print("\nNext step: Phase 5 — decomposition visuals and ROI charts.")


if __name__ == "__main__":
    main()