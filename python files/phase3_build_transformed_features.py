import numpy as np
import pandas as pd

from config import CHANNEL_PARAMS
from transforms import apply_pipeline


INPUT_FILE = "mmm_dataset.csv"
OUTPUT_FILE = "mmm_dataset_transformed.csv"

CHANNEL_MAP = {
    "tv": "tv_spend",
    "search": "search_spend",
    "social": "social_spend",
    "email": "email_spend",
    "display": "display_spend",
}


def validate_input_columns(df: pd.DataFrame) -> None:
    required = ["week", "seasonality", "holiday_flag", "holiday_lift", "competitor_promo", "revenue"]
    required += list(CHANNEL_MAP.values())

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataset: {missing}")


def add_transformed_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for channel, spend_col in CHANNEL_MAP.items():
        params = CHANNEL_PARAMS[channel]
        spend_array = out[spend_col].to_numpy(dtype=float)

        result = apply_pipeline(
            spend_array=spend_array,
            decay_rate=params["decay"],
            hill_K=params["hill_K"],
            hill_S=params["hill_S"],
        )

        out[f"{channel}_adstocked"] = result["adstocked"]
        out[f"{channel}_normalized"] = result["normalized"]
        out[f"{channel}_saturated"] = result["saturated"]

    return out


def run_sanity_checks(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("PHASE 3 SANITY CHECKS — TRANSFORMED FEATURES")
    print("=" * 70)

    for channel in CHANNEL_MAP:
        raw_col = f"{channel}_spend"
        ad_col = f"{channel}_adstocked"
        norm_col = f"{channel}_normalized"
        sat_col = f"{channel}_saturated"

        raw_std = df[raw_col].std()
        ad_std = df[ad_col].std()
        sat_min = df[sat_col].min()
        sat_max = df[sat_col].max()
        norm_min = df[norm_col].min()
        norm_max = df[norm_col].max()
        null_count = df[[ad_col, norm_col, sat_col]].isnull().sum().sum()

        print(f"\n{channel.upper()}")
        print(f"  Raw std:          {raw_std:,.2f}")
        print(f"  Adstock std:      {ad_std:,.2f}")
        print(f"  Normalized range: [{norm_min:.4f}, {norm_max:.4f}]")
        print(f"  Saturated range:  [{sat_min:.4f}, {sat_max:.4f}]")
        print(f"  Null values:      {null_count}")

        if null_count > 0:
            raise ValueError(f"Null values found in transformed columns for {channel}")

        if norm_min < -1e-9 or norm_max > 1 + 1e-9:
            raise ValueError(f"Normalized values out of [0,1] range for {channel}")

        if sat_min < -1e-9 or sat_max > 1 + 1e-9:
            raise ValueError(f"Saturated values out of [0,1] range for {channel}")

    print("\n" + "=" * 70)
    print("RELATIVE SMOOTHING CHECK")
    print("=" * 70)

    tv_ratio = df["tv_adstocked"].std() / df["tv_spend"].std()
    search_ratio = df["search_adstocked"].std() / df["search_spend"].std()

    print(f"TV adstock/raw std ratio:      {tv_ratio:.3f}")
    print(f"Search adstock/raw std ratio:  {search_ratio:.3f}")

    print("\nInterpretation:")
print("  Note: std ratio is not a pure smoothing metric because adstock accumulates carryover and can increase scale.")
print("  Use time-series plots or week-to-week change metrics to evaluate smoothness.")
print("  TV should generally smooth more than Search because TV decay is much higher.")
    print("\n" + "=" * 70)
    print("CORRELATION CHECK: SATURATED FEATURES VS REVENUE")
    print("=" * 70)

    for channel in CHANNEL_MAP:
        r = df[f"{channel}_saturated"].corr(df["revenue"])
        print(f"  {channel.upper():8s} r = {r:+.3f}")

    print("\nDone. Dataset is ready for Phase 4 regression.")


def main() -> None:
    print("=" * 70)
    print("PHASE 3 — BUILD TRANSFORMED MMM FEATURES")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {INPUT_FILE}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    validate_input_columns(df)

    transformed_df = add_transformed_features(df)
    run_sanity_checks(transformed_df)

    transformed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Final shape: {transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")

    print("\nNew columns added:")
    for channel in CHANNEL_MAP:
        print(f"  {channel}_adstocked, {channel}_normalized, {channel}_saturated")


if __name__ == "__main__":
    main()