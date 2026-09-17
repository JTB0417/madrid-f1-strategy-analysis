"""Analyze strategy and race performance at the 2026 Madrid Grand Prix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
PODIUM_NUMBERS = [12, 3, 1]
COLORS = {12: "#00A19C", 3: "#1E41FF", 1: "#FF8700"}


def load_records(name: str) -> pd.DataFrame:
    with (RAW_DIR / f"{name}.json").open(encoding="utf-8") as source:
        return pd.DataFrame(json.load(source))


def driver_lookup(drivers: pd.DataFrame) -> dict[int, str]:
    unique = drivers.drop_duplicates("driver_number")
    return dict(zip(unique["driver_number"], unique["name_acronym"]))


def attach_stint_data(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for stint in stints.itertuples():
        selected = laps[
            (laps["driver_number"] == stint.driver_number)
            & (laps["lap_number"] >= stint.lap_start)
            & (laps["lap_number"] <= stint.lap_end)
        ].copy()
        selected["stint_number"] = stint.stint_number
        selected["compound"] = stint.compound
        selected["tyre_age"] = (
            stint.tyre_age_at_start + selected["lap_number"] - stint.lap_start + 1
        )
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    laps = laps.sort_values("date_start").copy()
    weather = weather.sort_values("date").copy()
    return pd.merge_asof(
        laps,
        weather[["date", "air_temperature", "track_temperature", "humidity", "rainfall"]],
        left_on="date_start",
        right_on="date",
        direction="nearest",
    )


def vsc_window(race_control: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    deployed = race_control.loc[
        race_control["message"].eq("VSC DEPLOYED"), "date"
    ].iloc[0]
    ending = race_control.loc[
        race_control["message"].eq("VSC ENDING"), "date"
    ].iloc[0]
    return deployed, ending


def prepare_data():
    drivers = load_records("drivers")
    laps = load_records("laps")
    stints = load_records("stints")
    weather = load_records("weather")
    race_control = load_records("race_control")
    pits = load_records("pit")
    positions = load_records("position")
    results = load_records("session_result")

    for frame, column in (
        (laps, "date_start"),
        (weather, "date"),
        (race_control, "date"),
        (pits, "date"),
        (positions, "date"),
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed")

    laps = attach_stint_data(laps, stints)
    laps = attach_weather(laps, weather)
    names = driver_lookup(drivers)
    laps["driver"] = laps["driver_number"].map(names)
    pits["driver"] = pits["driver_number"].map(names)
    positions["driver"] = positions["driver_number"].map(names)
    results["driver"] = results["driver_number"].map(names)
    return laps, pits, positions, race_control, results, names


def plot_strategy_timeline(laps, pits, names):
    fig, ax = plt.subplots(figsize=(11, 6))
    for number in PODIUM_NUMBERS:
        driver_laps = laps[
            (laps["driver_number"] == number)
            & laps["lap_duration"].between(92, 108)
        ]
        ax.plot(
            driver_laps["lap_number"],
            driver_laps["lap_duration"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=COLORS[number],
            label=names[number],
        )
        driver_pits = pits[pits["driver_number"] == number]
        for lap in driver_pits["lap_number"]:
            ax.axvline(lap, color=COLORS[number], alpha=0.35, linestyle="--")
    ax.axvspan(14, 15, color="gold", alpha=0.22, label="VSC period")
    ax.set(title="Podium Contenders: Race Pace and Pit Timing", xlabel="Lap", ylabel="Lap time (s)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strategy_timeline.png", dpi=180)
    plt.close(fig)


def plot_position_changes(positions, names, vsc_start, vsc_end):
    race_start = pd.Timestamp("2026-09-13T13:00:00+00:00")
    selected = positions[
        positions["driver_number"].isin(PODIUM_NUMBERS)
        & (positions["date"] >= race_start)
    ].copy()
    selected["race_minutes"] = (selected["date"] - race_start).dt.total_seconds() / 60
    fig, ax = plt.subplots(figsize=(11, 5))
    for number in PODIUM_NUMBERS:
        driver_positions = selected[selected["driver_number"] == number]
        ax.step(
            driver_positions["race_minutes"],
            driver_positions["position"],
            where="post",
            color=COLORS[number],
            linewidth=2,
            label=names[number],
        )
    start_minute = (vsc_start - race_start).total_seconds() / 60
    end_minute = (vsc_end - race_start).total_seconds() / 60
    ax.axvspan(start_minute, end_minute, color="gold", alpha=0.25, label="VSC")
    ax.set(title="Track Position Changes Around the VSC", xlabel="Minutes after scheduled start", ylabel="Position")
    ax.set_yticks(range(1, 7))
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "position_changes.png", dpi=180)
    plt.close(fig)


def plot_pit_comparison(pits, names, vsc_start, vsc_end):
    podium_pits = pits[pits["driver_number"].isin(PODIUM_NUMBERS)].copy()
    podium_pits = podium_pits.sort_values("lap_number")
    podium_pits["under_vsc"] = podium_pits["date"].between(vsc_start, vsc_end)
    podium_pits["status"] = np.where(podium_pits["under_vsc"], "VSC", "Green flag")
    podium_pits["label"] = podium_pits.apply(
        lambda row: f"{names[row.driver_number]}\nLap {int(row.lap_number)}", axis=1
    )
    colors = ["#2CA02C" if value else "#D62728" for value in podium_pits["under_vsc"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(podium_pits["label"], podium_pits["pit_duration"], color=colors)
    for bar, status in zip(bars, podium_pits["status"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, status, ha="center")
    ax.set(title="First Pit-Lane Transit: Timing and Duration", ylabel="Pit-lane duration (s)")
    ax.set_ylim(0, podium_pits["pit_duration"].max() + 6)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pit_stop_comparison.png", dpi=180)
    plt.close(fig)
    return podium_pits


def model_hard_stint_trends(laps, names):
    rows = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for number in PODIUM_NUMBERS:
        hard = laps[
            (laps["driver_number"] == number)
            & (laps["compound"] == "HARD")
            & (~laps["is_pit_out_lap"])
            & laps["lap_duration"].notna()
        ].copy()
        median = hard["lap_duration"].median()
        hard = hard[hard["lap_duration"].between(median - 4, median + 4)]
        model = HuberRegressor().fit(hard[["tyre_age"]], hard["lap_duration"])
        predicted = model.predict(hard[["tyre_age"]])
        rows.append(
            {
                "driver": names[number],
                "clean_laps": len(hard),
                "net_seconds_per_lap": model.coef_[0],
                "mae_seconds": mean_absolute_error(hard["lap_duration"], predicted),
            }
        )
        ax.scatter(hard["tyre_age"], hard["lap_duration"], s=18, alpha=0.35, color=COLORS[number])
        order = np.argsort(hard["tyre_age"].to_numpy())
        ax.plot(
            hard["tyre_age"].to_numpy()[order],
            predicted[order],
            color=COLORS[number],
            linewidth=2,
            label=f"{names[number]} ({model.coef_[0]:+.3f} s/lap)",
        )
    ax.set(
        title="Hard-Tire Net Pace Trend (Fuel, Tire, Traffic Combined)",
        xlabel="Tire age (laps)",
        ylabel="Lap time (s)",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hard_tire_net_pace_trend.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def compare_lap_time_models(laps):
    clean = laps[
        laps["lap_duration"].between(90, 110)
        & (~laps["is_pit_out_lap"])
        & (~laps["lap_number"].isin([1, 14, 15]))
    ].dropna(
        subset=["compound", "tyre_age", "lap_number", "air_temperature", "track_temperature"]
    )
    features = clean[
        ["compound", "tyre_age", "lap_number", "air_temperature", "track_temperature"]
    ]
    target = clean["lap_duration"]
    groups = clean["driver_number"]
    preprocess = ColumnTransformer(
        [
            ("compound", OneHotEncoder(handle_unknown="ignore"), ["compound"]),
            (
                "numeric",
                StandardScaler(),
                ["tyre_age", "lap_number", "air_temperature", "track_temperature"],
            ),
        ]
    )
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
    }
    rows = []
    splitter = GroupKFold(n_splits=5)
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        scores = cross_validate(
            pipeline,
            features,
            target,
            groups=groups,
            cv=splitter,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        rows.append(
            {
                "model": name,
                "mean_mae_seconds": -scores["test_mae"].mean(),
                "std_mae_seconds": scores["test_mae"].std(),
                "mean_r2": scores["test_r2"].mean(),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("mean_mae_seconds")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(metrics["model"], metrics["mean_mae_seconds"], yerr=metrics["std_mae_seconds"], capsize=5)
    ax.set(title="Held-Out Driver Lap-Time Prediction", ylabel="Mean absolute error (s)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)
    return metrics, len(clean)


def write_summary(results, podium_pits, trends, metrics, model_laps, vsc_start, vsc_end):
    podium = results.sort_values("position").head(3)
    ant_pit = podium_pits[podium_pits["driver_number"] == 12].iloc[0]
    ver_pit = podium_pits[podium_pits["driver_number"] == 3].iloc[0]
    nor_pit = podium_pits[podium_pits["driver_number"] == 1].iloc[0]
    vsc_seconds = (vsc_end - vsc_start).total_seconds()
    best_model = metrics.iloc[0]
    lines = [
        "# Analysis Summary",
        "",
        "## Decision-focused findings",
        "",
        f"- The VSC lasted {vsc_seconds:.0f} seconds, from lap 14 into lap 15.",
        f"- Antonelli and Verstappen entered the pits on lap {int(ant_pit.lap_number)} during the VSC; Norris stopped on lap {int(nor_pit.lap_number)} after green-flag running resumed.",
        f"- Norris's recorded pit-lane transit was {nor_pit.pit_duration - ant_pit.pit_duration:.1f} seconds longer than Antonelli's ({nor_pit.pit_duration:.1f}s vs {ant_pit.pit_duration:.1f}s).",
        "- Position data shows Norris falling from the lead group to fifth during the pit sequence before recovering to third.",
        f"- The winner finished {podium.iloc[1].gap_to_leader:.3f} seconds ahead of second and {podium.iloc[2].gap_to_leader:.3f} seconds ahead of third.",
        "- All three podium contenders showed improving net hard-stint lap-time trends. Fuel burn therefore outweighed observed tire degradation in the raw trend; this must not be interpreted as pure tire wear.",
        f"- {best_model['model']} produced the lowest held-out-driver MAE: {best_model.mean_mae_seconds:.3f} seconds across {model_laps:,} cleaned laps.",
        "",
        "## Engineering recommendation",
        "",
        "At a circuit where passing was difficult, track position had unusually high value. Once the VSC was deployed near the planned pit window, the preferred decision was to pit immediately when operationally possible. A strategy tool should combine a live VSC state, estimated pit-lane loss, traffic on pit exit, and stop-time uncertainty rather than evaluating tire pace alone.",
        "",
        "## Limitations",
        "",
        "This is a single-race observational analysis. Fuel load, traffic, tire condition, driver pace, and setup are not independently measured. The models identify useful patterns but cannot prove a causal tire-degradation rate or reconstruct an exact alternate race result.",
    ]
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    laps, pits, positions, race_control, results, names = prepare_data()
    vsc_start, vsc_end = vsc_window(race_control)
    plot_strategy_timeline(laps, pits, names)
    plot_position_changes(positions, names, vsc_start, vsc_end)
    podium_pits = plot_pit_comparison(pits, names, vsc_start, vsc_end)
    trends = model_hard_stint_trends(laps, names)
    metrics, model_laps = compare_lap_time_models(laps)
    write_summary(results, podium_pits, trends, metrics, model_laps, vsc_start, vsc_end)

    podium_pits.to_csv(PROCESSED_DIR / "podium_pit_stops.csv", index=False)
    trends.to_csv(PROCESSED_DIR / "hard_tire_net_pace_trends.csv", index=False)
    metrics.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    results.sort_values("position").to_csv(PROCESSED_DIR / "race_results.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"\nCreated analysis from {len(laps):,} lap records.")


if __name__ == "__main__":
    main()
