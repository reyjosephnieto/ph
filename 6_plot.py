# 6_plot.py
"""Plot grouped panels, single-protocol plots, and orthogonality comparisons."""

from __future__ import annotations

import json
import pathlib
import typing

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import fives_shared as fs

OUTPUT_DIR = pathlib.Path("plot_results")
LEGEND_NCOL = 6
ORTHO_LEGEND_NCOL = 3

# Plot defaults
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 12
BASE_FONT_SIZE = float(plt.rcParams.get("font.size", 12))
LEGEND_FONT_SIZE = BASE_FONT_SIZE * 1.5
ORTHO_LEGEND_FONT_SIZE = BASE_FONT_SIZE * 1.0

PerfRecord = dict[str, typing.Any]
SeriesMap = dict[str, list[float]]

PROTOCOL_META: dict[str, tuple[str, str]] = {
    "drift": ("Drift", "Illumination Offset"),
    "rotation": ("Rotation", "Rotation Angle (Degrees)"),
    "gamma": ("Gamma", "Gamma"),
    "contrast": ("Contrast", "Contrast Factor"),
    "blur": ("Blur", "Gaussian Sigma"),
    "gau_noise": ("Gaussian Noise", "Gaussian Sigma"),
    "poi_noise": ("Poisson Noise", "Severity"),
    "spepper_noise": ("Salt and Pepper", "Severity"),
    "bit_depth": ("Bit Depth", "Bit Depth (bits)"),
    "resolution": ("Resolution", "Resolution (px)"),
}
BASELINE_LEVELS: dict[str, float] = {
    "drift": 0.0,
    "rotation": 0.0,
    "gamma": 1.0,
    "contrast": 1.0,
    "blur": 0.0,
    "gau_noise": 0.0,
    "poi_noise": 0.0,
    "spepper_noise": 0.0,
    "bit_depth": 8.0,
    "resolution": 2048.0,
}
SERIES_ORDER: typing.Sequence[str] = (
    "acc_h0",
    "acc_h1",
    "acc_hs",
    "acc_combined",
    "acc_hu",
)
SERIES_STYLES: dict[str, dict[str, typing.Any]] = {
    "acc_h0": {"label": "H0", "color": "#cc0000"},
    "acc_h1": {"label": "H1", "color": "#1c5fd4"},
    "acc_hs": {"label": "HS", "color": "#f28c28"},
    "acc_combined": {"label": "H0H1HS", "color": "#2e7d32"},
    "acc_hu": {"label": "Hu Moments", "color": "#7f7f7f"},
}
LINESTYLE_MAP: dict[str, typing.Any] = {
    "acc_h0": "-",
    "acc_h1": "-",
    "acc_hs": "-",
    "acc_combined": "-",
    "acc_hu": (0, (2, 2)),
}

PANEL_CONFIG: dict[str, tuple[str, typing.Sequence[str]]] = {
    "Mechanical": (
        "plot_panel_mechanical.png",
        ("rotation", "blur", "resolution"),
    ),
    "Radiometric": (
        "plot_panel_radiometric.png",
        ("drift", "gamma", "contrast"),
    ),
    "Failure": (
        "plot_panel_failure.png",
        ("gau_noise", "spepper_noise", "poi_noise"),
    ),
}

ORTHO_PROTOCOLS: typing.Sequence[str] = ("rotation", "drift")
ORTHO_KEYS: typing.Sequence[str] = (
    "acc_h0",
    "acc_h1",
    "acc_hs",
    "acc_combined",
    "acc_hu",
)


def load_perf_log(path: pathlib.Path) -> list[PerfRecord]:
    """Load performance records from a JSONL log.

    Parameters
    ----------
    path : pathlib.Path
        JSONL log path.

    Returns
    -------
    list of dict
        Performance records.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing perf log: {path}")

    records: list[PerfRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_latest_run(records: typing.Sequence[PerfRecord]) -> str:
    """Return the latest run_id.

    Parameters
    ----------
    records : typing.Sequence[dict]
        Performance records.

    Returns
    -------
    str
        Latest run identifier.
    """
    run_ids = [record.get("run_id") for record in records if record.get("run_id")]
    if not run_ids:
        raise ValueError("Missing run_id values in perf log.")
    return max(run_ids)


def build_series(
    records: typing.Sequence[PerfRecord],
    protocol: str,
    baseline: typing.Optional[PerfRecord],
    baseline_level: typing.Optional[float],
) -> tuple[list[float], SeriesMap, list[str]]:
    """Build the accuracy series for one protocol.

    Parameters
    ----------
    records : typing.Sequence[dict]
        Performance records for a run.
    protocol : str
        Protocol name.
    baseline : dict or None
        Baseline record to inject if missing.
    baseline_level : float or None
        Baseline level to inject if missing.

    Returns
    -------
    tuple
        Levels, series map, and available keys.
    """
    proto_records = [
        record
        for record in records
        if record.get("protocol") == protocol and "level" in record
    ]
    if not proto_records:
        raise ValueError(f"Missing protocol data for {protocol}.")

    available = [key for key in SERIES_ORDER if key in proto_records[0]]
    if not available:
        raise ValueError("Missing accuracy keys in perf log records.")

    series: list[dict[str, float]] = []
    for record in proto_records:
        point = {"level": float(record["level"])}
        for key in available:
            if key not in record:
                raise ValueError(
                    "Missing {key} in perf_log; rerun 5_ablate.".format(key=key)
                )
            point[key] = float(record[key]) * 100.0
        series.append(point)

    levels_present = {row["level"] for row in series}
    if (
        baseline is not None
        and baseline_level is not None
        and baseline_level not in levels_present
    ):
        point = {"level": float(baseline_level)}
        for key in available:
            if key not in baseline:
                raise ValueError(
                    "Missing {key} in perf_log baseline; rerun 5_ablate.".format(
                        key=key
                    )
                )
            point[key] = float(baseline[key]) * 100.0
        series.append(point)

    series.sort(key=lambda row: row["level"])
    levels = [row["level"] for row in series]
    series_map = {key: [row[key] for row in series] for key in available}

    return levels, series_map, available


def set_accuracy_limits(
    ax: plt.Axes, series: typing.Sequence[list[float]]
) -> None:
    """Set y-axis limits from the data range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to update.
    series : typing.Sequence[list[float]]
        Series values for limits.

    Returns
    -------
    None
    """
    values: list[float] = []
    for values_list in series:
        values.extend(values_list)
    lower = max(0.0, min(values) - 5.0)
    upper = min(100.0, max(values) + 5.0)
    ax.set_ylim(lower, upper)


def build_bit_depth_ticks(
    levels: typing.Sequence[float],
) -> tuple[list[float], list[str]]:
    """Return tick positions and integer labels for bit depth.

    Parameters
    ----------
    levels : typing.Sequence[float]
        Bit depth levels.

    Returns
    -------
    tuple
        Tick positions and labels.
    """
    ticks: list[float] = []
    labels: list[str] = []
    for level in levels:
        if level <= 0:
            continue
        ticks.append(level)
        labels.append(str(int(round(level))))
    return ticks, labels


def thin_ticks(
    levels: typing.Sequence[float],
    labels: typing.Optional[typing.Sequence[str]] = None,
    max_ticks: int = 10,
) -> tuple[list[float], typing.Optional[list[str]]]:
    """Thin tick marks to roughly max_ticks evenly spaced values.

    Parameters
    ----------
    levels : typing.Sequence[float]
        Full list of level values.
    labels : typing.Optional[typing.Sequence[str]], optional
        Tick labels aligned with levels, if precomputed.
    max_ticks : int, optional
        Target tick count after thinning.

    Returns
    -------
    tuple
        Thinned levels and labels (if provided).
    """
    if len(levels) <= max_ticks:
        return list(levels), list(labels) if labels is not None else None
    indices = np.linspace(0, len(levels) - 1, max_ticks, dtype=int)
    indices = np.unique(indices)
    thinned_levels = [levels[int(idx)] for idx in indices]
    if labels is None:
        return thinned_levels, None
    thinned_labels = [labels[int(idx)] for idx in indices]
    return thinned_levels, thinned_labels


def plot_panel(
    filename: str,
    protocols: typing.Sequence[str],
    run_records: typing.Sequence[PerfRecord],
    baseline: typing.Optional[PerfRecord],
) -> None:
    """Plot a multi-panel figure for the specified protocols.

    Parameters
    ----------
    filename : str
        Output filename.
    protocols : typing.Sequence[str]
        Protocols to plot.
    run_records : typing.Sequence[dict]
        Performance records for a run.
    baseline : dict or None
        Baseline record for injection.

    Returns
    -------
    None
    """
    series_data: list[
        tuple[list[float], SeriesMap, list[str]]
    ] = []
    for protocol in protocols:
        series_data.append(
            build_series(
                run_records,
                protocol,
                baseline,
                BASELINE_LEVELS.get(protocol),
            )
        )

    ncols = len(protocols)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.4 * ncols * 1.1, 4.8),
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    for idx, (protocol, ax) in enumerate(zip(protocols, axes)):
        levels, series_map, keys = series_data[idx]
        for key in SERIES_ORDER:
            if key not in series_map:
                continue
            style = SERIES_STYLES.get(key, {})
            label = style.get("label", key)
            color = style.get("color")
            plot_kwargs = {
                "linewidth": 1,
                "linestyle": LINESTYLE_MAP.get(key, "-"),
                "label": label if idx == 0 else "_nolegend_",
            }
            if key == "acc_combined":
                plot_kwargs["linewidth"] = 2.5
            elif key == "acc_hu":
                plot_kwargs["linewidth"] = 1.5
            if color:
                plot_kwargs["color"] = color
            ax.plot(levels, series_map[key], **plot_kwargs)

        baseline_x = BASELINE_LEVELS.get(protocol)
        if baseline_x is not None:
            ax.axvline(
                baseline_x,
                color="#7f7f7f",
                linestyle="--",
                linewidth=2,
                label="baseline" if idx == 0 else "_nolegend_",
            )

        title, xlabel = PROTOCOL_META[protocol]
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        if idx == 0:
            ax.set_ylabel("Classification Accuracy (%)")
        else:
            ax.set_ylabel("")
        ax.yaxis.set_major_locator(MultipleLocator(5))
        if protocol == "bit_depth":
            ticks, labels = build_bit_depth_ticks(levels)
            ticks, labels = thin_ticks(ticks, labels)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ticks, _ = thin_ticks(levels)
            ax.set_xticks(ticks)
        if protocol == "resolution":
            base_size = float(plt.rcParams.get("font.size", 12))
            ax.tick_params(axis="x", labelsize=base_size * 0.75)
        if protocol in ("poi_noise", "spepper_noise"):
            base_size = float(plt.rcParams.get("font.size", 12))
            ax.tick_params(axis="x", labelsize=base_size * 0.7)
        ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.6)

    all_series: list[list[float]] = []
    for _, series_map, _ in series_data:
        all_series.extend(series_map.values())
    set_accuracy_limits(axes[0], all_series)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.06, 1.0, 0.2),
        mode="expand",
        ncol=LEGEND_NCOL,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_orthogonality(
    run_records: typing.Sequence[PerfRecord],
    baseline: typing.Optional[PerfRecord],
) -> None:
    """Plot rotation vs drift orthogonality panels.

    Parameters
    ----------
    run_records : typing.Sequence[dict]
        Performance records for a run.
    baseline : dict or None
        Baseline record for injection.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)

    for idx, protocol in enumerate(ORTHO_PROTOCOLS):
        levels, series_map, _ = build_series(
            run_records,
            protocol,
            baseline,
            BASELINE_LEVELS.get(protocol),
        )
        ax = axes[idx]
        for key in ORTHO_KEYS:
            style = SERIES_STYLES.get(key, {})
            ax.plot(
                levels,
                series_map[key],
                label=style.get("label", key) if idx == 0 else "_nolegend_",
                color=style.get("color"),
                linestyle=LINESTYLE_MAP.get(key, "-"),
                linewidth=2,
            )
        baseline_x = BASELINE_LEVELS.get(protocol)
        if baseline_x is not None:
            ax.axvline(
                baseline_x,
                color="#7f7f7f",
                linestyle="--",
                linewidth=2,
                label="baseline" if idx == 0 else "_nolegend_",
            )
        title, xlabel = PROTOCOL_META[protocol]
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        if idx == 0:
            ax.set_ylabel("Classification Accuracy (%)")
        else:
            ax.set_ylabel("")
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ticks, _ = thin_ticks(levels)
        ax.set_xticks(ticks)
        ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.6)

    all_series: list[list[float]] = []
    for protocol in ORTHO_PROTOCOLS:
        _, series_map, _ = build_series(
            run_records,
            protocol,
            baseline,
            BASELINE_LEVELS.get(protocol),
        )
        for key in ORTHO_KEYS:
            all_series.append(series_map[key])
    set_accuracy_limits(axes[0], all_series)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.08, 1.0, 0.2),
        mode="expand",
        ncol=ORTHO_LEGEND_NCOL,
        frameon=False,
        fontsize=ORTHO_LEGEND_FONT_SIZE,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / "plot_orthogonality.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def plot_single_protocol(
    protocol: str,
    run_records: typing.Sequence[PerfRecord],
    baseline: typing.Optional[PerfRecord],
) -> None:
    """Plot a single protocol series with legend.

    Parameters
    ----------
    protocol : str
        Protocol name.
    run_records : typing.Sequence[dict]
        Performance records for a run.
    baseline : dict or None
        Baseline record for injection.

    Returns
    -------
    None
    """
    levels, series_map, _ = build_series(
        run_records,
        protocol,
        baseline,
        BASELINE_LEVELS.get(protocol),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in SERIES_ORDER:
        if key not in series_map:
            continue
        style = SERIES_STYLES.get(key, {})
        label = style.get("label", key)
        color = style.get("color")
        linewidth = 1.4
        if key == "acc_combined":
            linewidth = 2.5
        elif key == "acc_hu":
            linewidth = 1.6
        ax.plot(
            levels,
            series_map[key],
            label=label,
            color=color,
            linestyle=LINESTYLE_MAP.get(key, "-"),
            linewidth=linewidth,
        )

    baseline_x = BASELINE_LEVELS.get(protocol)
    if baseline_x is not None:
        ax.axvline(
            baseline_x,
            color="#7f7f7f",
            linestyle="--",
            linewidth=2,
            label="baseline",
        )

    title, xlabel = PROTOCOL_META[protocol]
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Classification Accuracy (%)")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    if protocol == "bit_depth":
        ticks, labels = build_bit_depth_ticks(levels)
        ticks, labels = thin_ticks(ticks, labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ticks, _ = thin_ticks(levels)
        ax.set_xticks(ticks)
    if protocol == "resolution":
        base_size = float(plt.rcParams.get("font.size", 12))
        ax.tick_params(axis="x", labelsize=base_size * 0.75)
    if protocol in ("poi_noise", "spepper_noise"):
        base_size = float(plt.rcParams.get("font.size", 12))
        ax.tick_params(axis="x", labelsize=base_size * 0.7)
    ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.6)

    set_accuracy_limits(ax, list(series_map.values()))

    handles, labels = ax.get_legend_handles_labels()
    if protocol == "bit_depth":
        ax.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.05),
            ncol=3,
            frameon=True,
            fontsize=BASE_FONT_SIZE,
        )
    else:
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.02, 1.0, 0.2),
            mode="expand",
            ncol=LEGEND_NCOL,
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"plot_single_{protocol}.png"
    if protocol == "bit_depth":
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate grouped panels, single-protocol plots, and orthogonality plot."""
    fs.seed_everything()
    records = load_perf_log(fs.PERF_LOG_PATH)
    step_records = [record for record in records if record.get("step") == "5_ablate"]
    if not step_records:
        raise ValueError("Missing 5_ablate records in perf log.")

    run_id = select_latest_run(step_records)
    run_records = [record for record in step_records if record.get("run_id") == run_id]
    baseline = next(
        (record for record in run_records if record.get("protocol") == "baseline"),
        None,
    )

    for _, (filename, protocols) in PANEL_CONFIG.items():
        plot_panel(filename, protocols, run_records, baseline)

    plot_orthogonality(run_records, baseline)
    for protocol in PROTOCOL_META:
        plot_single_protocol(protocol, run_records, baseline)


if __name__ == "__main__":
    main()
