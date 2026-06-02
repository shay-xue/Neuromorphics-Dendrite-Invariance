"""Dendrite simulator and plotting helpers.

This file combines:
1. channel currents from channels.py
2. spike inputs from inputs.py
3. voltage simulation + scoring/plots
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .channels import I_axial, I_channels
from .inputs import I_AMPA, g_AMPA_vector


def simulate_dendrite(
    n_segments: int = 12,
    T: float = 80.0,
    dt: float = 0.02,
    G_NMDA: float = 20.0,
    G_KIR: float = 20.0,
    G_ax: float = 1.0,
    ampa_events: list[dict] | None = None,
    plateau_segments: int = 5,
    v_rest: float = -1.0,
    v_plateau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the dendrite simulation.

    The dendrite starts with a plateau at the tip. Then AMPA inputs
    arrive at different segments and we check whether the plateau
    propagates through the sequence.
    """
    if ampa_events is None:
        ampa_events = []

    times = np.arange(0, T + dt, dt)
    V_hist = np.zeros((len(times), n_segments))

    # Start at rest, then set the dendrite tip to plateau.
    v = np.ones(n_segments) * v_rest
    v[:plateau_segments] = v_plateau
    V_hist[0] = v

    for idx, t in enumerate(times[1:], start=1):
        # Current from voltage-dependent channels
        intrinsic = I_channels(v, G_NMDA=G_NMDA, G_KIR=G_KIR)

        # Current between neighboring segments
        axial = I_axial(v, G_ax=G_ax)

        # Current from incoming spikes
        g_ampa = g_AMPA_vector(t, n_segments, ampa_events)
        synaptic = I_AMPA(v, g_ampa)

        # Euler update
        dvdt = intrinsic + axial + synaptic
        v = v + dt * dvdt

        # Prevent numerical blow-up
        v = np.clip(v, -1.5, 1.5)

        V_hist[idx] = v

    return times, V_hist


def activation_times(
    times: np.ndarray,
    V_hist: np.ndarray,
    segments: list[int],
    threshold: float = 0.0,
) -> dict[int, float]:
    """Find the first time each segment crosses threshold."""
    out: dict[int, float] = {}

    for seg in segments:
        crossed = np.where(V_hist[:, seg] >= threshold)[0]
        out[int(seg)] = float(times[crossed[0]]) if len(crossed) else np.nan

    return out


def score_detection(
    times: np.ndarray,
    V_hist: np.ndarray,
    sequence_segments: list[int],
    threshold: float = 0.0,
    final_window_ms: float = 5.0,
) -> dict:
    """Score whether the dendrite detected the sequence.

    A successful detection means:
    1. every sequence segment crossed threshold,
    2. the crossings happened in the right order,
    3. the final segment stayed high near the end.
    """
    act = activation_times(times, V_hist, sequence_segments, threshold=threshold)
    act_values = [act[int(seg)] for seg in sequence_segments]

    all_detected = all(np.isfinite(x) for x in act_values)
    ordered = all_detected and all(x <= y for x, y in zip(act_values[:-1], act_values[1:]))

    final_seg = int(sequence_segments[-1])
    final_mask = times >= (times[-1] - final_window_ms)
    final_voltage = float(np.mean(V_hist[final_mask, final_seg]))
    final_high = final_voltage >= threshold

    success = bool(ordered and final_high)

    return {
        "success": success,
        "all_detected": bool(all_detected),
        "ordered": bool(ordered),
        "final_high": bool(final_high),
        "final_voltage": final_voltage,
        "activation_times": act,
    }


def print_score(score: dict) -> None:
    """Print the detection score in a readable way."""
    print("success:", score["success"])
    print("all_detected:", score["all_detected"])
    print("ordered:", score["ordered"])
    print("final_high:", score["final_high"])
    print("final_voltage:", round(score["final_voltage"], 3))
    print("activation_times:", score["activation_times"])


def plot_input_raster(raster: dict[int, list[float]], title: str) -> None:
    """Plot input spike times for each segment."""
    plt.figure(figsize=(9, 2.8))

    sorted_segments = sorted(raster.keys())
    for y, seg in enumerate(sorted_segments):
        plt.vlines(raster[seg], y - 0.35, y + 0.35)

    plt.yticks(range(len(sorted_segments)), [f"seg {seg}" for seg in sorted_segments])
    plt.xlabel("time (ms)")
    plt.ylabel("input segment")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_voltage_heatmap(times: np.ndarray, V_hist: np.ndarray, title: str) -> None:
    """Plot voltage across all dendrite segments over time."""
    plt.figure(figsize=(10, 4.5))

    plt.imshow(
        V_hist.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, V_hist.shape[1] - 1],
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(label="voltage")
    plt.xlabel("time (ms)")
    plt.ylabel("dendrite segment")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_voltage_traces(
    times: np.ndarray,
    V_hist: np.ndarray,
    segments_to_plot: list[int],
    title: str,
) -> None:
    """Plot voltage traces for selected segments."""
    plt.figure(figsize=(10, 4.5))

    for seg in segments_to_plot:
        plt.plot(times, V_hist[:, seg], label=f"segment {seg}")

    plt.axhline(0, linestyle="--", linewidth=1, label="threshold = 0")
    plt.xlabel("time (ms)")
    plt.ylabel("voltage")
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_ampa_conductances(
    times: np.ndarray,
    n_segments: int,
    ampa_events: list[dict],
    title: str,
) -> None:
    """Plot AMPA input conductance for the stimulated segments."""
    plt.figure(figsize=(10, 4))

    segments = sorted(set(int(event["segment"]) for event in ampa_events))

    for seg in segments:
        g_trace = [g_AMPA_vector(float(t), n_segments, ampa_events)[seg] for t in times]
        plt.plot(times, g_trace, label=f"segment {seg}")

    plt.xlabel("time (ms)")
    plt.ylabel("AMPA conductance")
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()
