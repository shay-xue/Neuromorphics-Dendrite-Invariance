"""Input spike generators and AMPA conductance functions.

This file handles the input side:
when spikes happen, which segment they hit, and the AMPA current they create.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .channels import E_AMPA


@dataclass
class AMPAEvent:
    """One AMPA input spike onto one dendrite segment."""

    segment: int
    time: float
    weight: float = 1.0


def alpha_conductance(
    t: float,
    spike_time: float,
    weight: float = 1.0,
    tau: float = 3.0,
) -> float:
    """AMPA alpha-function conductance from one spike."""
    x = t - spike_time

    if x < 0:
        return 0.0

    return weight * (x / tau) * np.exp(1 - x / tau)


def g_AMPA_vector(
    t: float,
    n_segments: int,
    ampa_events: list[dict] | list[AMPAEvent],
    tau: float = 3.0,
) -> np.ndarray:
    """Total AMPA conductance at each segment at time t."""
    g = np.zeros(n_segments)

    for event in ampa_events:
        if isinstance(event, AMPAEvent):
            segment = event.segment
            spike_time = event.time
            weight = event.weight
        else:
            segment = int(event["segment"])
            spike_time = float(event["time"])
            weight = float(event.get("weight", 1.0))

        g[segment] += alpha_conductance(
            t=t,
            spike_time=spike_time,
            weight=weight,
            tau=tau,
        )

    return g


def I_AMPA(v: np.ndarray, g_ampa: np.ndarray) -> np.ndarray:
    """AMPA synaptic current for each dendrite segment."""
    return g_ampa * (E_AMPA - v)

def nmda_conductance(
    t,
    spike_time,
    weight=1.0,
    tau_rise=1.4,
    tau_decay_fast=12.6,
    tau_decay_mid=57.3,
    tau_decay_slow=372.0,
):
    """NMDA conductance kernel from one spike.

    Important:
    Do NOT normalize by max(raw) inside this function when it is called
    one timestep at a time. That turns every positive scalar value into 1.
    """
    dt = np.asarray(t) - spike_time

    # scalar-safe output
    g = np.zeros_like(dt, dtype=float)
    mask = dt >= 0

    x = dt[mask]

    raw = (
        0.5 * np.exp(-x / tau_decay_fast)
        + 1.0 * np.exp(-x / tau_decay_mid)
        + 1.0 * np.exp(-x / tau_decay_slow)
        - 2.5 * np.exp(-x / tau_rise)
    )

    raw = np.maximum(raw, 0.0)

    # Fixed normalization constant
    # With these coefficients, the peak is around 0.6.
    NMDA_KERNEL_PEAK = 2.1694
    raw = raw / NMDA_KERNEL_PEAK

    g[mask] = weight * raw

    # Return a scalar if t was scalar, otherwise return array
    if np.ndim(t) == 0:
        return float(g)

    return g

def g_NMDA_vector(t, n_segments, nmda_events):
    g_nmda = np.zeros(n_segments)

    for event in nmda_events:
        seg = int(event["segment"])
        spike_time = float(event["time"])
        weight = float(event.get("weight", 1.0))

        g_nmda[seg] += nmda_conductance(
            t,
            spike_time,
            weight=weight,
        )

    return g_nmda


def make_sleep_like_sequence(
    sequence_segments: list[int],
    start_time: float = 10.0,
    step_ms: float = 5.0,
    weight: float = 1.0,
) -> tuple[list[dict], dict[int, list[float]]]:
    """Make a one-spike-per-segment sequence."""
    events = []
    raster: dict[int, list[float]] = {}

    for i, segment in enumerate(sequence_segments):
        spike_time = start_time + i * step_ms

        events.append(
            {
                "segment": int(segment),
                "time": float(spike_time),
                "weight": float(weight),
            }
        )

        raster.setdefault(int(segment), []).append(float(spike_time))

    return events, raster


def make_burst_sequence(
    sequence_segments: list[int],
    start_time: float = 10.0,
    onset_step_ms: float = 5.0,
    burst_len: int = 3,
    isi_ms: float = 2.0,
    weight: float = 1.0,
) -> tuple[list[dict], dict[int, list[float]]]:
    """Make a burst version of the same sequence.

    onset_step_ms controls when the next segment starts.
    Smaller onset_step_ms means more overlap.
    """
    events = []
    raster: dict[int, list[float]] = {}

    for i, segment in enumerate(sequence_segments):
        burst_start = start_time + i * onset_step_ms

        for k in range(burst_len):
            spike_time = burst_start + k * isi_ms

            events.append(
                {
                    "segment": int(segment),
                    "time": float(spike_time),
                    "weight": float(weight),
                    "burst_index": int(k),
                    "sequence_index": int(i),
                }
            )

            raster.setdefault(int(segment), []).append(float(spike_time))

    return events, raster


def burst_overlap_duration(
    onset_step_ms: float,
    burst_len: int,
    isi_ms: float,
) -> float:
    """Compute overlap time between neighboring bursts."""
    burst_duration = (burst_len - 1) * isi_ms
    overlap = burst_duration - onset_step_ms

    return max(0.0, float(overlap))


def first_spike_filter(
    ampa_events: list[dict] | list[AMPAEvent],
) -> list[dict]:
    """Keep only the first spike from each segment."""
    first_events: dict[int, dict] = {}

    for event in ampa_events:
        if isinstance(event, AMPAEvent):
            event_dict = {
                "segment": int(event.segment),
                "time": float(event.time),
                "weight": float(event.weight),
            }
        else:
            event_dict = dict(event)

        segment = int(event_dict["segment"])
        spike_time = float(event_dict["time"])

        if segment not in first_events:
            first_events[segment] = event_dict
        elif spike_time < float(first_events[segment]["time"]):
            first_events[segment] = event_dict

    return [first_events[seg] for seg in sorted(first_events.keys())]
