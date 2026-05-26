# Neuromorphics-Dendrite-Invariance
Can a dendrite built for sleep handle being awake? We test whether burst-induced spike overlap breaks sequence detection in the HW2 dendrite model, and whether three fixes — swap-tolerance, a first-spike rule, or scaling down NMDA conductance — can make it robust to both sleep-like and wake-like firing.

---

## File Structure

```
Neuromorphics-Dendrite-Invariance/
│
├── src/
│   ├── __init__.py               # Exports all public functions from the three modules
│   ├── channels.py               # Channel models: sigma_NMDA, sigma_KIR, I_channels, I_axial;
│   │                             #   all gating parameters and reversal potentials live here
│   ├── inputs.py                 # Input generators: alpha-function AMPA conductance,
│   │                             #   single-spike event builders, burst event builders
│   │                             #   (non-overlapping and overlapping)
│   └── dendrite.py               # Core dendrite simulator (simulate_dendrite) and
│                                 #   plotting helpers (voltage surface, traces, AMPA traces)
│
├── notebooks/
│   ├── 00_baseline.ipynb         # Base code from our homework
│   │                             #   (sleep-like) sequence detection as the control condition
│   ├── 01_burst_regimes.ipynb    # Compare the three input regimes: single-spike, non-overlapping
│   │                             #   bursts, and overlapping bursts; identify the failure mode
│   ├── 02_conductance_dynamics.ipynb  # Track NMDA vs AMPA conductance accumulation across
│   │                                  #   spikes in a burst; explain mechanistically why overlap fails
│   ├── 03_swap_tolerance.ipynb   # Sweep NMDA/KIR ratio to find the swap-tolerant regime;
│   │                             #   generate the phase diagram (overlap duration x conductance ratio)
│   ├── 04_first_spike_rule.ipynb # Test upstream first-spike filtering as a fix;
│   │                             #   vary overlap and filtering threshold
│   └── 05_nmda_scaling.ipynb     # Test uniform NMDA conductance scaling as a fix;
│                                 #   vary overlap and NMDA magnitude
│
├── data/                         # Saved simulation outputs (.npy / .pkl) from long parameter
│                                 #   sweeps; load these to replot without re-running simulations
│
├── HW2.ipynb                     # Original homework — reference only, do not modify
├── Neuromorphics Project Proposal.pdf
├── Slides and Readings/          # Course lecture slides and assigned readings by week
└── README.md
```
