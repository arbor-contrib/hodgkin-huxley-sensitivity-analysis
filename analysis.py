#!/usr/bin/env python3
"""Sensitivity Analysis on a Hodgkin-Huxley Neuron

Reproduces:

"Uncertainty Propagation in Nerve Impulses Through the Action Potential Mechanism", Valderrama et al.
https://doi.org/10.1186/2190-8567-5-3

with details from:

"Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience",
Simen Tennøe et al.
https://doi.org/10.3389/fninf.2018.00049

Sebastian Schmitt, 2021
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol


def plot(problem, x, y, S1s):
    """Plot HH and Sobol indicies."""

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    ax0 = fig.add_subplot(gs[:, :3])
    ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')
    ax0.set_xlim(5, 15)
    ax0.set_ylim(-65, 10)

    vertical_line_positions = [8.315, 10.4]

    for vlp in vertical_line_positions:
        ax0.axvline(vlp, linestyle="dashed", color="black")

    sobol_axes = [fig.add_subplot(gs[i//3, 3 + i%3], sharex=ax0) for i in range(len(problem["names"]) + 1)]

    for i, ax in enumerate(sobol_axes[:-1]):
        ax.plot(x, S1s[:, i], color='black')
        ax.set_title(problem["pretty_names"][i])

    for i, ax in enumerate(sobol_axes):

        ax.set_xlabel("t (ms)")
        ax.set_ylim(0, 1.09)

        for vlp in vertical_line_positions:
            ax.axvline(vlp, linestyle="dashed", color="black")

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    sobol_axes[-1].plot(x, np.nansum(S1s, axis=1), color="black")
    sobol_axes[-1].set_title("Sum")
    sobol_axes[-1].set_ylabel("First-order Sobol index")

    # in percent
    prediction_interval = 90

    ax0.fill_between(x,
                     np.percentile(y, 50 - prediction_interval/2., axis=0),
                     np.percentile(y, 50 + prediction_interval/2., axis=0),
                     alpha=0.5, color='black',
                     label=f"{prediction_interval} % prediction interval")


    ax0_2 = ax0.twinx()
    ax0_2.plot(x, np.std(y, axis=0), linestyle="-.", color="black", label="Standard deviation")
    ax0_2.set_ylabel("Standard deviation (mV)")

    ax0.set_xlabel("t (ms)")
    ax0.set_ylabel("Membrane potential (mV)")

    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_2.get_legend_handles_labels()
    ax0_2.legend(lines + lines2, labels + labels2,
                 loc='upper right')._legend_box.align = "left"
    ax0.set_title("Membrane potential")

    return fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sensitivity analysis of a parabola')

    parser.add_argument('--show', help="show plot",
                        action="store_true", default=False)
    parser.add_argument('--save', help="save to given file name")

    parser.add_argument('--sample', help="file name for sampled parameters")

    args = parser.parse_args()

    # Cf. uncertainpy: examples/valderrama/valderrama.py
    # relative uncertainty has to be applied to unshifted potentials
    v0 = -10
    cm = 0.01
    gnabar = 0.12
    gkbar = 0.036
    gl = 0.0003
    ena = 112
    ek = -12
    el = 10.613

    # relative uncertainty on parameters
    interval = 0.2
    shift = -65

    problem = {
        'num_vars': 8,

        'pretty_names' : ["$V_0$", "$C_m$", r"$E_\mathregular{l}$",
                          r"$\bar{g}_\mathregular{Na}$", r"$\bar{g}_\mathregular{K}$", r"$g_\mathregular{l}$",
                          r"$E_\mathregular{Na}$", r"$E_\mathregular{K}$"],

        'names': ['V0', 'Cm', 'el',
                  'gnabar', 'gkbar', "gl",
                  "ena", "ek"],

        'bounds': [[v0*(1+interval) + shift, v0*(1-interval) + shift],
                   [cm*(1-interval), cm*(1+interval)],
                   [el*(1-interval) + shift, el*(1+interval) + shift],
                   [gnabar*(1-interval), gnabar*(1+interval)],
                   [gkbar*(1-interval), gkbar*(1+interval)],
                   [gl*(1-interval), gl*(1+interval)],
                   [ena*(1-interval) + shift, ena*(1+interval) + shift],
                   [ek*(1+interval) + shift, ek*(1-interval) + shift],
        ]
    }

    if args.sample:
        print(f"Sampling parameter space and saving to {args.sample}")
        np.save(args.sample, saltelli.sample(problem, 2**8, calc_second_order=False))
        sys.exit(0)

    if not args.show and not args.save:
        print("Neither --show nor --save selected, "
              "simulation will run but no output will be produced.")

    membrane = np.load("hh_sensitivity_membrane.npy")
    time = np.load("hh_sensitivity_time.npy")

    sobol_indices = [sobol.analyze(problem, y, calc_second_order=False) for y in membrane.T]

    S1s = np.array([s['S1'] for s in sobol_indices])

    if np.isnan(S1s).any():
        print("Warning NaN in Sobol indices")

    fig = plot(problem, time, membrane, S1s)

    if args.save:
        fig.savefig(args.save)

    if args.show:
        plt.show()
