#!/usr/bin/env python3
"""Simulation for the Hodgkin-Huxley Sensitivity Analysis

Reproduces:

"Uncertainty Propagation in Nerve Impulses Through the Action Potential Mechanism", Valderrama et al.
https://doi.org/10.1186/2190-8567-5-3

with details from:

"Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience",
Simen Tennøe et al.
https://doi.org/10.3389/fninf.2018.00049

Sebastian Schmitt, 2021
"""

import argparse
import numpy as np
import arbor

class Sensitivity(arbor.recipe):
    def __init__(self, probes, parameters):
        """Initialize the recipe

        probes -- probes
        parameters -- list like array, one entry per cell

        The entries per cell are the parameters:
            * initial membrane voltage
            * membrane capacity
            * leak potential
            * maximal Na conductance
            * maximal K conductance
            * leak conductance
            * Na reversal potential
            * K reversal potential
        """

        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)

        self.the_probes = probes
        self.the_props = arbor.neuron_cable_properties()
        self.the_cat = arbor.default_catalogue()
        self.the_props.register(self.the_cat)

        self.parameters = parameters

    def num_cells(self):
        """Number of cells. One per parameter set."""
        return len(self.parameters)

    def num_sources(self, gid):
        assert gid < len(self.parameters)

        return 0

    def cell_kind(self, gid):
        assert gid < len(self.parameters)

        return arbor.cell_kind.cable

    def cell_description(self, gid):
        assert gid < len(self.parameters)

        tree = arbor.segment_tree()

        # 1 um
        radius = 1
        tree.append(arbor.mnpos,
                    arbor.mpoint(-radius, 0, 0, radius),
                    arbor.mpoint(radius, 0, 0, radius),
                    tag=1)

        labels = arbor.label_dict({'soma':   '(tag 1)',
                                   'midpoint': '(location 0 0.5)'})

        parameters = self.parameters[gid]
        v0, cm, el, gnabar, gkbar, gl, ena, ek = parameters

        decor = arbor.decor()
        decor.set_property(Vm=v0, cm=cm)
        decor.set_ion("na", rev_pot=ena)
        decor.set_ion("k", rev_pot=ek)

        decor.paint('"soma"', arbor.mechanism('hh', {'gnabar' : gnabar,
                                                     'gkbar' : gkbar,
                                                     'gl' : gl,
                                                     'el' : el
                                                     }))

        # convert 140 uA/cm^2 to total current in nA
        area = 4 * np.pi * (radius * 1e-6)**2
        I = (140e-6/0.01**2 * area)/1e-9
        decor.place('"midpoint"', arbor.iclamp(0, 15, I))

        cell = arbor.cable_cell(tree, labels, decor)

        return cell

    def probes(self, gid):
        assert gid < len(self.parameters)

        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hodgkin-Huxley Sensitivity Analysis')

    parser.add_argument('--parameter_file', help="name of parameter file", default="parameters_hh.npy")

    # parse the command line arguments
    args = parser.parse_args()

    # set up probes
    membrane_probes = [arbor.cable_probe_membrane_voltage('"midpoint"')]
    state_probes = [arbor.cable_probe_density_state('"midpoint"', "hh", "m"),
                    arbor.cable_probe_density_state('"midpoint"', "hh", "n"),
                    arbor.cable_probe_density_state('"midpoint"', "hh", "h")]

    # load parameters and instantiate recipe
    parameters = np.load(args.parameter_file)
    recipe = Sensitivity(membrane_probes + state_probes, parameters)

    # create a default execution context and a default domain decomposition
    context = arbor.context()
    domains = arbor.partition_load_balance(recipe, context)

    # configure the simulation and handles for the probes
    sim = arbor.simulation(recipe, domains, context)

    # time step for simulation and sampling in ms
    dt = 0.01

    membrane_handles = [sim.sample((i, 0), arbor.regular_schedule(dt)) for i in range(recipe.num_cells())]
    m_handles = [sim.sample((i, 1), arbor.regular_schedule(dt)) for i in range(recipe.num_cells())]
    n_handles = [sim.sample((i, 2), arbor.regular_schedule(dt)) for i in range(recipe.num_cells())]
    h_handles = [sim.sample((i, 3), arbor.regular_schedule(dt)) for i in range(recipe.num_cells())]

    # run the simulation for 30 ms
    sim.run(tfinal=30, dt=dt)

    # store results
    np.save("hh_sensitivity_time.npy", np.array(sim.samples(membrane_handles[0])[0][0][:,0]))
    np.save("hh_sensitivity_membrane.npy", np.array([sim.samples(handle)[0][0][:,1] for handle in membrane_handles]))

    np.save("hh_sensitivity_m.npy", np.array([sim.samples(handle)[0][0][:,1] for handle in m_handles]))
    np.save("hh_sensitivity_n.npy", np.array([sim.samples(handle)[0][0][:,1] for handle in n_handles]))
    np.save("hh_sensitivity_h.npy", np.array([sim.samples(handle)[0][0][:,1] for handle in h_handles]))
