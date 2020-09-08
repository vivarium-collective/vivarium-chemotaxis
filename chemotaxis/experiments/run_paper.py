'''
====================
Chemotaxis Experiments
====================

Chemotaxis provides several pre-configured :py:class:`Experiments`
with different chemotactic agents and environments.
'''

import os
import argparse

from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    plot_simulation_output,
    EXPERIMENT_OUT_DIR,
)


from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config,
)
from chemotaxis.processes.flagella_motor import run_variable_flagella
from chemotaxis.processes.chemoreceptor_cluster import (
    test_receptor,
    get_pulse_timeline,
)
from chemotaxis.compartments.chemotaxis_flagella import test_variable_chemotaxis
from chemotaxis.compartments.flagella_expression import make_flagella_network

# plots
from cell.plots.metabolism import plot_exchanges
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output








def BiGG_metabolism(out_dir='out'):
    # configure BiGG metabolism
    config = get_iAF1260b_config()
    metabolism = Metabolism(config)

    # simulation settings
    sim_settings = {
        'environment': {
            'volume': 1e-5 * units.L,
        },
        'total_time': 2520,  # 2520 sec (42 min) is the expected doubling time in minimal media
    }

    # run simulation
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot settings
    plot_settings = {
        'max_rows': 30,
        'remove_zeros': True,
        'skip_ports': ['exchange', 'reactions']}

    # make plots from simulation output
    plot_simulation_output(timeseries, plot_settings, out_dir, 'BiGG_simulation')
    plot_exchanges(timeseries, sim_settings, out_dir)


    import ipdb; ipdb.set_trace()


# figure 6a
def flagella_expression_network(out_dir='out'):
    make_flagella_network(out_dir)

# figure 6b

# figure 6c

# figure 7a
def variable_flagella(out_dir='out'):
    run_variable_flagella(out_dir)

# figure 7b
def run_chemoreceptor_pulse(out_dir='out'):
    timeline = get_pulse_timeline()
    timeseries = test_receptor(timeline)
    plot_receptor_output(timeseries, out_dir, 'pulse')

# figure 7c
def run_chemotaxis_transduction(out_dir='out'):
    test_variable_chemotaxis(
        out_dir=out_dir
    )


experiments_library = {
    '4a': BiGG_metabolism,
    '6a': flagella_expression_network,
    '7a': variable_flagella,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
}

def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def add_arguments():
    parser = argparse.ArgumentParser(description='chemotaxis control')
    parser.add_argument(
        'experiment_id',
        type=str,
        default=None,
        help='experiment number')
    return parser.parse_args()

def main():
    """
    Execute experiments
    """

    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'chemotaxis')
    make_dir(out_dir)

    args = add_arguments()

    if args.experiment_id:
        # get a preset experiment
        # make a directory for this experiment
        experiment_id = str(args.experiment_id)
        control_out_dir = os.path.join(out_dir, experiment_id)
        make_dir(control_out_dir)

        run_function = experiments_library[experiment_id]
        run_function(control_out_dir)

    else:
        print('provide experiment number')


if __name__ == '__main__':
    main()
