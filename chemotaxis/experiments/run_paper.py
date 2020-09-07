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


# plots
from cell.plots.metabolism import plot_exchanges

def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def add_arguments():
    parser = argparse.ArgumentParser(description='chemotaxis control')
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default=None,
        help='preconfigured experiments')
    return parser.parse_args()


def figure_1a(out_dir):
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

    import ipdb;
    ipdb.set_trace()


def main():
    """
    Execute experiments from the paper
    """

    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'chemotaxis')
    make_dir(out_dir)

    args = add_arguments()
    if args.experiment:
        # get a preset experiment
        # make a directory for this experiment
        experiment_name = str(args.experiment)
        control_out_dir = os.path.join(out_dir, experiment_name)
        make_dir(control_out_dir)

        if experiment_name == '1':
            figure_1a(out_dir)

    else:
        print('provide figure number')


if __name__ == '__main__':
    main()
