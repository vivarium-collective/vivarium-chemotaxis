"""
====================
Experiment Workflow
====================

Handles experiment specifications for run_papers.py
"""

import os
import argparse

# directories
from cell.plots.multibody_physics import plot_tags, plot_snapshots
from vivarium.core.composition import (
    plot_agents_multigen,
    plot_compartment_topology,
)

from chemotaxis import EXPERIMENT_OUT_DIR



def plot_control(data, config, out_dir='out'):
    environment_config = config.get('environment_config')
    emit_fields = config.get('emit_fields')
    tagged_molecules = config.get('tagged_molecules')
    topology_network = config.get('topology_network')

    # extract data
    multibody_config = environment_config['config']['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    plot_data = {
        'agents': agents,
        'fields': fields,
        'config': multibody_config,
    }

    # multigen plot
    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

    # tag plot
    if tagged_molecules:
        plot_config = {
            'tagged_molecules': tagged_molecules,
            'n_snapshots': 5,
            'out_dir': out_dir}
        plot_tags(plot_data, plot_config)

    # snapshot plot
    if emit_fields:
        plot_config = {
            'fields': emit_fields,
            'n_snapshots': 5,
            'out_dir': out_dir}
        plot_snapshots(plot_data, plot_config)

    # topology network
    if topology_network:
        compartment = topology_network['compartment']
        settings = {'show_ports': True}
        plot_compartment_topology(
            compartment,
            settings,
            out_dir)


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def add_arguments(experiments_library):
    parser = argparse.ArgumentParser(description='chemotaxis paper experiments')
    parser.add_argument(
        'experiment_id',
        type=str,
        choices=list(experiments_library.keys()),
        help='experiment id corresponds to figure number from chemotaxis paper')
    return parser.parse_args()


def control(experiments_library):
    """
    Execute experiments from the command line
    """

    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'chemotaxis')
    make_dir(out_dir)

    args = add_arguments(experiments_library)

    if args.experiment_id:
        # retrieve preset experiment
        experiment_id = str(args.experiment_id)
        experiment_type = experiments_library[experiment_id]

        if callable(experiment_type):
            control_out_dir = os.path.join(out_dir, experiment_id)
            make_dir(control_out_dir)
            experiment_type(control_out_dir)
        elif isinstance(experiment_type, list):
            # iterate over list with multiple experiments
            for sub_experiment_id in experiment_type:
                control_out_dir = os.path.join(out_dir, sub_experiment_id)
                make_dir(control_out_dir)
                exp = experiments_library[sub_experiment_id]
                try:
                    exp(control_out_dir)
                except:
                    print('{} experiment failed'.format(sub_experiment_id))
    else:
        print('provide experiment number')
