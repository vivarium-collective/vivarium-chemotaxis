"""
====================
Experiment Control
====================

Handles experiment specifications for `paper_experiments.py`
"""

import os
import argparse
import numpy as np
import math

# vivarium-core imports
from vivarium.library.units import units

# directories
from vivarium_cell.plots.multibody_physics import plot_tags, plot_snapshots
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.plots.agents_multigen import plot_agents_multigen

from chemotaxis import EXPERIMENT_OUT_DIR

PI = math.pi


def single_agent_config(config):
    width = 1
    length = 2
    # volume = volume_from_length(length, width)
    bounds = config.get('bounds')
    location = config.get('location')
    location = [loc * bounds[n] for n, loc in enumerate(location)]

    return {
        'boundary': {
            'location': location,
            'angle': np.random.uniform(0, 2 * PI),
            # 'volume': volume,
            'length': length,
            'width': width,
            'mass': 1339 * units.fg,
        },
        'membrane': {
            'PMF': -140.0
        }
    }

def agent_body_config(config):
    agent_ids = config['agent_ids']
    agent_config = {
        agent_id: single_agent_config(config)
        for agent_id in agent_ids}
    return {
        'agents': agent_config}


def plot_control(data, config, out_dir='out'):
    environment_config = config.get('environment_config')
    emit_fields = config.get('emit_fields')
    tagged_molecules = config.get('tagged_molecules')
    topology_network = config.get('topology_network')
    n_snapshots = config.get('n_snapshots', 5)

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
    plot_settings = {
        # 'remove_zeros': True
    }
    plot_agents_multigen(data, plot_settings, out_dir)

    # tag plot
    if tagged_molecules:
        plot_config = {
            'tagged_molecules': tagged_molecules,
            'n_snapshots': n_snapshots,
            'convert_to_concs': False,
            'out_dir': out_dir}
        plot_tags(plot_data, plot_config)

    # snapshot plot
    if emit_fields:
        plot_config = {
            'fields': emit_fields,
            'n_snapshots': n_snapshots,
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
