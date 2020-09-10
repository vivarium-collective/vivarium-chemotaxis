'''
====================
Chemotaxis Experiments
====================

Chemotaxis provides several pre-configured :py:class:`Experiments`
with different chemotactic agents and environments.
'''

import os
import argparse

# vivarium-core imports
from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    simulate_compartment_in_experiment,
    agent_environment_experiment,
    plot_simulation_output,
    plot_agents_multigen,
    EXPERIMENT_OUT_DIR,
)

# vivarium-cell imports
from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config,
)
from cell.processes.transcription import UNBOUND_RNAP_KEY
from cell.processes.translation import UNBOUND_RIBOSOME_KEY
from cell.compartments.lattice import Lattice
from cell.experiments.lattice_experiment import get_iAF1260b_environment

# chemotaxis imports
from chemotaxis.processes.flagella_motor import run_variable_flagella
from chemotaxis.processes.chemoreceptor_cluster import (
    test_receptor,
    get_pulse_timeline,
)
from chemotaxis.composites.chemotaxis_flagella import (
    test_variable_chemotaxis,
    get_chemotaxis_timeline,
)
from chemotaxis.composites.flagella_expression import (
    FlagellaExpressionMetabolism,
    get_flagella_expression_compartment,
)

from chemotaxis.plots.chemotaxis_experiments import plot_chemotaxis_experiment

# data
from chemotaxis.data.chromosomes.flagella_chromosome import FlagellaChromosome
from cell.data.nucleotides import nucleotides
from cell.data.amino_acids import amino_acids

# plots
from cell.plots.metabolism import plot_exchanges
from cell.plots.gene_expression import (
    plot_timeseries_heatmaps,
    gene_network_plot,
)
from cell.plots.multibody_physics import (
    plot_snapshots,
    plot_tags
)
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output



# figure 5a
def BiGG_metabolism(out_dir='out'):
    # configure metabolism process iAF1260b BiGG model
    config = get_iAF1260b_config()
    metabolism = Metabolism(config)
    external_concentrations = metabolism.initial_state['external']

    # run simulation with the helper function simulate_process_in_experiment
    sim_settings = {
        'environment': {
            'volume': 1e-5 * units.L,
                'concentrations': external_concentrations
        },
        'total_time': 2500,
    }
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot
    plot_exchanges(timeseries, sim_settings, out_dir)


# figure 5b
def transport_metabolism(out_dir='out'):
    pass


# figure 5c
def transport_metabolism_environment(out_dir='out'):
    pass


# figure 6a
def flagella_expression_network(out_dir='out'):
    '''
    Make a network plot of the flagella expression processes.
    This saves an networkx plot with a default layout, along with
    node and edge list files of the network for analysis by network
    visualization software.
    '''

    # load the compartment
    flagella_compartment = get_flagella_expression_compartment({})

    # make expression network plot
    flagella_expression_network = flagella_compartment.generate()
    flagella_expression_processes = flagella_expression_network['processes']
    operons = flagella_expression_processes['transcription'].genes
    promoters = flagella_expression_processes['transcription'].templates
    complexes = flagella_expression_processes['complexation'].stoichiometry
    data = {
        'operons': operons,
        'templates': promoters,
        'complexes': complexes}
    gene_network_plot(data, out_dir)


# figure 6b
def make_flagella_expression_initial_state():
    ## make the initial state
    flagella_data = FlagellaChromosome()
    chromosome_config = flagella_data.chromosome_config

    molecules = {}
    for nucleotide in nucleotides.values():
        molecules[nucleotide] = 5000000
    for amino_acid in amino_acids.values():
        molecules[amino_acid] = 1000000

    return {
        'molecules': molecules,
        'transcripts': {
            gene: 0
            for gene in chromosome_config['genes'].keys()
        },
        'proteins': {
            'CpxR': 10,
            'CRP': 10,
            'Fnr': 10,
            'endoRNAse': 1,
            'flagella': 4,
            UNBOUND_RIBOSOME_KEY: 100,  # e. coli has ~ 20000 ribosomes
            UNBOUND_RNAP_KEY: 100
        },
        'boundary': {
            'location': [8, 8]
        }
    }


def flagella_just_in_time(out_dir='out'):

    ## make the compartment
    compartment = get_flagella_expression_compartment({})

    # get the initial state
    initial_state = make_flagella_expression_initial_state()

    ## run simulation
    settings = {
        # a cell cycle of 2520 sec is expected to express 8 flagella.
        # 2 flagella expected in approximately 630 seconds.
        'total_time': 500,
        'emit_step': 10.0,
        'verbose': True,
        'initial_state': initial_state}
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    ## plot output
    flagella_data = FlagellaChromosome()
    plot_config = {
        'name': 'flagella',
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'molecules'
        },
        'plot_ports': {
            'transcripts': list(flagella_data.chromosome_config['genes'].keys()),
            'proteins': flagella_data.complexation_monomer_ids + flagella_data.complexation_complex_ids,
            'molecules': list(nucleotides.values()) + list(amino_acids.values())
        }
    }

    plot_timeseries_heatmaps(
        timeseries,
        plot_config,
        out_dir)


# figure 6c
def run_flagella_metabolism_experiment(out_dir='out'):

    total_time = 6000
    emit_step = 10
    process_time_step = 10
    bounds = [17, 17]
    emit_fields = ['glc__D_e']

    ## make the experiment
    # configure
    agents_config = {
        'ids': ['flagella_metabolism'],
        'type': FlagellaExpressionMetabolism,
        'config': {
            'time_step': process_time_step,
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
        }}
    environment_config = {
        'type': Lattice,
        'config': get_iAF1260b_environment(
            time_step=process_time_step,
            bounds=bounds,
            depth=6000.0,
            diffusion=1e-2,
            scale_concentration=5,
            keep_fields_emit=emit_fields)}
    initial_agent_state = make_flagella_expression_initial_state()

    # use agent_environment_experiment to make the experiment
    experiment_settings = {
        'experiment_name': 'heterogeneous_flagella_experiment',
        'description': '..',
        'total_time': total_time,
        'emit_step': emit_step}
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    ## run the experiment
    experiment.update(total_time)
    data = experiment.emitter.get_data()

    ## plot output
    # extract data
    multibody_config = environment_config['config']['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    plot_data = {
        'agents': agents,
        'fields': fields,
        'config': multibody_config}

    # multigen plot
    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

    # make tag and snapshot plots
    plot_config = {
        'fields': emit_fields,
        'tagged_molecules': [('proteins', 'flagella')],
        'n_snapshots': 5,
        # 'background_color': background_color,
        'out_dir': out_dir}
    plot_tags(plot_data, plot_config)
    plot_snapshots(plot_data, plot_config)


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
        out_dir=out_dir,
        timeline=get_chemotaxis_timeline(
            timestep=0.1,
            total_time=90),
    )


# put all the experiments for the paper in a dictionary
# for easy access by main
experiments_library = {
    '5a': BiGG_metabolism,
    '6a': flagella_expression_network,
    '6b': flagella_just_in_time,
    '6c': run_flagella_metabolism_experiment,
    '7a': variable_flagella,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
    '7': ['7a', '7b', '7c'],
}

def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def add_arguments():
    parser = argparse.ArgumentParser(description='chemotaxis paper experiments')
    parser.add_argument(
        'experiment_id',
        type=str,
        choices=list(experiments_library.keys()),
        help='experiment id corresponds to figure number from chemotaxis paper')
    return parser.parse_args()

def main():
    """
    Execute experiments
    """

    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'chemotaxis')
    make_dir(out_dir)

    args = add_arguments()

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
                exp(control_out_dir)
    else:
        print('provide experiment number')


if __name__ == '__main__':
    main()
