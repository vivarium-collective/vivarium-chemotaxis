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
    plot_simulation_output,
    EXPERIMENT_OUT_DIR,
)

# vivarium-cell imports
from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config,
)
from cell.processes.transcription import UNBOUND_RNAP_KEY
from cell.processes.translation import UNBOUND_RIBOSOME_KEY

# chemotaxis imports
from chemotaxis.processes.flagella_motor import run_variable_flagella
from chemotaxis.processes.chemoreceptor_cluster import (
    test_receptor,
    get_pulse_timeline,
)
from chemotaxis.compartments.chemotaxis_flagella import (
    test_variable_chemotaxis,
    get_chemotaxis_timeline,
)
from chemotaxis.compartments.flagella_expression import (
    flagella_expression_compartment,
)

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
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output





# figure 5a
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
    flagella_compartment = flagella_expression_compartment({})

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
def flagella_just_in_time(out_dir='out'):

    ## make the compartment
    compartment = flagella_expression_compartment({})

    ## make the initial state
    flagella_data = FlagellaChromosome()
    chromosome_config = flagella_data.chromosome_config

    molecules = {}
    for nucleotide in nucleotides.values():
        molecules[nucleotide] = 5000000
    for amino_acid in amino_acids.values():
        molecules[amino_acid] = 1000000

    initial_state = {
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
            'flagella': 8,
            UNBOUND_RIBOSOME_KEY: 100,  # e. coli has ~ 20000 ribosomes
            UNBOUND_RNAP_KEY: 100
        }
    }

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


experiments_library = {
    '4a': BiGG_metabolism,
    '6a': flagella_expression_network,
    '6b': flagella_just_in_time,
    '7a': variable_flagella,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
    '7': ['7a', '7b', '7c'],
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
        experiment_type = experiments_library[experiment_id]

        if callable(experiment_type):
            control_out_dir = os.path.join(out_dir, experiment_id)
            make_dir(control_out_dir)
            experiment_type(control_out_dir)
        elif isinstance(experiment_type, list):
            for exp_id in experiment_type:
                control_out_dir = os.path.join(out_dir, experiment_id, exp_id)
                make_dir(control_out_dir)
                exp = experiments_library[exp_id]
                exp(control_out_dir)

    else:
        print('provide experiment number')


if __name__ == '__main__':
    main()
