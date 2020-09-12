'''
====================
Chemotaxis Experiments
====================

Chemotaxis provides several pre-configured :py:class:`Experiments`
with different chemotactic agents and environments.
'''

import os
import argparse
import uuid

import numpy as np

# vivarium-core imports
from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    simulate_compartment_in_experiment,
    agent_environment_experiment,
    plot_simulation_output,
    plot_agents_multigen,
)
from vivarium.core.emitter import time_indexed_timeseries_from_data

# vivarium-cell imports
from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config,
)
from cell.processes.transcription import UNBOUND_RNAP_KEY
from cell.processes.translation import UNBOUND_RIBOSOME_KEY
from cell.processes.static_field import make_field
from cell.composites.lattice import Lattice
from cell.composites.static_lattice import StaticLattice
from cell.experiments.lattice_experiment import get_iAF1260b_environment

# chemotaxis process imports
from chemotaxis.processes.flagella_motor import run_variable_flagella
from chemotaxis.processes.chemoreceptor_cluster import (
    test_receptor,
    get_pulse_timeline,
)

# chemotaxis composite imports
from chemotaxis.composites.chemotaxis_minimal import ChemotaxisMinimal
from chemotaxis.composites.chemotaxis_flagella import (
    ChemotaxisVariableFlagella,
    get_chemotaxis_timeline,
)
from chemotaxis.composites.flagella_expression import (
    FlagellaExpressionMetabolism,
    get_flagella_expression_compartment,
)
from chemotaxis.composites.transport_metabolism import (
    TransportMetabolismExpression,
    get_metabolism_initial_external_state,
)
from chemotaxis.composites.chemotaxis_master import ChemotaxisMaster

# data
from chemotaxis.data.chromosomes.flagella_chromosome import FlagellaChromosome
from cell.data.nucleotides import nucleotides
from cell.data.amino_acids import amino_acids

# plots
from chemotaxis.plots.chemotaxis_experiments import plot_chemotaxis_experiment
from cell.plots.metabolism import plot_exchanges
from cell.plots.gene_expression import (
    plot_timeseries_heatmaps,
    gene_network_plot,
)
from cell.plots.multibody_physics import (
    plot_agent_trajectory,
    plot_snapshots,
    plot_tags
)
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output
from chemotaxis.plots.transport_metabolism import analyze_transport_metabolism
from chemotaxis.plots.flagella_activity import plot_signal_transduction

# directories
from chemotaxis import EXPERIMENT_OUT_DIR





def make_agent_ids(agents_config):
    agent_ids = []
    for config in agents_config:
        number = config.get('number', 1)
        if 'name' in config:
            name = config['name']
            if number > 1:
                new_agent_ids = [name + '_' + str(num) for num in range(number)]
            else:
                new_agent_ids = [name]
        else:
            new_agent_ids = [str(uuid.uuid1()) for num in range(number)]
        config['ids'] = new_agent_ids
        agent_ids.extend(new_agent_ids)
    return agent_ids



# figure 3b
def growth_division_experiment(out_dir='out'):
    pass


# figure 5a
def BiGG_metabolism(out_dir='out'):
    # configure metabolism process iAF1260b BiGG model
    config = get_iAF1260b_config()
    metabolism = Metabolism(config)

    # get default minimal external concentrations
    external_concentrations = metabolism.initial_state['external']

    # run simulation with the helper function simulate_process_in_experiment
    sim_settings = {
        'environment': {
            'volume': 1e-5 * units.L,
            'concentrations': external_concentrations},
        'total_time': 2500}
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot
    plot_exchanges(timeseries, sim_settings, out_dir)


# figure 5b
def transport_metabolism(out_dir='out'):
    total_time = 3000
    environment_volume = 1e-13 * units.L
    initial_concentrations = {
        'glc__D_e': 1.0,
        'lcts_e': 1.0}

    # make the compartment
    compartment = TransportMetabolismExpression({
        'agent_id': '0',
        'divide': False})

    # get external state with adjusted minimal concentrations
    external_state = get_metabolism_initial_external_state(
        scale_concentration=100,
        override=initial_concentrations)

    # configure non-spatial environment
    # TransportMetabolismExpression redirects external through boundary port
    sim_settings = {
        'environment': {
            'volume': environment_volume,
            'concentrations': external_state,
            'ports': {
                'fields': ('fields',),
                'external': ('boundary', 'external'),
                'dimensions': ('dimensions',),
                'global': ('boundary',)}},
        'total_time': total_time}

    # run simulation
    timeseries = simulate_compartment_in_experiment(compartment, sim_settings)

    # plot
    plot_config = {
        'end_time': total_time,
        'environment_volume': environment_volume}
    analyze_transport_metabolism(timeseries, plot_config, out_dir)


# figure 5c
def transport_metabolism_environment(out_dir='out'):
    n_agents = 1
    total_time = 5000
    process_time_step = 10  # TODO -- pass time_step to compartment, processes
    bounds = [20, 20]
    emit_step = 100
    emit_fields = ['glc__D_e', 'lcts_e']

    # agent configuration
    agents_config = [{
        'name': 'transport_metabolism',
        'type': TransportMetabolismExpression,
        'number': n_agents,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
            'metabolism': {'time_step': 10},
            'transport': {'time_step': 10},
            'division': {'division_volume': 1.3 * units.fL}
        }}]
    # add agent_ids
    agent_ids = make_agent_ids(agents_config)

    # TODO -- get initial agent_state from transport
    # import ipdb; ipdb.set_trace()


    initial_agent_state = {
        'boundary': {
            'location': [8, 8],
            'external': {
                'glc__D_e': 1.0,
                'lcts_e': 1.0}}}

    # environment configuration
    environment_config = {
        'type': Lattice,
        'config': get_iAF1260b_environment(
            # time_step=process_time_step,
            bounds=bounds,
            depth=6000.0,
            diffusion=1e-2,
            scale_concentration=5,
            keep_fields_emit=emit_fields)}

    # make the experiment
    experiment_settings = {
        'experiment_name': 'transport_metabolism_environment',
        'description': 'glucose-lactose diauxic shifters are placed in a shallow environment with glucose and '
                       'lactose. They start off with no internal LacY and uptake only glucose, but LacY is '
                       'expressed upon depletion of glucose they begin to uptake lactose. Cells have an iAF1260b '
                       'BiGG metabolism, kinetic transport of glucose and lactose, and ode-based gene expression '
                       'of LacY',
        'total_time': total_time,
        'emit_step': emit_step}

    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    # run simulation
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
        'tagged_molecules': [('cytoplasm', 'LacY')],
        'n_snapshots': 5,
        'out_dir': out_dir}
    plot_tags(plot_data, plot_config)
    plot_snapshots(plot_data, plot_config)


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


# function to make initial state for flagella expression processes
def make_flagella_expression_initial_state():
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


# figure 6b
def flagella_just_in_time(out_dir='out'):

    # make the compartment
    compartment = get_flagella_expression_compartment({})

    # get the initial state
    initial_state = make_flagella_expression_initial_state()

    # run simulation
    settings = {
        # a cell cycle of 2520 sec is expected to express 8 flagella.
        # 2 flagella expected in approximately 630 seconds.
        'total_time': 500,
        'emit_step': 10.0,
        'verbose': True,
        'initial_state': initial_state}
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # plot output
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
    # time_step = 0.01
    # # make timeline with both chemoreceptor variation and flagella counts
    # timeline = get_chemoreceptor_timeline(
    #     total_time=3,
    #     time_step=time_step,
    #     rate=2.0,
    # )
    # timeline_flagella = [
    #     (0.5, {('internal_counts', 'flagella'): 1}),
    #     (1.0, {('internal_counts', 'flagella'): 2}),
    #     (1.5, {('internal_counts', 'flagella'): 3}),
    #     (2.0, {('internal_counts', 'flagella'): 4}),
    #     (2.5, {('internal_counts', 'flagella'): 5}),
    # ]
    # timeline.extend(timeline_flagella)
    #
    # # run simulation
    # data = test_flagella_motor(
    #     timeline=timeline,
    #     time_step=time_step,
    # )
    #
    # # plot
    # plot_settings = {}
    # timeseries = timeseries_from_data(data)
    # plot_simulation_output(timeseries, plot_settings, out_dir)
    # plot_activity(data, plot_settings, out_dir)


# figure 7b
def run_chemoreceptor_pulse(out_dir='out'):
    timeline = get_pulse_timeline()
    timeseries = test_receptor(timeline)
    plot_receptor_output(timeseries, out_dir, 'pulse')


# figure 7c
def run_chemotaxis_transduction(out_dir='out'):
    n_flagella = 5
    ligand_id = 'MeAsp'
    total_time = 90

    # configure the compartment
    config = {
        'receptor': {
            'ligand_id': 'MeAsp',
            'initial_ligand': 1e-2},
        'flagella': {
            'n_flagella': n_flagella}}
    compartment = ChemotaxisVariableFlagella(config)

    # make a timeline
    timeline = get_chemotaxis_timeline(
        ligand_id=ligand_id,
        timestep=0.1,
        total_time=total_time)

    # run experiment
    experiment_settings = {
        'timeline': {
            'timeline': timeline,
            'ports': {'external': ('boundary', 'external')}}}
    timeseries = simulate_compartment_in_experiment(
        compartment,
        experiment_settings)

    # plot
    plot_signal_transduction(timeseries, out_dir)



# helper functions for chemotaxis
def single_agent_config(config):
    width = 1
    length = 2
    # volume = volume_from_length(length, width)
    bounds = config.get('bounds')
    location = config.get('location')
    location = [loc * bounds[n] for n, loc in enumerate(location)]
    return {'boundary': {
        'location': location,
        # 'angle': np.random.uniform(0, 2 * PI),
        # 'volume': volume,
        'length': length,
        'width': width,
        'mass': 1339 * units.fg,
        'thrust': 0,
        'torque': 0}}

def agent_body_config(config):
    agent_ids = config['agent_ids']
    agent_config = {
        agent_id: single_agent_config(config)
        for agent_id in agent_ids}
    return {
        'agents': agent_config}


# figure 7d
def run_chemotaxis_experiment(out_dir='out'):
    total_time = 30
    emit_step = 5
    time_step = 0.001
    tumble_jitter = 4000  # why tumble jitter?

    ligand_id = 'glc__D_e'  # BiGG id for external glucose
    bounds = [1000, 3000]
    initial_agent_location = [0.5, 0.1]

    # exponential field parameters
    # TODO -- not uppercase!
    FIELD_SCALE = 4.0
    EXPONENTIAL_BASE = 1.3e2
    FIELD_CENTER = [0.5, 0.0]
    LOC_DX = (initial_agent_location[0] - FIELD_CENTER[0]) * bounds[0]
    LOC_DY = (initial_agent_location[1] - FIELD_CENTER[1]) * bounds[1]
    DIST = np.sqrt(LOC_DX ** 2 + LOC_DY ** 2)
    INITIAL_LIGAND = FIELD_SCALE * EXPONENTIAL_BASE ** (DIST / 1000)

    # configure agents
    agents_config = [{
        'number': 2,
        'name': 'receptor + motor',
        'type': ChemotaxisMinimal,
        'config': {
            'ligand_id': ligand_id,
            'initial_ligand': INITIAL_LIGAND,
            'external_path': ('global',),
            'agents_path': ('..', '..', 'agents'),
            'daughter_path': tuple(),
            'receptor': {
                'time_step': time_step},
            'motor': {
                'tumble_jitter': tumble_jitter,
                'time_step': time_step}
        }
    }]
    agent_ids = make_agent_ids(agents_config)
    # agents_config = {
    #     'ids': ['chemotaxis_master'],
    #     'type': ChemotaxisMaster,
    #     'config': {
    #         'agents_path': ('..', '..', 'agents'),
    #         'fields_path': ('..', '..', 'fields'),
    #         'dimensions_path': ('..', '..', 'dimensions')}}

    # configure environment
    environment_config = {
        'type': StaticLattice,
        'config': {
            'multibody': {
                'bounds': bounds
            },
            'field': {
                'molecules': [ligand_id],
                'gradient': {
                    'type': 'exponential',
                    'molecules': {
                        ligand_id: {
                            'center': FIELD_CENTER,
                            'scale': FIELD_SCALE,
                            'base': EXPONENTIAL_BASE}}},
                'bounds': bounds
            }
        }
    }

    # initialize state
    initial_state = {}
    initial_agent_body = agent_body_config({
        'bounds': bounds,
        'agent_ids': agent_ids,
        'location': initial_agent_location})
    initial_state.update(initial_agent_body)

    # use agent_environment_experiment to make the experiment
    experiment_settings = {
        'experiment_name': 'chemotaxis_experiment',
        'description': '..',
        'total_time': total_time,
        'emit_step': emit_step}
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_state=initial_state,
        settings=experiment_settings)

    # run the experiment
    experiment.update(total_time)
    data = experiment.emitter.get_data()

    # plot trajectory
    field_config = environment_config['config']['field']
    indexed_timeseries = time_indexed_timeseries_from_data(data)
    field = make_field(field_config)
    trajectory_config = {
        'bounds': field_config['bounds'],
        'field': field,
        'rotate_90': True}
    plot_agent_trajectory(indexed_timeseries, trajectory_config, out_dir, 'trajectory')



# put all the experiments for the paper in a dictionary
# for easy access by main
experiments_library = {
    '3b': growth_division_experiment,
    '5a': BiGG_metabolism,
    '5b': transport_metabolism,
    '5c': transport_metabolism_environment,
    '6a': flagella_expression_network,
    '6b': flagella_just_in_time,
    '6c': run_flagella_metabolism_experiment,
    '7a': variable_flagella,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
    '7d': run_chemotaxis_experiment,
    '5': ['5a', '5b', '5c'],
    '6': ['6a', '6b', '6c'],
    '7': ['7a', '7b', '7c', '7d'],
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