"""
=============================
Chemotaxis Paper Experiments
=============================

Includes functions for configuring, running, and plotting all experiments reported in the paper:
    Agmon, E. and Spangler, R.K., "A Multi-Scale Approach to Modeling E. coli Chemotaxis"

These experiments can be triggered from the command line by entering the figure number.
Available experiments include: '3b', '5a', '5b', '5c', '5d', '6a', '6b', '6c', '7a', '7b', '7c', '7d'.

```
$ python chemotaxis/experiments/paper_experiments.py 7a
```

Notes:
    * some of the larger experiments require a MongoDB connection.
    These have the experiment setting {'emitter': {'type': 'database'}}.
    To run them without saving to a database, remove this setting.

"""

import numpy as np
import copy

# vivarium-core imports
from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    simulate_compartment_in_experiment,
    agent_environment_experiment,
    plot_simulation_output,
    plot_agents_multigen,
    make_agent_ids,
)
from vivarium.core.emitter import (
    time_indexed_timeseries_from_data,
    timeseries_from_data)
from vivarium.library.dict_utils import deep_merge

# experiment workflow
from chemotaxis.experiments.control import (
    control,
    plot_control,
    agent_body_config,
)

# vivarium-cell imports
from cell.processes.metabolism import (
    Metabolism,
    get_minimal_media_iAF1260b,
    get_iAF1260b_config,
)
from cell.composites.growth_division import GrowthDivision
from cell.processes.static_field import make_field
from cell.composites.lattice import (
    Lattice,
    make_lattice_config,
)
from cell.composites.static_lattice import StaticLattice

# chemotaxis processes
from chemotaxis.processes.flagella_motor import (
    FlagellaMotor,
    get_chemoreceptor_activity_timeline,
)
from chemotaxis.processes.chemoreceptor_cluster import (
    ReceptorCluster,
    get_pulse_timeline,
    get_brownian_ligand_timeline,
)

# chemotaxis composites
from chemotaxis.composites.chemotaxis_master import ChemotaxisMaster
from chemotaxis.composites.chemotaxis_minimal import ChemotaxisMinimal
from chemotaxis.composites.flagella_expression import (
    FlagellaExpressionMetabolism,
    get_flagella_metabolism_initial_state)
from chemotaxis.composites.transport_metabolism import (
    TransportMetabolismExpression,
    ODE_expression,
    get_metabolism_initial_external_state,
    get_lacY_expression_config)

# data
from chemotaxis.data.chromosomes.flagella_chromosome import FlagellaChromosome
from cell.data.nucleotides import nucleotides
from cell.data.amino_acids import amino_acids

# plots
from cell.plots.metabolism import plot_exchanges
from cell.plots.gene_expression import (
    plot_timeseries_heatmaps,
    gene_network_plot)
from cell.plots.multibody_physics import plot_agent_trajectory
from chemotaxis.plots.chemotaxis_experiments import plot_motility
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output
from chemotaxis.plots.transport_metabolism import plot_glc_lcts_environment
from chemotaxis.plots.flagella_activity import (
    plot_signal_transduction,
    plot_activity)


# figure 3b
def growth_division_experiment(out_dir='out'):

    # simulation parameters
    total_time = 21000
    emit_step = 120
    initial_agent_id = 'growth_division'
    parallel = True

    # environment parameters
    env_time_step = 60
    fields = ['glc__D_e', 'lcts_e']
    emit_fields = ['glc__D_e']

    # make agent configuration
    agents_config = {
        'ids': [initial_agent_id],
        'type': GrowthDivision,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions')}}

    # make environment configuration
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=env_time_step,
            bounds=[30, 30],
            molecules=fields,
            keep_fields_emit=emit_fields,
            parallel=parallel)}

    # make the experiment using helper function agent_environment_experiment
    experiment_settings = {
        'experiment_name': '3b',
        'description': 'a simple GrowthDivision agent is placed in a Lattice environment and grown.',
        'total_time': total_time,
        'emit_step': emit_step}
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        settings=experiment_settings)

    # run the simulation
    experiment.update(total_time)

    # retrieve the data from the emitter
    data = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # plot output
    plot_config = {
        'environment_config': environment_config,
        'emit_fields': emit_fields,
        'topology_network': {
            'compartment': GrowthDivision({
                'agent_id': initial_agent_id})}}
    plot_control(data, plot_config, out_dir)


# figure 5a
def BiGG_metabolism(out_dir='out'):

    # simulation parameters
    total_time = 2500
    env_volume = 1e-13 * units.L

    # get iAF1260b BiGG model configuration
    process_config = get_iAF1260b_config()

    # make the metabolism process
    metabolism = Metabolism(process_config)

    # configure the experiment
    # use default minimal external concentrations from metabolism
    sim_settings = {
        'environment': {
            'volume': env_volume,
            'concentrations': metabolism.initial_state['external']},
        'total_time': total_time}

    # run simulation with the helper function simulate_process_in_experiment
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot output
    plot_config = {
        'environment': {'volume': env_volume},
        'legend': False,
        'aspect_ratio': 1.0}
    plot_exchanges(timeseries, plot_config, out_dir)


# figure 5b
def transport_metabolism(out_dir='out'):

    # experiment parameters
    total_time = 5000
    environment_volume = 1e-13 * units.L
    initial_concentrations = {
        'glc__D_e': 1.0,
        'lcts_e': 1.0}

    # make the agent configuration
    # parameters passed to each process override compartment defaults
    # increased leak rate in expression process makes for improved visualization
    agent_config = {
        'agent_id': '0',
        'metabolism': {'time_step': 10},
        'transport': {'time_step': 10},
        'expression': {
            'transcription_leak': {
                'rate': 4e-3,
                'magnitude': 2e-7}},
        'divide': False}

    # make the compartment
    compartment = TransportMetabolismExpression(agent_config)

    # get external state with adjusted minimal concentrations
    external_state = get_metabolism_initial_external_state(
        scale_concentration=1000,
        override=initial_concentrations)

    # configure the experiment
    # uses non-spatial environment process with environment_volume
    sim_settings = {
        'environment': {
            'volume': environment_volume,
            'concentrations': external_state,
            'ports': {
                'fields': ('fields',),
                'external': ('boundary', 'external'),  # redirect external through boundary port
                'dimensions': ('dimensions',),
                'global': ('boundary',)}},
        'total_time': total_time}

    # run simulation with helper function simulate_compartment_in_experiment
    timeseries = simulate_compartment_in_experiment(compartment, sim_settings)

    # plot output
    plot_settings = {
        'internal_path': ('cytoplasm',),
        'external_path': ('boundary', 'external'),
        'global_path': ('boundary',),
        'environment_volume': environment_volume,
        'aspect_ratio': 1.0}
    plot_glc_lcts_environment(timeseries, plot_settings, out_dir)
    plot_simulation_output(timeseries, {}, out_dir)


# figure 5c
def transport_metabolism_environment(out_dir='out'):

    # simulation parameters
    total_time = 30000
    emit_step = 100
    parallel = True  # TODO -- make this an option you can pass in

    # environment parameters
    # put cells in a very shallow environment, with low concentrations of glucose.
    # this makes it possible to observe the glucose-lactose shift in individual cells.
    bounds = [35, 35]
    n_bins = [30, 30]
    depth = 50.0
    diffusion = 5e-3
    jitter_force = 1e-2
    env_timestep = 10
    initial_external = {
        'glc__D_e': 1.0,
        'lcts_e': 8.0}

    # cell parameters
    # small time steps required in depleted environment
    n_agents = 3
    process_timestep = 10

    # plotting parameters
    emit_fields = [
        'glc__D_e',
        'lcts_e']
    tagged_molecules = [
        ('cytoplasm', 'LacY'),
        ('cytoplasm', 'lacy_RNA'),
        ('cytoplasm', 'lcts_p'),
        ('flux_bounds', 'EX_glc__D_e')]

    # agent configuration
    # parameters passed to each process override compartment default
    # paths to ports are assigned to go up two levels in the hierarchy
    # So they can plug into the Lattice environment.
    agents_config = [{
        'name': 'transport_metabolism',
        'type': TransportMetabolismExpression,
        'number': n_agents,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
            'division': {'time_step': process_timestep},
            'metabolism': {
                'time_step': process_timestep,
                '_parallel': parallel},
            'transport': {'time_step': process_timestep},
            'expression': {
                'transcription_leak': {
                    'rate': 5e-5,
                    'magnitude': 5e-7}}
        }}]
    make_agent_ids(agents_config)  # add agent_ids based on n_agents

    # initial agent state
    initial_agent_state = {
        'boundary': {
            'location': [bound/2 for bound in bounds],
            'external': initial_external}}

    # get minimal media for iAF1260b
    # scale concentrations of nutrients other than initial_external
    # to ensure they remain in full supply in the shallow environment
    media = get_minimal_media_iAF1260b(
        scale_concentration=1e6,
        override_initial=initial_external)

    # environment configuration
    # use make_lattice_config to override defaults
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=env_timestep,
            bounds=bounds,
            n_bins=n_bins,
            depth=depth,
            diffusion=diffusion,
            concentrations=media,
            jitter_force=jitter_force,
            keep_fields_emit=emit_fields,
            parallel=parallel)}

    # configure the experiment
    # database emitter saves to mongoDB
    experiment_settings = {
        'experiment_name': '5c',
        'description': 'TransportMetabolismExpression cells are placed in a shallow environment with glucose and '
                       'lactose. They start with no internal LacY and uptake only glucose, but upon depletion of '
                       'glucose some cells begin to express LacY uptake lactose.',
        'total_time': total_time,
        'emit_step': emit_step,
        'emitter': {'type': 'database'}}

    # make the experiment with helper function agent_environment_experiment
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    # run simulation
    experiment.update(total_time)
    data = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # plot output
    plot_config = {
        'environment_config': environment_config,
        'emit_fields': emit_fields,
        'tagged_molecules': tagged_molecules}
    plot_control(data, plot_config, out_dir)


# figure 5d
def lacy_expression(out_dir='out'):
    """ two experiments for ODE-based LacY expression

    These experiments use only the ode_expression process,
    and no transport from the resulting proteins
        * experiment 1: external glucose starts high and drops to 0.0
        * experiment 2: external glucose starts high and drops to 0.0 along with an increase in internal lactose.
    """

    # parameters
    total_time = 8000
    shift_time1 = int(total_time / 5)
    shift_time2 = int(4 * total_time / 5)

    # get process configuration
    expression_config = get_lacY_expression_config()

    # make the expression process
    process = ODE_expression(expression_config)

    # make timelines for two experiments
    timeline1 = [
        (0, {('external', 'glc__D_e'): 10}),
        (shift_time1, {('external', 'glc__D_e'): 0.0}),
        (shift_time2, {('external', 'glc__D_e'): 10.}),
        (total_time, {})]
    timeline2 = [
        (0, {('external', 'glc__D_e'): 10}),
        (shift_time1, {
            ('external', 'glc__D_e'): 0,
            ('internal', 'lcts_p'): 0.1
        }),
        (shift_time2, {('external', 'glc__D_e'): 0.1}),
        (total_time, {})]

    # experiment 1
    settings1 = {'timeline': {'timeline': timeline1}}
    timeseries1 = simulate_process_in_experiment(process, settings1)
    plot_simulation_output(timeseries1, {}, out_dir, 'experiment_1')

    # experiment 2
    settings2 = {'timeline': {'timeline': timeline2}}
    timeseries2 = simulate_process_in_experiment(process, settings2)
    plot_simulation_output(timeseries2, {}, out_dir, 'experiment_2')


# figure 6a
def flagella_expression_network(out_dir='out'):
    """
    Make a network plot of the flagella expression processes.
    This saves an networkx plot with a default layout, along with
    node and edge list files of the network for analysis by network
    visualization software.
    """

    # load the compartment and pull out the processes
    flagella_compartment = FlagellaExpressionMetabolism()
    flagella_expression_network = flagella_compartment.generate()
    flagella_expression_processes = flagella_expression_network['processes']

    # make expression network plot
    data = {
        'operons': flagella_expression_processes['transcription'].genes,
        'templates': flagella_expression_processes['transcription'].templates,
        'complexes': flagella_expression_processes['complexation'].stoichiometry}
    gene_network_plot(data, out_dir)


# figure 6b
def flagella_just_in_time(out_dir='out'):

    # experiment parameters
    total_time = 4000

    # configuration
    # longer time steps speed up the simulation,
    # and are sufficient to provide the required nutrients
    compartment_config = {
        'expression_time_step': 60,
        'transport': {'time_step': 60},
        'metabolism': {'time_step': 60},
        'chromosome': {'tsc_affinity_scaling': 1e-1},
        'divide': False}

    # make the compartment
    compartment = FlagellaExpressionMetabolism(compartment_config)

    # get initial state
    initial_state = get_flagella_metabolism_initial_state()

    # run simulation with helper function simulate_compartment_in_experiment
    settings = {
        'total_time': total_time,
        'initial_state': initial_state}
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # plot output as a heat maps
    # transcript_list is made in expected just-in-time order
    # order proteins and small molecules alphabetically
    flagella_data = FlagellaChromosome()
    transcript_list = list(flagella_data.chromosome_config['genes'].keys())
    protein_list = flagella_data.complexation_monomer_ids + flagella_data.complexation_complex_ids
    protein_list.sort()
    molecule_list = list(nucleotides.values()) + list(amino_acids.values())
    molecule_list.sort()
    plot_config = {
        'name': 'flagella',
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'molecules'},
        'plot_ports': {
            'transcripts': transcript_list,
            'proteins': protein_list,
            'molecules': molecule_list}}
    plot_timeseries_heatmaps(timeseries, plot_config, out_dir)

    # plot output
    sim_plot_config = {
        'max_rows': 30,
        'remove_zeros': True,
        'skip_ports': ['chromosome', 'ribosomes']}
    plot_simulation_output(timeseries, sim_plot_config, out_dir)


# figure 6c
def run_heterogeneous_flagella_experiment(out_dir='out'):

    total_time = 15000
    emit_step = 120
    process_time_step = 60
    environment_time_step = 120
    bounds = [18, 18]
    tagged_molecules = [
        ('proteins', 'flagella'),
        ('proteins', 'fliA'),
        ('proteins', 'flhDC'),
    ]
    emit_fields = ['glc__D_e']
    parallel = True

    # configure agents and environment
    agents_config = {
        'ids': ['flagella_metabolism'],
        'type': FlagellaExpressionMetabolism,
        'config': {
            'expression_time_step': process_time_step,
            'transport': {'time_step': process_time_step},
            'metabolism': {'time_step': process_time_step},
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions')}}

    # get minimal media concentrations
    media = get_minimal_media_iAF1260b()

    # configure the environment
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=environment_time_step,
            concentrations=media,
            bounds=bounds,
            depth=6000.0,
            keep_fields_emit=emit_fields,
            parallel=parallel)}

    # get initial agent state
    initial_agent_state = get_flagella_metabolism_initial_state()
    initial_agent_state.update({'boundary': {'location': [bound/2 for bound in bounds]}})

    # use agent_environment_experiment to make the experiment
    # database emitter saves to mongoDB
    experiment_settings = {
        'experiment_name': '6c',
        'description': 'A single FlagellaExpressionMetabolism compartment is placed in a Lattice environment and'
                       'grown into a small colony to demonstrate heterogeneous expression of flagellar genes.',
        'total_time': total_time,
        'emit_step': emit_step,
        'emitter': {'type': 'database'},
    }
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    # run the experiment
    experiment.update(total_time)
    data = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # remove the first timestep so that it is not always the brightest
    if 0.0 in data:
        del data[0.0]

    # plot output
    plot_config = {
        'environment_config': environment_config,
        'tagged_molecules': tagged_molecules,
        'emit_fields': emit_fields,
        'topology_network': {
            'compartment': FlagellaExpressionMetabolism({})}}
    plot_control(data, plot_config, out_dir)


# figure 7a
def run_flagella_activity(out_dir='out'):
    total_time = 80
    time_step = 0.01
    initial_flagella = 1

    # make the process
    process_config = {'n_flagella': initial_flagella}
    process = FlagellaMotor(process_config)

    # make a timeline with varying chemoreceptor activity and flagella counts
    timeline = get_chemoreceptor_activity_timeline(
        total_time=total_time,
        time_step=time_step,
        rate=2.0)
    timeline_flagella = [
        (20, {('internal_counts', 'flagella'): initial_flagella + 1}),
        (40, {('internal_counts', 'flagella'): initial_flagella + 2}),
        (60, {('internal_counts', 'flagella'): initial_flagella + 3})]
    timeline.extend(timeline_flagella)

    # run simulation with helper function simulate_process_in_experiment
    settings = {
        'return_raw_data': True,
        'timeline': {
            'timeline': timeline,
            'time_step': time_step}}
    data = simulate_process_in_experiment(process, settings)

    # plot output
    plot_settings = {'aspect_ratio': 0.4}
    plot_activity(data, plot_settings, out_dir)


# figure 7b
def run_chemoreceptor_pulse(out_dir='out'):
    ligand = 'MeAsp'
    timeline = get_pulse_timeline(ligand=ligand)

    # initialize the process
    initial_ligand = timeline[0][1][('external', ligand)]
    process_config = {'initial_ligand': initial_ligand}
    receptor = ReceptorCluster(process_config)

    # run experiment with helper function simulate_process_in_experiment
    experiment_settings = {
        'timeline': {
            'timeline': timeline}}
    timeseries = simulate_process_in_experiment(receptor, experiment_settings)

    # plot output
    plot_settings = {'aspect_ratio': 0.4}
    plot_receptor_output(timeseries, plot_settings, out_dir, 'pulse')


# figure 7c
def run_chemotaxis_transduction(out_dir='out'):
    total_time = 60
    time_step = 0.1
    n_flagella = 5
    ligand_id = 'MeAsp'
    initial_ligand = 1e-1

    # configure the compartment
    compartment_config = {
        'receptor': {
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand,
            'time_step': time_step},
        'flagella': {
            'n_flagella': n_flagella,
            'time_step': time_step}}
    compartment = ChemotaxisMaster(compartment_config)

    # make a timeline of external ligand concentrations
    timeline = get_brownian_ligand_timeline(
        ligand_id=ligand_id,
        initial_conc=initial_ligand,
        timestep=time_step,
        total_time=total_time,
        speed=8)

    # run experiment with helper function simulate_compartment_in_experiment
    experiment_settings = {
        'emit_step': time_step,
        'timeline': {
            'timeline': timeline,
            'time_step': time_step,
            'ports': {
                'external': ('boundary', 'external')},
        }}
    timeseries = simulate_compartment_in_experiment(
        compartment,
        experiment_settings)

    # plot output
    plot_config = {
        'ligand_id': ligand_id,
        'aspect_ratio': 0.4}
    plot_signal_transduction(timeseries, plot_config, out_dir)


# figure 7d
def run_chemotaxis_experiment(out_dir='out'):

    # simulation parameters
    total_time = 60 * 8
    n_receptor_motor = 6
    n_motor = 6

    # agent parameters
    fast_process_timestep = 0.01
    slow_process_timestep = 10

    # environment parameters
    ligand_id = 'MeAsp'
    bounds = [1000, 3000]

    # exponential field parameters
    initial_agent_location = [0.5, 0.1]
    field_center = [0.5, 0.0]
    field_scale = 4.0
    exponential_base = 1.3e2

    # initialize ligand concentration based on position in exponential field
    # this allows the receptor process to initialize at a steady states
    # TODO -- this can be calculated by static_field.get_concentration
    loc_dx = (initial_agent_location[0] - field_center[0]) * bounds[0]
    loc_dy = (initial_agent_location[1] - field_center[1]) * bounds[1]
    dist = np.sqrt(loc_dx ** 2 + loc_dy ** 2)
    initial_ligand = field_scale * exponential_base ** (dist / 1000)

    # configure agents
    master_chemotaxis_config = {
        'agents_path': ('..', '..', 'agents'),
        'fields_path': ('..', '..', 'fields'),
        'dimensions_path': ('..', '..', 'dimensions'),
        'daughter_path': tuple(),
        'receptor': {
            'time_step': fast_process_timestep,
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand},
        'flagella': {'time_step': fast_process_timestep},
        'PMF': {'time_step': fast_process_timestep},
        'transport': {'time_step': slow_process_timestep},
        'metabolism': {'time_step': slow_process_timestep},
        'transcription': {'time_step': slow_process_timestep},
        'translation': {'time_step': slow_process_timestep},
        'degradation': {'time_step': slow_process_timestep},
        'complexation': {'time_step': slow_process_timestep},
        'division': {'time_step': slow_process_timestep}}

    # chemoreceptor configuration with 'None' ligand_id,
    # which will leave it in a steady state.
    no_receptor_config = deep_merge(
        copy.deepcopy(master_chemotaxis_config),
        {'receptor': {'ligand_id': 'None', 'initial_ligand': 1}})

    # list of agent configurations
    agents_config = [
        {
            'number': n_receptor_motor,
            'name': 'receptor + motor',
            'type': ChemotaxisMaster,
            'config': master_chemotaxis_config,
        },
        {
            'number': n_motor,
            'name': 'motor',
            'type': ChemotaxisMaster,
            'config': no_receptor_config,
        },
    ]
    agent_ids = make_agent_ids(agents_config)

    # configure the environment
    environment_config = {
        'type': StaticLattice,
        'config': {
            'multibody': {
                'time_step': fast_process_timestep,
                'bounds': bounds},
            'field': {
                'time_step': fast_process_timestep,
                'molecules': [ligand_id],
                'gradient': {
                    'type': 'exponential',
                    'molecules': {
                        ligand_id: {
                            'center': field_center,
                            'scale': field_scale,
                            'base': exponential_base}}},
                'bounds': bounds,
            }
        }}

    # initialize experiment state
    initial_state = {}
    initial_agent_body = agent_body_config({
        'bounds': bounds,
        'agent_ids': agent_ids,
        'location': initial_agent_location})
    initial_state.update(initial_agent_body)

    # configure the experiment
    # database emitter saves to mongoDB
    experiment_settings = {
        'experiment_name': '7d',
        'description': 'Two configurations of ChemotaxisMaster -- one with receptors for MeAsp another without '
                       'useful chemoreceptors -- are placed in a large StaticLattice environment with an '
                       'exponential gradient to demonstrate their chemotaxis.',
        'total_time': total_time,
        'emit_step': fast_process_timestep * 10,
        'emitter': {'type': 'database'},
    }

    # use helper function agent_environment_experiment to make the experiment
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_state=initial_state,
        settings=experiment_settings)

    # run the experiment
    experiment.update(total_time)
    data = experiment.emitter.get_data()
    experiment.end()

    # plot trajectory
    field_config = environment_config['config']['field']
    indexed_timeseries = time_indexed_timeseries_from_data(data)
    field = make_field(field_config)
    plot_config = {
        'bounds': field_config['bounds'],
        'field': field,
        'rotate_90': True,
    }
    plot_agent_trajectory(indexed_timeseries, plot_config, out_dir, 'trajectory')

    # multigen agents plot
    plot_settings = {
        'agents_key': 'agents',
        'max_rows': 30}
    plot_agents_multigen(data, plot_settings, out_dir)

    # motility
    embdedded_timeseries = timeseries_from_data(data)
    plot_motility(embdedded_timeseries, out_dir)


# put all the experiments for the paper in a dictionary
# for easy access by main
experiments_library = {
    '3b': growth_division_experiment,
    '5a': BiGG_metabolism,
    '5b': transport_metabolism,
    '5c': transport_metabolism_environment,
    '5d': lacy_expression,
    '6a': flagella_expression_network,
    '6b': flagella_just_in_time,
    '6c': run_heterogeneous_flagella_experiment,
    '7a': run_flagella_activity,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
    '7d': run_chemotaxis_experiment,
    '5': ['5a', '5b', '5c', '5d'],
    '6': ['6a', '6b', '6c'],
    '7': ['7a', '7b', '7c', '7d'],
    'all': [
        '3b',
        '5a', '5b', '5c', '5d',
        '6a', '6b', '6c',
        '7a', '7b', '7c', '7d'],
}


if __name__ == '__main__':
    control(experiments_library)
