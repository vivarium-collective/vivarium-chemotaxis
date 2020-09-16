"""
=============================
Chemotaxis Paper Experiments
=============================

Includes functions for configuring, running, and plotting all experiments reported in the paper:
    Agmon, E. and Spangler, R.K., "A Multi-Scale Approach to Modeling E. coli Chemotaxis"

These experiments can be triggered from the command line by entering the figure number.
Available experiments include: '3b', '5a', '5b', '5c', '5d', 6a', '6b', '6c', '7a', '7b', '7c', '7d'.

```
$ python chemotaxis/experiments/paper_experiments.py 7a
```

Notes:
    * some of the larger experiments require a MongoDB connection.
    These have the experiment setting {'emitter': {'type': 'database'}}.
    To run them without saving to a database, remove this setting.

"""

import numpy as np

# vivarium-core imports
from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    simulate_compartment_in_experiment,
    agent_environment_experiment,
    plot_simulation_output,
    make_agent_ids,
)
from vivarium.core.emitter import time_indexed_timeseries_from_data

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
    get_flagella_metabolism_initial_state,
)
from chemotaxis.composites.transport_metabolism import (
    TransportMetabolismExpression,
    ODE_expression,
    get_metabolism_initial_external_state,
    lacy_expression_config,
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
from cell.plots.multibody_physics import plot_agent_trajectory
from chemotaxis.plots.chemoreceptor_cluster import plot_receptor_output
from chemotaxis.plots.transport_metabolism import plot_glc_lcts_environment
from chemotaxis.plots.flagella_activity import (
    plot_signal_transduction,
    plot_activity,
)


# figure 3b
def growth_division_experiment(out_dir='out'):
    total_time = 21000
    emit_step = 120
    env_time_step = 60
    fields = ['glc__D_e', 'lcts_e']
    emit_fields = ['glc__D_e']
    initial_agent_id = 'growth_division'
    parallel = False

    # configure the agents and environment
    agents_config = {
        'ids': [initial_agent_id],
        'type': GrowthDivision,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
        }
    }
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=env_time_step,
            bounds=[30, 30],
            molecules=fields,
            keep_fields_emit=emit_fields,
            parallel=parallel,
        )
    }

    # make the experiment
    experiment_settings = {
        'experiment_name': 'growth_division_experiment',
        'description': 'simple growth_division agents are placed in a lattice'
                       ' environment and grown.',
        'total_time': total_time,
        'emit_step': emit_step,
    }
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        settings=experiment_settings)

    # run simulation
    experiment.update(total_time)
    data = experiment.emitter.get_data()
    experiment.end()  # end required for parallel processes

    # plots
    plot_config = {
        'environment_config': environment_config,
        'emit_fields': emit_fields,
        'topology_network': {
            'compartment': GrowthDivision({
                'agent_id': initial_agent_id})
        }
    }
    plot_control(data, plot_config, out_dir)


# figure 5a
def BiGG_metabolism(out_dir='out'):
    total_time = 2500
    env_volume = 1e-13 * units.L

    # configure metabolism process with iAF1260b BiGG model
    process_config = get_iAF1260b_config()
    metabolism = Metabolism(process_config)

    # run simulation with the helper function simulate_process_in_experiment
    # use default minimal external concentrations from metabolism processes
    external_concentrations = metabolism.initial_state['external']
    sim_settings = {
        'environment': {
            'volume': env_volume,
            'concentrations': external_concentrations,
        },
        'total_time': total_time,
    }
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot output
    plot_config = {
        'environment': {
            'volume': env_volume
        },
        'legend': False,
        'aspect_ratio': 0.7,
    }
    plot_exchanges(timeseries, plot_config, out_dir)


# figure 5b
def transport_metabolism(out_dir='out'):
    total_time = 6000
    environment_volume = 1e-13 * units.L
    initial_concentrations = {
        'glc__D_e': 1.0,
        'lcts_e': 1.0,
    }

    # make the compartment
    compartment_config = {
        'agent_id': '0',
        'metabolism': {'time_step': 10},
        'transport': {'time_step': 10},
        'expression': {
            'time_step': 1,
            # increased leak rate makes more frequent bursts
            'transcription_leak': {
                'rate': 5e-3,
            },
        },
        'divide': False,
    }
    compartment = TransportMetabolismExpression(compartment_config)

    # get external state with adjusted minimal concentrations
    external_state = get_metabolism_initial_external_state(
        scale_concentration=100,
        override=initial_concentrations)

    # run simulation with helper function simulate_compartment_in_experiment
    sim_settings = {
        # configure non-spatial environment
        'environment': {
            'volume': environment_volume,
            'concentrations': external_state,
            'ports': {
                'fields': ('fields',),
                # redirect external through boundary port
                'external': ('boundary', 'external'),
                'dimensions': ('dimensions',),
                'global': ('boundary',),
            },
        },
        'total_time': total_time,
    }
    timeseries = simulate_compartment_in_experiment(compartment, sim_settings)

    # plot output
    plot_settings = {
        'internal_path': ('cytoplasm',),
        'external_path': ('boundary', 'external'),
        'global_path': ('boundary',),
        'environment_volume': environment_volume,
        'aspect_ratio': 0.7,
    }
    plot_glc_lcts_environment(timeseries, plot_settings, out_dir)
    plot_simulation_output(timeseries, {}, out_dir)


# figure 5c
def transport_metabolism_environment(out_dir='out'):
    n_agents = 2
    total_time = 15000
    emit_step = 100
    bounds = [30, 30]
    emit_fields = ['glc__D_e', 'lcts_e']
    tagged_molecules = [('cytoplasm', 'LacY')]
    initial_external = {
        'glc__D_e': 0.5,
        'lcts_e': 8.0,
    }
    parallel = True

    # agent configuration
    process_timestep = 10
    agents_config = [{
        'name': 'transport_metabolism',
        'type': TransportMetabolismExpression,
        'number': n_agents,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
            'metabolism': {
                'time_step': process_timestep},
            'transport': {
                'time_step': process_timestep},
            'expression': {
                'time_step': process_timestep},  # TODO -- is this causing big spikes?
            'division': {
                'time_step': process_timestep,
            },
        }}]
    make_agent_ids(agents_config)  # add agent_ids

    # initial agent state
    initial_agent_state = {
        'boundary': {
            'location': [bound/2 for bound in bounds],
            'external': initial_external
        }
    }

    # environment configuration
    media = get_minimal_media_iAF1260b(
        # scale concentration to ensure other nutrients remain in full supply
        scale_concentration=1000,
        override_initial=initial_external,
    )
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=60,
            bounds=bounds,
            n_bins=[bound*2 for bound in bounds],
            jitter_force=1e-2,
            depth=50.0,
            diffusion=1e-3,
            concentrations=media,
            keep_fields_emit=emit_fields,
            parallel=parallel,
        )
    }

    # make the experiment
    experiment_settings = {
        'experiment_name': 'transport_metabolism_environment',
        'description': 'glucose-lactose diauxic shifters are placed in a shallow environment with glucose and '
                       'lactose. They start off with no internal LacY and uptake only glucose, but LacY is '
                       'expressed upon depletion of glucose they begin to uptake lactose. Cells have an iAF1260b '
                       'BiGG metabolism, kinetic transport of glucose and lactose, and ode-based gene expression '
                       'of LacY',
        'total_time': total_time,
        'emit_step': emit_step,
        'emitter': {'type': 'database'},
    }
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
        'tagged_molecules': tagged_molecules,
    }
    plot_control(data, plot_config, out_dir)


# figure 5d
def lacy_expression(out_dir='out'):
    """ two experiments for ODE-based LacY expression

    These experiments use only the ode_expression process,
    and no transport from the resulting proteins
        * experiment 1: external glucose starts high and drops to 0.0
        * experiment 2: external glucose starts high and drops to 0.0 along with an increase in internal lactose.
    """

    total_time = 8000
    shift_time1 = int(total_time / 5)
    shift_time2 = int(4 * total_time / 5)

    # make the ode expression process
    lacy_expression = lacy_expression_config()
    process = ODE_expression(lacy_expression)

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

    # experiment 1 -- simulate and plot output
    settings1 = {'timeline': {'timeline': timeline1}}
    timeseries1 = simulate_process_in_experiment(process, settings1)
    plot_simulation_output(timeseries1, {}, out_dir, 'experiment_1')

    # experiment 2 -- simulate and plot output
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
        'complexes': flagella_expression_processes['complexation'].stoichiometry,
    }
    gene_network_plot(data, out_dir)


# figure 6b
def flagella_just_in_time(out_dir='out'):
    total_time = 2500

    # make the compartment
    compartment_config = {
        'transport': {'time_step': 60},
        'metabolism': {'time_step': 60},
    }
    compartment = FlagellaExpressionMetabolism(compartment_config)

    # get initial state
    initial_state = get_flagella_metabolism_initial_state()

    # run simulation with helper function simulate_compartment_in_experiment
    settings = {
        'total_time': total_time,
        'emit_step': 4000.0,
        'initial_state': initial_state,
    }
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # plot output as heat maps
    flagella_data = FlagellaChromosome()
    plot_config = {
        'name': 'flagella',
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'molecules',
        },
        'plot_ports': {
            'transcripts': list(flagella_data.chromosome_config['genes'].keys()),
            'proteins': flagella_data.complexation_monomer_ids + flagella_data.complexation_complex_ids,
            'molecules': list(nucleotides.values()) + list(amino_acids.values()),
        }
    }
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
    bounds = [17, 17]
    tagged_molecules = [('proteins', 'flagella')]
    emit_fields = ['glc__D_e']
    parallel = False

    # configure agents and environment
    agents_config = {
        'ids': ['flagella_metabolism'],
        'type': FlagellaExpressionMetabolism,
        'config': {
            'time_step': process_time_step,
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
            'transport': {},
        }
    }
    media = get_minimal_media_iAF1260b()
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=process_time_step,
            concentrations=media,
            bounds=bounds,
            depth=6000.0,
            keep_fields_emit=emit_fields,
            parallel=parallel,
        )
    }

    # get initial agent state
    initial_agent_state = get_flagella_metabolism_initial_state()
    initial_agent_state.update({'boundary': {'location': [8, 8]}})

    # use agent_environment_experiment to make the experiment
    experiment_settings = {
        'experiment_name': 'heterogeneous_flagella_experiment',
        'description': '..',
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

    # plot output
    plot_config = {
        'environment_config': environment_config,
        'tagged_molecules': tagged_molecules,
        'emit_fields': emit_fields,
        'topology_network': {
            'compartment': FlagellaExpressionMetabolism({}),
        },
    }
    plot_control(data, plot_config, out_dir)


# figure 7a
def variable_flagella(out_dir='out'):
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
            'time_step': time_step
        },
    }
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
    process_config = {
        'initial_ligand': initial_ligand,
    }
    receptor = ReceptorCluster(process_config)

    # run experiment with helper function simulate_process_in_experiment
    experiment_settings = {
        'timeline': {
            'timeline': timeline,
        },
    }
    timeseries = simulate_process_in_experiment(receptor, experiment_settings)

    # plot output
    plot_settings = {'aspect_ratio': 0.4}
    plot_receptor_output(timeseries, plot_settings, out_dir, 'pulse')


# figure 7c
def run_chemotaxis_transduction(out_dir='out'):
    total_time = 60
    time_step = 0.1
    n_flagella = 4
    ligand_id = 'MeAsp'
    initial_ligand = 1e-2

    # configure the compartment
    compartment_config = {
        'receptor': {
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand,
            'time_step': time_step,
        },
        'flagella': {
            'n_flagella': n_flagella,
            'time_step': time_step,
        },
    }
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
                'external': ('boundary', 'external')
            },
        },
    }
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
    total_time = 30
    emit_step = 5
    time_step = 0.001
    tumble_jitter = 4000  # TODO -- why tumble jitter?

    ligand_id = 'glc__D_e'
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
                'time_step': time_step,
            },
            'motor': {
                'tumble_jitter': tumble_jitter,
                'time_step': time_step,
            },
        },
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
                            'base': EXPONENTIAL_BASE,
                        },
                    },
                },
                'bounds': bounds,
            },
        },
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
        'emit_step': emit_step,
        'emitter': {'type': 'database'},
    }
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
    plot_config = {
        'bounds': field_config['bounds'],
        'field': field,
        'rotate_90': True,
    }
    plot_agent_trajectory(indexed_timeseries, plot_config, out_dir, 'trajectory')


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
    '7a': variable_flagella,
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
