"""
=============================
Chemotaxis Paper Experiments
=============================

Includes functions for configuring, running, and plotting all experiments reported in the paper:
    Agmon, E. and Spangler, R.K., "A Multi-Scale Approach to Modeling E. coli Chemotaxis"

These experiments can be triggered from the command line by entering the figure number.
Available experiments include: '3b', '5a', '5b', '5c', '6a', '6b', '6c', '7a', '7b', '7c', '7d'.

```
$ python chemotaxis/experiments/paper_experiments.py 7a
```

"""

import numpy as np

# vivarium-core imports
from vivarium.library.units import units
from vivarium.core.composition import (
    simulate_process_in_experiment,
    simulate_compartment_in_experiment,
    agent_environment_experiment,
    make_agent_ids,
)
from vivarium.core.emitter import time_indexed_timeseries_from_data

# experiment workflow
from chemotaxis.experiments.control import (
    control,
    plot_control,
)

# vivarium-cell imports
from cell.processes.metabolism import (
    Metabolism,
    get_minimal_media_iAF1260b,
    get_iAF1260b_config,
)
from cell.processes.static_field import make_field
from cell.composites.lattice import (
    Lattice,
    make_lattice_config,
)
from cell.composites.static_lattice import StaticLattice
from cell.composites.growth_division import GrowthDivision

# chemotaxis processes
from chemotaxis.processes.flagella_motor import (
    FlagellaMotor,
    get_chemoreceptor_activity_timeline
)
from chemotaxis.processes.chemoreceptor_cluster import (
    ReceptorCluster,
    get_pulse_timeline,
    get_brownian_ligand_timeline,
)

# chemotaxis composites
from chemotaxis.composites.chemotaxis_minimal import ChemotaxisMinimal
from chemotaxis.composites.flagella_expression import (
    FlagellaExpressionMetabolism,
    get_flagella_metabolism_initial_state,
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
            # parallel=True,
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

    # configure metabolism process iAF1260b BiGG model
    process_config = get_iAF1260b_config()
    metabolism = Metabolism(process_config)

    # get default minimal external concentrations
    external_concentrations = metabolism.initial_state['external']

    # run simulation with the helper function simulate_process_in_experiment
    sim_settings = {
        'environment': {
            'volume': env_volume,
            'concentrations': external_concentrations,
        },
        'total_time': total_time,
    }
    timeseries = simulate_process_in_experiment(metabolism, sim_settings)

    # plot
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
    total_time = 3000
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
        'expression': {'time_step': 1},
        'divide': False,
    }
    compartment = TransportMetabolismExpression(compartment_config)

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
                'global': ('boundary',),
            },
        },
        'total_time': total_time,
    }

    # run simulation
    timeseries = simulate_compartment_in_experiment(compartment, sim_settings)

    # plot
    plot_settings = {
        'internal_path': ('cytoplasm',),
        'external_path': ('boundary', 'external'),
        'global_path': ('boundary',),
        'environment_volume': environment_volume,
        'aspect_ratio': 0.7,
    }
    plot_glc_lcts_environment(timeseries, plot_settings, out_dir)


# figure 5c
def transport_metabolism_environment(out_dir='out'):
    n_agents = 2
    total_time = 10000
    emit_step = 100
    bounds = [30, 30]
    emit_fields = ['glc__D_e', 'lcts_e']
    tagged_molecules = [('cytoplasm', 'LacY')]
    initial_external = {
        'glc__D_e': 0.2,
        'lcts_e': 8.0,
    }

    # agent configuration
    agents_config = [{
        'name': 'transport_metabolism',
        'type': TransportMetabolismExpression,
        'number': n_agents,
        'config': {
            'agents_path': ('..', '..', 'agents'),
            'fields_path': ('..', '..', 'fields'),
            'dimensions_path': ('..', '..', 'dimensions'),
            'metabolism': {
                'time_step': 10},
            'transport': {
                'time_step': 10},
            'expression': {
                'time_step': 1},
            'division': {
                'time_step': 10,
                # 'division_volume': 1.3 * units.fL,
            },
        }
    }]
    # add agent_ids
    make_agent_ids(agents_config)

    # TODO -- get initial agent_state from transport
    initial_agent_state = {
        'boundary': {
            'location': [8, 8],
            'external': initial_external
        }
    }

    # environment configuration
    media = get_minimal_media_iAF1260b(
        scale_concentration=100,
        override_initial=initial_external,
    )
    environment_config = {
        'type': Lattice,
        'config': make_lattice_config(
            time_step=10,
            bounds=bounds,
            n_bins=[40, 40],
            depth=50.0,
            diffusion=1e-4,
            concentrations=media,
            keep_fields_emit=emit_fields,
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
    }
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    # run simulation
    experiment.update(total_time)
    data = experiment.emitter.get_data()

    # plot output
    plot_config = {
        'environment_config': environment_config,
        'emit_fields': emit_fields,
        'tagged_molecules': tagged_molecules,
    }
    plot_control(data, plot_config, out_dir)


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

    # make the compartment
    compartment_config = {}
    compartment = FlagellaExpressionMetabolism(compartment_config)

    # get the initial state
    initial_state = get_flagella_metabolism_initial_state()

    # run simulation
    settings = {
        # a cell cycle of 2520 sec is expected to express 8 flagella.
        # 2 flagella expected in approximately 630 seconds.
        'total_time': 500,
        'emit_step': 10.0,
        'verbose': True,
        'initial_state': initial_state,
    }
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    # plot output
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


# figure 6c
def run_heterogeneous_flagella_experiment(out_dir='out'):

    total_time = 15000
    emit_step = 120
    process_time_step = 60
    bounds = [17, 17]
    tagged_molecules = [('proteins', 'flagella')]
    emit_fields = ['glc__D_e']

    # configurations
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
            keep_fields_emit=emit_fields)
    }

    # initial state
    initial_agent_state = get_flagella_metabolism_initial_state()
    initial_agent_state.update({'boundary': {'location': [8, 8]}})

    # use agent_environment_experiment to make the experiment
    experiment_settings = {
        'experiment_name': 'heterogeneous_flagella_experiment',
        'description': '..',
        'total_time': total_time,
        'emit_step': emit_step,
    }
    experiment = agent_environment_experiment(
        agents_config=agents_config,
        environment_config=environment_config,
        initial_agent_state=initial_agent_state,
        settings=experiment_settings)

    # run the experiment
    experiment.update(total_time)
    data = experiment.emitter.get_data()

    # plots
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

    # make timeline with varying chemoreceptor activity and flagella counts
    timeline = get_chemoreceptor_activity_timeline(
        total_time=total_time,
        time_step=time_step,
        rate=2.0)
    timeline_flagella = [
        (20, {('internal_counts', 'flagella'): initial_flagella + 1}),
        (40, {('internal_counts', 'flagella'): initial_flagella + 2}),
        (60, {('internal_counts', 'flagella'): initial_flagella + 3})]
    timeline.extend(timeline_flagella)

    # run simulation
    process_config = {'n_flagella': initial_flagella}
    process = FlagellaMotor(process_config)
    settings = {
        'return_raw_data': True,
        'timeline': {
            'timeline': timeline,
            'time_step': time_step
        },
    }
    data = simulate_process_in_experiment(process, settings)

    # plot
    plot_settings = {'aspect_ratio': 0.4}
    plot_activity(data, plot_settings, out_dir)


# figure 7b
def run_chemoreceptor_pulse(out_dir='out'):
    ligand = 'MeAsp'
    timeline = get_pulse_timeline(ligand=ligand)

    # initialize process
    initial_ligand = timeline[0][1][('external', ligand)]
    process_config = {
        'initial_ligand': initial_ligand,
    }
    receptor = ReceptorCluster(process_config)

    # run experiment
    experiment_settings = {
        'timeline': {
            'timeline': timeline,
        },
    }
    timeseries = simulate_process_in_experiment(receptor, experiment_settings)

    # plot
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

    # run experiment
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

    # plot
    plot_config = {
        'ligand_id': ligand_id,
        'aspect_ratio': 0.4}
    plot_signal_transduction(timeseries, plot_config, out_dir)



# helper functions for chemotaxis
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
            # 'angle': np.random.uniform(0, 2 * PI),
            # 'volume': volume,
            'length': length,
            'width': width,
            'mass': 1339 * units.fg,
            # 'thrust': 0,
            # 'torque': 0,
        }
    }

def agent_body_config(config):
    agent_ids = config['agent_ids']
    agent_config = {
        agent_id: single_agent_config(config)
        for agent_id in agent_ids}
    return {'agents': agent_config}


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
    '6a': flagella_expression_network,
    '6b': flagella_just_in_time,
    '6c': run_heterogeneous_flagella_experiment,
    '7a': variable_flagella,
    '7b': run_chemoreceptor_pulse,
    '7c': run_chemotaxis_transduction,
    '7d': run_chemotaxis_experiment,
    '5': ['5a', '5b', '5c'],
    '6': ['6a', '6b', '6c'],
    '7': ['7a', '7b', '7c', '7d'],
    'all': [
        '3b',
        '5a', '5b', '5c',
        '6a', '6b', '6c',
        '7a', '7b', '7c', '7d'],
}


if __name__ == '__main__':
    control(experiments_library)
