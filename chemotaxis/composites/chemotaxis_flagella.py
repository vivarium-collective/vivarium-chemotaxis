"""
==============================
Chemotaxis Flagella Composites
==============================
"""

import os
import sys
import argparse

from vivarium.library.units import units
from vivarium.core.process import Generator
from vivarium.core.composition import (
    plot_compartment_topology,
    simulate_compartment_in_experiment,
    plot_simulation_output,
)

# processes
from chemotaxis.processes.chemoreceptor_cluster import (
    ReceptorCluster,
    get_brownian_ligand_timeline
)
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from cell.processes.transcription import Transcription
from cell.processes.translation import Translation
from cell.processes.degradation import RnaDegradation
from cell.processes.complexation import Complexation
from cell.processes.growth_protein import GrowthProtein
from cell.processes.derive_globals import DeriveGlobals
from cell.processes.ode_expression import ODE_expression, get_flagella_expression
from chemotaxis.processes.flagella_motor import FlagellaMotor
from chemotaxis.composites.flagella_expression import (
    get_flagella_expression_config,
    get_flagella_initial_state,
    plot_gene_expression_output,
)

# plots
from chemotaxis.plots.flagella_activity import plot_signal_transduction

# directories
from chemotaxis import COMPOSITE_OUT_DIR


NAME = 'chemotaxis_flagella'

DEFAULT_ENVIRONMENT_PORT = 'external'
DEFAULT_LIGAND = 'MeAsp'
DEFAULT_INITIAL_LIGAND = 1e-2


class ChemotaxisVariableFlagella(Generator):
    """
    A composite of the ReceptorCluster and FlagellaMotor processes
    """

    n_flagella = 5
    time_step = 0.01
    defaults = {
        'receptor': {
            'time_step': time_step,
            'ligand_id': DEFAULT_LIGAND,
            'initial_ligand': DEFAULT_INITIAL_LIGAND
        },
        'flagella': {
            'time_step': time_step,
            'n_flagella': n_flagella
        },
    }

    def __init__(self, config):
        super(ChemotaxisVariableFlagella, self).__init__(config)

    def generate_processes(self, config):
        receptor = ReceptorCluster(config['receptor'])
        flagella = FlagellaMotor(config['flagella'])

        return {
            'receptor': receptor,
            'flagella': flagella}

    def generate_topology(self, config):
        boundary_path = ('boundary',)
        external_path = boundary_path + ('external',)
        return {
            'receptor': {
                'external': external_path,
                'internal': ('internal',)},
            'flagella': {
                'internal': ('internal',),
                'membrane': ('membrane',),
                'internal_counts': ('proteins',),
                'flagella': ('flagella',),
                'boundary': boundary_path},
        }


class ChemotaxisODEExpressionFlagella(Generator):
    """
    A composite of the ReceptorCluster, FlagellaMotor, ODE_expression,
    and GrowthProtein processes.

    ODE_expression is configured to express flagella, which sets the number
    of flagella in FlagellaMotor.
    """

    ligand_id = 'MeAsp'
    initial_ligand = DEFAULT_INITIAL_LIGAND
    n_flagella = 5
    initial_mass = 1339.0 * units.fg
    growth_rate = 0.000275
    time_step = 0.01
    defaults = {
        'expression': get_flagella_expression(),
        'receptor': {
            'time_step': time_step,
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand
        },
        'flagella': {
            'time_step': time_step,
            'n_flagella': n_flagella
        },
        'growth': {
            'initial_mass': initial_mass,
            'growth_rate': growth_rate
        },
        'mass_deriver': {},
        'global_deriver': {},
        'boundary_path': ('boundary',),
        'external_path': ('boundary', 'external',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple(),
        'agent_id': 'chemotaxis_flagella'
    }

    def __init__(self, config=None):
        super(ChemotaxisODEExpressionFlagella, self).__init__(config)

    def generate_processes(self, config):
        # division config
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)

        return {
            'receptor': ReceptorCluster(config['receptor']),
            'flagella': FlagellaMotor(config['flagella']),
            'expression': ODE_expression(config['expression']),
            'growth': GrowthProtein(config['growth']),
            'division': MetaDivision(division_config),
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        external_path = config['external_path']
        agents_path = config['agents_path']

        return {
            'receptor': {
                'external': external_path,
                'internal': ('cell',)},

            'flagella': {
                'internal': ('internal',),
                'membrane': ('membrane',),
                'internal_counts': ('proteins',),
                'flagella': ('flagella',),
                'boundary': boundary_path},

            'expression': {
                'internal': ('internal_concentrations',),
                'counts': ('proteins',),
                'external': external_path,
                'global': boundary_path},

            'growth': {
                'internal': ('proteins',),
                'global': boundary_path},

            'division': {
                'global': boundary_path,
                'agents': agents_path},
        }


class ChemotaxisExpressionFlagella(Generator):
    """
    A composite of the ReceptorCluster, FlagellaMotor, GrowthProtein
    and stochastic gene expression processes: Translation, Transcription,
    Complexation, and Degradation.

    The gene expression processes express flagella based on sequence and
    transcription unit data.
    """

    ligand_id = 'MeAsp'
    initial_ligand = 0.1
    n_flagella = 5
    initial_mass = 1339.0 * units.fg
    growth_rate = 0.000275
    defaults = {
        'transcription': get_flagella_expression_config({})['transcription'],
        'translation': get_flagella_expression_config({})['translation'],
        'degradation': get_flagella_expression_config({})['degradation'],
        'complexation': get_flagella_expression_config({})['complexation'],
        'receptor': {
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand
        },
        'flagella': {
            'n_flagella': n_flagella
        },
        'growth': {
            'initial_mass': initial_mass,
            'growth_rate': growth_rate
        },
        'mass_deriver': {},
        'global_deriver': {},
        'boundary_path': ('boundary',),
        'external_path': ('boundary', 'external',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple(),
        'agent_id': 'chemotaxis_flagella'
    }

    def __init__(self, config=None):
        super(ChemotaxisExpressionFlagella, self).__init__(config)

    def generate_processes(self, config):
        # division config
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)

        return {
            'receptor': ReceptorCluster(config['receptor']),
            'flagella': FlagellaMotor(config['flagella']),
            'transcription': Transcription(config['transcription']),
            'translation': Translation(config['translation']),
            'degradation': RnaDegradation(config['degradation']),
            'complexation': Complexation(config['complexation']),
            'growth': GrowthProtein(config['growth']),
            'division': MetaDivision(division_config),
            'mass_deriver': TreeMass(config['mass_deriver']),
            'global_deriver': DeriveGlobals(config['global_deriver']),
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        external_path = config['external_path']
        agents_path = config['agents_path']

        return {

            'receptor': {
                'external': external_path,
                'internal': ('cell',)},

            'flagella': {
                'internal': ('internal',),
                'membrane': ('membrane',),
                'internal_counts': ('proteins',),
                'flagella': ('flagella',),
                'boundary': boundary_path},

            'transcription': {
                'chromosome': ('chromosome',),
                'molecules': ('internal',),
                'proteins': ('proteins',),
                'transcripts': ('transcripts',),
                'factors': ('concentrations',),
                'global': boundary_path},

            'translation': {
                'ribosomes': ('ribosomes',),
                'molecules': ('internal',),
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'concentrations': ('concentrations',),
                'global': boundary_path},

            'degradation': {
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'molecules': ('internal',),
                'global': boundary_path},

            'complexation': {
                'monomers': ('proteins',),
                'complexes': ('proteins',),
                'global': boundary_path},

            'growth': {
                'internal': ('aggregate_protein',),
                'global': boundary_path},

            'division': {
                'global': boundary_path,
                'agents': agents_path},

            'mass_deriver': {
                'global': boundary_path},

            'global_deriver': {
                'global': boundary_path},
        }


def get_baseline_config(n_flagella=5):
    return {
        'agents_path': ('agents',),  # Note -- should go two level up for experiments with environment
        'receptor': {
            'ligand_id': DEFAULT_LIGAND,
            'initial_ligand': DEFAULT_INITIAL_LIGAND,
        },
        'flagella': {
            'n_flagella': n_flagella}}


def print_growth(timeseries):
    volume_ts = timeseries['boundary']['volume']
    mass_ts = timeseries['boundary']['mass']
    print('volume growth: {}'.format(volume_ts[-1] / volume_ts[0]))
    print('mass growth: {}'.format(mass_ts[-1] / mass_ts[0]))


def plot_timeseries(timeseries, out_dir):
    # plot settings for the simulations
    plot_settings = {
        'max_rows': 30,
        'remove_zeros': True,
        'skip_ports': ['chromosome', 'ribosomes']
    }
    plot_simulation_output(
        timeseries,
        plot_settings,
        out_dir)


def test_ode_expression_chemotaxis(
        n_flagella=5,
        total_time=10,
        out_dir='out'
):
    # make the compartment
    config = get_baseline_config(n_flagella)
    compartment = ChemotaxisODEExpressionFlagella(config)

    # save the topology network
    plot_compartment_topology(compartment, {}, out_dir)

    # run experiment
    initial_state = {}
    timeline = get_brownian_ligand_timeline(total_time=total_time)
    experiment_settings = {
        'initial_state': initial_state,
        'timeline': {
            'timeline': timeline,
            'ports': {
                'external': ('boundary', 'external'),
                'global': ('boundary',),
            },
        },
    }
    timeseries = simulate_compartment_in_experiment(
        compartment,
        experiment_settings)

    print_growth(timeseries)

    # plot
    plot_timeseries(timeseries, out_dir)


def test_expression_chemotaxis(
        n_flagella=5,
        total_time=10,
        out_dir='out'
):
    # make the compartment
    config = get_baseline_config(n_flagella)
    compartment = ChemotaxisExpressionFlagella(config)

    # save the topology network
    plot_compartment_topology(compartment, {}, out_dir)

    # run experiment
    initial_state = get_flagella_initial_state({
        'molecules': 'internal'})
    timeline = get_brownian_ligand_timeline(total_time=total_time)
    experiment_settings = {
        'initial_state': initial_state,
        'timeline': {
            'timeline': timeline,
            'ports': {'external': ('boundary', 'external')}},
    }
    timeseries = simulate_compartment_in_experiment(
        compartment,
        experiment_settings)

    print_growth(timeseries)

    # plot
    plot_timeseries(timeseries, out_dir)

    # gene expression plot
    plot_config = {
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'internal'}}
    plot_gene_expression_output(
        timeseries,
        plot_config,
        out_dir)


def test_variable_chemotaxis(
        n_flagella=5,
        timeline=get_brownian_ligand_timeline(total_time=10),
        out_dir='out'
):
    # make the compartment
    config = get_baseline_config(n_flagella)
    compartment = ChemotaxisVariableFlagella(config)

    # save the topology network
    plot_compartment_topology(compartment, {}, out_dir)

    # run experiment
    initial_state = {}
    experiment_settings = {
        'initial_state': initial_state,
        'timeline': {
            'timeline': timeline,
            'ports': {'external': ('boundary', 'external')}},
    }
    timeseries = simulate_compartment_in_experiment(
        compartment,
        experiment_settings)

    # plot
    plot_timeseries(timeseries, out_dir)

    plot_signal_transduction(timeseries, out_dir)


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    make_dir(out_dir)

    parser = argparse.ArgumentParser(description='variable flagella')
    parser.add_argument('--variable', '-v', action='store_true', default=False)
    parser.add_argument('--ode', '-o', action='store_true', default=False)
    parser.add_argument('--expression', '-e', action='store_true', default=False)
    parser.add_argument('--flagella', '-f', type=int, default=5)
    args = parser.parse_args()
    no_args = (len(sys.argv) == 1)

    if args.variable or no_args:
        variable_out_dir = os.path.join(out_dir, 'variable')
        make_dir(variable_out_dir)
        test_variable_chemotaxis(
            n_flagella=args.flagella,
            timeline=get_brownian_ligand_timeline(total_time=90),
            out_dir=variable_out_dir)
    elif args.ode:
        # ODE flagella expression
        ode_out_dir = os.path.join(out_dir, 'ode_expression')
        make_dir(ode_out_dir)
        test_ode_expression_chemotaxis(
            n_flagella=args.flagella,
            # a cell cycle of 2520 sec is expected to express 8 flagella.
            # 2 flagella expected in 630 seconds.
            total_time=2520,
            out_dir=ode_out_dir)
    elif args.expression:
        expression_out_dir = os.path.join(out_dir, 'expression')
        make_dir(expression_out_dir)
        test_expression_chemotaxis(
            n_flagella=args.flagella,
            # a cell cycle of 2520 sec is expected to express 8 flagella.
            # 2 flagella expected in 630 seconds.
            total_time=700,
            out_dir=expression_out_dir)
