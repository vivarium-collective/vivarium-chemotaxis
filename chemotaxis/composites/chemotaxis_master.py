"""
===========================
Chemotaxis Master Composite
===========================
"""

import os
import argparse
import uuid

# core imports
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    plot_simulation_output,
    plot_compartment_topology,
)
from vivarium.library.dict_utils import deep_merge

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium.processes.tree_mass import TreeMass
from cell.processes.metabolism import Metabolism
from cell.processes.metabolism import get_iAF1260b_config as get_iAF1260b_path_config
from cell.processes.convenience_kinetics import ConvenienceKinetics
from cell.processes.transcription import Transcription
from cell.processes.translation import Translation
from cell.processes.degradation import RnaDegradation
from cell.processes.complexation import Complexation
from cell.processes.division_volume import DivisionVolume
from chemotaxis.processes.chemoreceptor_cluster import ReceptorCluster, get_brownian_ligand_timeline
from chemotaxis.processes.flagella_motor import FlagellaMotor
from chemotaxis.processes.membrane_potential import MembranePotential

# composites
from chemotaxis.composites.flagella_expression import (
    get_flagella_expression_config,
    get_flagella_metabolism_initial_state,
)
from chemotaxis.composites.transport_metabolism import (
    get_glucose_lactose_transport_config,
    get_iAF1260b_config,
)

# plots
from cell.plots.gene_expression import plot_gene_expression_output

# directories
from chemotaxis import COMPOSITE_OUT_DIR


NAME = 'chemotaxis_master'


def get_chemotaxis_master_schema_override():
    """ schema_override method to selectively turn off metabolic state emits """
    config = get_iAF1260b_path_config()
    metabolism = Metabolism(config)
    return {
        'metabolism': {
            'internal': {
                mol_id: {
                    '_emit': False,
                } for mol_id in metabolism.initial_state['internal'].keys()
            },
            'external': {
                mol_id: {
                    '_emit': False,
                } for mol_id in metabolism.initial_state['external'].keys()
            },
        }
    }


class ChemotaxisMaster(Generator):
    """ Chemotaxis Master Composite

     The most complete chemotaxis agent in the vivarium-chemotaxis repository
     """

    name = NAME
    defaults = {
        'dimensions_path': ('dimensions',),
        'fields_path': ('fields',),
        'boundary_path': ('boundary',),
        'agents_path': ('agents',),
        'daughter_path': tuple(),
        'transport': get_glucose_lactose_transport_config(),
        'metabolism': get_iAF1260b_config(),
        'receptor': {'ligand_id': 'MeAsp'},
        'flagella': {'n_flagella': 4},
        'PMF': {},
        'mass_deriver': {},
        'chromosome': {},
        'division': {},
        'divide': True,
        '_schema': get_chemotaxis_master_schema_override()
    }

    def __init__(self, config=None):
        if config is None:
            config = {}
        # make expression process configs with get_flagella_expression_config
        chromosome_config = config.get('chromosome', self.defaults['chromosome'])
        expression_config = get_flagella_expression_config(chromosome_config)
        config = deep_merge(dict(config), expression_config)
        super(ChemotaxisMaster, self).__init__(config)

        if 'agent_id' not in self.config:
            self.config['agent_id'] = str(uuid.uuid1())

    def generate_processes(self, config):
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']

        # Transport and Metabolism
        # get target fluxes from transport and add to metabolism
        transport = ConvenienceKinetics(config['transport'])
        target_fluxes = transport.kinetic_rate_laws.reaction_ids
        metabolism_config = config['metabolism']
        metabolism_config.update({'constrained_reaction_ids': target_fluxes})
        metabolism = Metabolism(metabolism_config)

        processes = {
            'transport': transport,
            'metabolism': metabolism,
            'receptor': ReceptorCluster(config['receptor']),
            'flagella': FlagellaMotor(config['flagella']),
            'PMF': MembranePotential(config['PMF']),
            'transcription': Transcription(config['transcription']),
            'translation': Translation(config['translation']),
            'degradation': RnaDegradation(config['degradation']),
            'complexation': Complexation(config['complexation']),
            'division': DivisionVolume(config['division']),
            'mass_deriver': TreeMass(config['mass_deriver']),
        }
        # divide process set to true, add meta-division processes
        if config['divide']:
            meta_division_config = dict(
                {},
                daughter_path=daughter_path,
                agent_id=agent_id,
                compartment=self)
            meta_division = MetaDivision(meta_division_config)
            processes['meta_division'] = meta_division

        return processes

    def generate_topology(self, config):
        dimensions_path = config['dimensions_path']
        fields_path = config['fields_path']
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        external_path = boundary_path + ('external',)
        topology = {
            'transport': {
                'internal': ('internal',),
                'external': external_path,
                'fields': ('null',),  # metabolism's exchange is used
                'fluxes': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
            },
            'metabolism': {
                'internal': ('internal',),
                'external': external_path,
                'reactions': ('reactions',),
                'fields': fields_path,
                'flux_bounds': ('flux_bounds',),
                'global': boundary_path,
                'dimensions': dimensions_path,
            },
            'transcription': {
                'chromosome': ('chromosome',),
                'molecules': ('internal',),
                'proteins': ('proteins',),
                'transcripts': ('transcripts',),
                'factors': ('concentrations',),
                'global': boundary_path,
            },
            'translation': {
                'ribosomes': ('ribosomes',),
                'molecules': ('internal',),
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'concentrations': ('concentrations',),
                'global': boundary_path,
            },
            'degradation': {
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'molecules': ('internal',),
                'global': boundary_path,
            },
            'complexation': {
                'monomers': ('proteins',),
                'complexes': ('proteins',),
                'global': boundary_path,
            },
            'receptor': {
                'external': external_path,
                'internal': ('internal',),
            },
            'flagella': {
                'internal': ('internal',),
                'membrane': ('membrane',),
                'internal_counts': ('proteins',),
                'flagella': ('flagella',),
                'boundary': boundary_path,
            },
            'PMF': {
                'external': external_path,
                'membrane': ('membrane',),
                'internal': ('internal',),
            },
            'division': {
                'global': boundary_path,
            },
            'mass_deriver': {
                'global': boundary_path,
            },
        }
        if config['divide']:
            topology.update({
                'meta_division': {
                    'global': boundary_path,
                    'agents': agents_path,
                }})
        return topology


def save_master_chemotaxis_topology(out_dir):
    compartment = ChemotaxisMaster({})
    settings = {'show_ports': True}
    plot_compartment_topology(
        compartment,
        settings,
        out_dir)


def test_chemotaxis_master(
    total_time=5,
    emit_step=1,
    chemotaxis_timestep=0.01,
):
    # get initial state
    initial_state = get_flagella_metabolism_initial_state()

    # configure the compartment
    compartment = ChemotaxisMaster({
        'receptor': {'time_step': chemotaxis_timestep},
        'flagella': {'time_step': chemotaxis_timestep},
        'agent_id': '0',
        'divide': False})

    # simulate
    timeline = get_brownian_ligand_timeline(
        total_time=total_time,
        timestep=chemotaxis_timestep)
    settings = {
        'emit_step': emit_step,
        'initial_state': initial_state,
        'timeline': {
            'timeline': timeline,
            'ports': {
                'external': ('boundary', 'external')}}}
    return simulate_compartment_in_experiment(compartment, settings)


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='flagella expression')
    parser.add_argument('--topology', '-t', action='store_true', default=False, )
    args = parser.parse_args()

    if args.topology:
        save_master_chemotaxis_topology(out_dir)
    else:
        total_time = 500

        # run sim
        timeseries = test_chemotaxis_master(
            total_time=total_time,
            chemotaxis_timestep=1,  # 0.01 is the appropriate timescale
        )
        volume_ts = timeseries['boundary']['volume']
        print('growth: {}'.format(volume_ts[-1]/volume_ts[0]))

        # plot output
        plot_settings = {
            'max_rows': 40,
            'remove_zeros': True,
            'skip_ports': ['reactions', 'prior_state', 'null']}
        plot_simulation_output(timeseries, plot_settings, out_dir)

        # gene expression plot
        gene_exp_plot_config = {
            'ports': {
                'transcripts': 'transcripts',
                'proteins': 'proteins',
                'molecules': 'internal'}}
        plot_gene_expression_output(
            timeseries,
            gene_exp_plot_config,
            out_dir)
