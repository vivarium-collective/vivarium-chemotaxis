"""
===========================
Chemotaxis Master Composite
===========================
"""

import os
import uuid

# core imports
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    plot_simulation_output,
    plot_compartment_topology,
)

# processes
from cell.processes.metabolism import (
    Metabolism,
    get_iAF1260b_config
)
from cell.processes.convenience_kinetics import ConvenienceKinetics
from cell.processes.transcription import Transcription
from cell.processes.translation import Translation
from cell.processes.degradation import RnaDegradation
from cell.processes.complexation import Complexation
from cell.processes.membrane_potential import MembranePotential
from cell.processes.division_volume import DivisionVolume
from chemotaxis.processes.chemoreceptor_cluster import ReceptorCluster
from chemotaxis.processes.flagella_motor import FlagellaMotor
from vivarium.processes.meta_division import MetaDivision

# composites
from chemotaxis.composites.flagella_expression import get_flagella_expression_config
from chemotaxis.composites.transport_metabolism import default_transport_config

# plots
from cell.plots.gene_expression import plot_gene_expression_output

# directories
from chemotaxis import COMPOSITE_OUT_DIR


NAME = 'chemotaxis_master'


def get_metabolism_config(time_step=1):
    metabolism_config = get_iAF1260b_config()
    metabolism_config.update({
        'time_step': time_step,
        'moma': False,
        'tolerance': {
            'EX_glc__D_e': [1.05, 1.0],
            'EX_lcts_e': [1.05, 1.0]}
        })
    return metabolism_config


class ChemotaxisMaster(Generator):

    defaults = {
        'dimensions_path': ('dimensions',),
        'fields_path': ('fields',),
        'boundary_path': ('boundary',),
        'agents_path': ('agents',),
        'daughter_path': tuple(),
        'transport': default_transport_config(),
        'metabolism': get_metabolism_config(10),
        'transcription': get_flagella_expression_config({})['transcription'],
        'translation': get_flagella_expression_config({})['translation'],
        'degradation': get_flagella_expression_config({})['degradation'],
        'complexation': get_flagella_expression_config({})['complexation'],
        'receptor': {'ligand': 'MeAsp'},
        'flagella': {'n_flagella': 5},
        'PMF': {},
        'division': {},
        'divide': True,
    }

    def __init__(self, config=None):
        super(ChemotaxisMaster, self).__init__(config)
        if 'agent_id' not in self.config:
            self.config['agent_id'] = str(uuid.uuid1())

    def generate_processes(self, config):
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']

        # Transport
        transport = ConvenienceKinetics(config['transport'])

        # Metabolism
        # add target fluxes from transport
        target_fluxes = transport.kinetic_rate_laws.reaction_ids
        config['metabolism']['constrained_reaction_ids'] = target_fluxes
        metabolism = Metabolism(config['metabolism'])

        # flagella expression
        transcription = Transcription(config['transcription'])
        translation = Translation(config['translation'])
        degradation = RnaDegradation(config['degradation'])
        complexation = Complexation(config['complexation'])

        # chemotaxis -- flagella activity, receptor activity, and PMF
        receptor = ReceptorCluster(config['receptor'])
        flagella = FlagellaMotor(config['flagella'])
        PMF = MembranePotential(config['PMF'])

        # Division
        # get initial volume from metabolism
        config['division']['initial_state'] = metabolism.initial_state
        division_condition = DivisionVolume(config['division'])

        processes = {
            'metabolism': metabolism,
            'transport': transport,
            'transcription': transcription,
            'translation': translation,
            'degradation': degradation,
            'complexation': complexation,
            'receptor': receptor,
            'flagella': flagella,
            'PMF': PMF,
            'division': division_condition,
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
                'dimensions': dimensions_path
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
            }
        }
        if config['divide']:
            topology.update({
                'meta_division': {
                    'global': boundary_path,
                    'agents': agents_path,
                }})
        return topology


def run_chemotaxis_master(out_dir):
    total_time = 10

    # make the compartment
    compartment = ChemotaxisMaster({'agent_id': '0'})

    # save the topology network
    settings = {'show_ports': True}
    plot_compartment_topology(
        compartment,
        settings,
        out_dir)

    # run an experinet
    settings = {
        'timestep': 1,
        'total_time': total_time}
    timeseries = simulate_compartment_in_experiment(compartment, settings)

    volume_ts = timeseries['boundary']['volume']
    print('growth: {}'.format(volume_ts[-1]/volume_ts[0]))

    # plots
    # simulation output
    plot_settings = {
        'max_rows': 40,
        'remove_zeros': True,
        'skip_ports': ['reactions', 'prior_state', 'null']}
    plot_simulation_output(timeseries, plot_settings, out_dir)

    # gene expression plot
    gene_exp_plot_config = {
        'name': 'flagella_expression',
        'ports': {
            'transcripts': 'transcripts',
            'proteins': 'proteins',
            'molecules': 'internal'}}
    plot_gene_expression_output(
        timeseries,
        gene_exp_plot_config,
        out_dir)

def test_chemotaxis_master(total_time=5):
    compartment = ChemotaxisMaster({})
    settings = {
        'timestep': 1,
        'total_time': total_time}
    return simulate_compartment_in_experiment(compartment, settings)


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_chemotaxis_master(out_dir)
